package dev.localrag.ollama

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.InetSocketAddress
import java.net.URI
import java.nio.charset.StandardCharsets.UTF_8
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class OllamaRerankerTest {
    @Test
    fun `reranker sends only question and candidate text to fixed local model`() {
        val requests = CopyOnWriteArrayList<JsonObject>()
        val server = FakeOllama { exchange ->
            requests += Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
            respond(exchange, 200, """{"choices":[{"message":{"content":"{\"ranked_indices\":[1,0]}"}}]}""")
        }
        server.use {
            val question = "Where is the calibration limit?"
            val candidates = listOf("Calibration limit = 19.", "Unrelated note.")

            assertEquals(listOf(1, 0), OllamaReranker(OllamaApi(it.uri)).rerank(question, candidates))

            val request = requests.single()
            assertEquals(LOCAL_RERANK_MODEL, request.getValue("model").jsonPrimitive.content)
            assertEquals(0, request.getValue("temperature").jsonPrimitive.content.toInt())
            assertFalse(request.containsKey("Authorization"))
            val messages = request.getValue("messages").jsonArray.map { value -> value.jsonObject }
            assertEquals(listOf("system", "user"), messages.map { it.getValue("role").jsonPrimitive.content })
            val payload = Json.parseToJsonElement(messages.last().getValue("content").jsonPrimitive.content).jsonObject
            assertEquals(setOf("question", "candidates"), payload.keys)
            assertEquals(question, payload.getValue("question").jsonPrimitive.content)
            assertEquals(
                candidates,
                payload.getValue("candidates").jsonArray.map { it.jsonObject.getValue("text").jsonPrimitive.content },
            )
        }
    }

    @Test
    fun `empty candidates do not call Ollama and malformed ranking fails locally`() {
        val requests = AtomicInteger()
        val server = FakeOllama { exchange ->
            requests.incrementAndGet()
            respond(exchange, 200, """{"choices":[{"message":{"content":"{\"ranked_indices\":[0,0]}"}}]}""")
        }
        server.use {
            val reranker = OllamaReranker(OllamaApi(it.uri))
            assertEquals(emptyList(), reranker.rerank("question", emptyList()))
            assertEquals(0, requests.get())

            val failure = assertFailsWith<LocalModelException> {
                reranker.rerank("SENSITIVE_QUERY", listOf("candidate one", "candidate two"))
            }

            assertContains(failure.message.orEmpty(), "reranker")
            assertFalse("SENSITIVE_QUERY" in failure.message.orEmpty())
            assertEquals(1, requests.get())
        }
    }

    private class FakeOllama(handler: (HttpExchange) -> Unit) : AutoCloseable {
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeOllama.executor
            createContext("/v1/chat/completions", handler)
            start()
        }
        val uri: URI get() = URI("http://127.0.0.1:${server.address.port}")

        override fun close() {
            server.stop(0)
            executor.shutdownNow()
        }
    }

    companion object {
        private fun respond(exchange: HttpExchange, status: Int, body: String) {
            val bytes = body.toByteArray(UTF_8)
            exchange.responseHeaders.set("Content-Type", "application/json; charset=utf-8")
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
    }
}
