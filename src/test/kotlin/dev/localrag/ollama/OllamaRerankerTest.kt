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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
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
            respond(exchange, 200, """{"message":{"content":"{\"ranked_indices\":[1,0]}"},"done":true}""")
        }
        server.use {
            val question = "Where is the calibration limit?"
            val candidates = listOf("Calibration limit = 19.", "Unrelated note.")

            assertEquals(listOf(1, 0), OllamaReranker(OllamaApi(it.uri)).rerank(question, candidates))

            val request = requests.single()
            assertEquals(LOCAL_RERANK_MODEL, request.getValue("model").jsonPrimitive.content)
            assertEquals(0, request.getValue("options").jsonObject.getValue("temperature").jsonPrimitive.content.toInt())
            assertFalse(request.containsKey("Authorization"))
            assertEquals(
                Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
                request.getValue("options").jsonObject.getValue("num_thread").jsonPrimitive.content.toInt(),
            )
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
    fun `Ollama embeddings and reranking share one inference permit`() {
        val activeRequests = AtomicInteger()
        val maximumConcurrentRequests = AtomicInteger()
        val requests = CopyOnWriteArrayList<JsonObject>()
        val firstRequestEntered = CountDownLatch(1)
        val secondRequestEntered = CountDownLatch(1)
        val releaseFirstRequest = CountDownLatch(1)
        val server = FakeOllama { exchange ->
            if (exchange.requestURI.path == "/api/tags") {
                respond(exchange, 200, """{"models":[{"name":"qwen2.5:7b"}]}""")
            } else {
                val active = activeRequests.incrementAndGet()
                requests += Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                maximumConcurrentRequests.accumulateAndGet(active, ::maxOf)
                if (firstRequestEntered.count == 1L) firstRequestEntered.countDown()
                else secondRequestEntered.countDown()
                try {
                    releaseFirstRequest.await(3, TimeUnit.SECONDS)
                    if (exchange.requestURI.path == "/api/embed") {
                        respond(exchange, 200, """{"embeddings":[[0.1,0.2]]}""")
                    } else {
                        respond(exchange, 200, """{"message":{"content":"{\"ranked_indices\":[0]}"},"done":true}""")
                    }
                } finally {
                    activeRequests.decrementAndGet()
                }
            }
        }
        val callers = Executors.newFixedThreadPool(3)
        server.use {
            try {
                val embedding = callers.submit<List<List<Float>>> { OllamaApi(it.uri).embed(EMBEDDING_MODEL, listOf("query")) }
                assertEquals(true, firstRequestEntered.await(2, TimeUnit.SECONDS))
                val reranking = callers.submit<String> {
                    OllamaApi(it.uri).chatCompletion("qwen2.5:3b", listOf("user" to "question"))
                }
                val models = callers.submit<Set<String>> { OllamaApi(it.uri).availableModels() }
                assertEquals(setOf("qwen2.5:7b"), models.get(500, TimeUnit.MILLISECONDS))
                assertEquals(false, secondRequestEntered.await(100, TimeUnit.MILLISECONDS))
                releaseFirstRequest.countDown()

                assertEquals(listOf(listOf(0.1f, 0.2f)), embedding.get(2, TimeUnit.SECONDS))
                assertEquals("""{"ranked_indices":[0]}""", reranking.get(2, TimeUnit.SECONDS))
                assertEquals(1, maximumConcurrentRequests.get())
                val embeddingRequest = requests.single { it.containsKey("input") }
                assertEquals(
                    Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
                    embeddingRequest.getValue("options").jsonObject.getValue("num_thread").jsonPrimitive.content.toInt(),
                )
            } finally {
                releaseFirstRequest.countDown()
                callers.shutdownNow()
            }
 
        }
    }

    @Test
    fun `empty candidates do not call Ollama and malformed ranking fails locally`() {
        val requests = AtomicInteger()
        val server = FakeOllama { exchange ->
            requests.incrementAndGet()
            respond(exchange, 200, """{"message":{"content":"{\"ranked_indices\":[0,0]}"},"done":true}""")
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
            createContext("/api/chat", handler)
            createContext("/api/embed", handler)
            createContext("/api/tags", handler)
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
