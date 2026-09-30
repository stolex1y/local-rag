package dev.localrag.ollama

import dev.localrag.domain.EmbeddingPort
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.IOException
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

const val EMBEDDING_MODEL = "embeddinggemma:300m"

class LocalModelException(message: String) : RuntimeException(message)

class OllamaApi(
    baseUri: URI = URI("http://127.0.0.1:11434"),
    private val http: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(5))
        .followRedirects(HttpClient.Redirect.NEVER)
        .build(),
) {
    private val root = validateLoopback(baseUri)

    fun embed(model: String, texts: List<String>): List<List<Float>> {
        if (texts.isEmpty()) return emptyList()
        require(texts.size <= 16) { "Embedding batches are limited to 16 inputs." }
        val payload = buildJsonObject {
            put("model", model)
            put("input", JsonArray(texts.map(::JsonPrimitive)))
            put("truncate", false)
        }
        val response = post("/api/embed", payload.toString())
        val vectors = try {
            JsonObject(kotlinx.serialization.json.Json.parseToJsonElement(response).jsonObject).getValue("embeddings").jsonArray
                .map { vector -> vector.jsonArray.map { it.jsonPrimitive.content.toFloat() } }
        } catch (_: Exception) {
            throw LocalModelException("Локальная embedding-модель вернула ответ неожиданного формата.")
        }
        if (vectors.size != texts.size || vectors.isEmpty() || vectors.any { vector ->
                vector.isEmpty() || vector.any { !it.isFinite() } || vector.size != vectors.first().size
            }) {
            throw LocalModelException("Локальная embedding-модель вернула неполные или несовместимые векторы.")
        }
        return vectors
    }


    fun availableModels(): Set<String> {
        val response = get("/api/tags")
        return try {
            kotlinx.serialization.json.Json.parseToJsonElement(response).jsonObject["models"]!!.jsonArray
                .mapNotNull { it.jsonObject["name"]?.jsonPrimitive?.content }
                .toSet()
        } catch (_: Exception) {
            throw LocalModelException("Локальный Ollama API вернул список моделей неожиданного формата.")
        }
    }

    private fun get(path: String): String = request(path, null)

    private fun post(path: String, body: String): String = request(path, body)

    private fun request(path: String, body: String?): String {
        val builder = HttpRequest.newBuilder(root.resolve(path))
            .timeout(Duration.ofMinutes(15))
            .header("Accept", "application/json")
        if (body == null) {
            builder.GET()
        } else {
            builder.header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofString(body))
        }
        val response = try {
            http.send(builder.build(), HttpResponse.BodyHandlers.ofString())
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            throw LocalModelException("Обращение к локальной модели прервано.")
        } catch (_: IOException) {
            throw LocalModelException("Не удалось связаться с Ollama на локальном компьютере. Проверьте, что Ollama запущен и выбранные модели загружены.")
        }
        if (response.statusCode() !in 200..299) {
            throw LocalModelException("Ollama вернул HTTP ${response.statusCode()}. Проверьте локальную модель и журнал Ollama.")
        }
        return response.body()
    }

    private fun validateLoopback(uri: URI): URI {
        require(uri.scheme == "http" && uri.host == "127.0.0.1" && uri.userInfo == null && uri.query == null && uri.fragment == null) {
            "Ollama is restricted to an HTTP service bound to 127.0.0.1."
        }
        require(uri.port in -1..65535 && uri.port != 0) { "Ollama port is invalid." }
        return if (uri.path.isNullOrEmpty() || uri.path == "/") URI("${uri.scheme}://${uri.host}${if (uri.port == -1) "" else ":${uri.port}"}/") else uri
    }
}

class OllamaEmbeddingPort(
    private val api: OllamaApi,
    override val modelName: String = EMBEDDING_MODEL,
) : EmbeddingPort {
    override fun embed(texts: List<String>): List<List<Float>> = api.embed(modelName, texts)
}
