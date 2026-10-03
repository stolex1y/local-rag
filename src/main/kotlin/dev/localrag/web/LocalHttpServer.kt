package dev.localrag.web

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import dev.localrag.app.ApiException
import dev.localrag.app.ApplicationService
import dev.localrag.app.IndexNotReadyException
import dev.localrag.app.JobBusyException
import dev.localrag.app.JobNotFoundException
import dev.localrag.benchmark.BenchmarkQuestion
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.ModelSelection
import dev.localrag.generation.CloudModelException
import dev.localrag.ollama.LocalModelException
import dev.localrag.source.ImportInProgressException
import dev.localrag.source.ImportNotFoundException
import dev.localrag.source.SourceCatalog
import dev.localrag.source.SourceNotFoundException
import dev.localrag.source.SourceBusyException
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors

@Serializable
private data class ErrorPayload(val error: ErrorDetail)

@Serializable
private data class ErrorDetail(val code: String, val message: String)

@Serializable
private data class BeginImportRequest(val files: List<dev.localrag.source.ImportFileMetadata>)

@Serializable
private data class DeleteRequest(val confirm: Boolean)

@Serializable
private data class ChatTurnHttpRequest(
    val turnId: String,
    val question: String,
    val strategy: ChunkStrategy = ChunkStrategy.STRUCTURAL,
    val topK: Int? = null,
)

@Serializable
private data class ChatMemoryFactRequest(val content: String)

@Serializable
private data class IndexRequest(
    val confirm: Boolean,
    val sourceIds: List<String>,
    val totalBytes: Long,
)

@Serializable
private data class QueryRequest(
    val question: String,
    val strategy: ChunkStrategy,
    val topK: Int? = null,
)

@Serializable
private data class QuestionsRequest(val questions: List<BenchmarkQuestion>)

@Serializable
private data class ReviewRequest(
    val questionId: String,
    val strategy: ChunkStrategy,
    val baselineRating: String,
    val ragRating: String,
    val note: String? = null,
)
@Serializable
private data class ModelSelectionRequest(val providerId: String, val modelId: String)

@Serializable
private data class SavedResponse(val saved: Boolean)

class LocalHttpServer(
    private val application: ApplicationService,
    port: Int = 8765,
) : AutoCloseable {
    private val server = HttpServer.create(InetSocketAddress(InetAddress.getByName("127.0.0.1"), port), 0)
    private val workers = Executors.newFixedThreadPool(4) { task ->
        Thread(task, "local-rag-http").apply { isDaemon = true }
    }
    private val json = Json { encodeDefaults = true }

    init {
        server.executor = workers
        server.createContext("/") { exchange -> handle(exchange) }
    }

    val port: Int get() = server.address.port

    fun start() = server.start()

    private fun handle(exchange: HttpExchange) {
        try {
            if (!isSafeLocalRequest(exchange)) {
                sendError(exchange, 403, "local_origin_required", "Запрос разрешён только с локального адреса приложения.")
                return
            }
            val path = exchange.requestURI.path
            if (path.startsWith("/api/")) handleApi(exchange, path)
            else handleStatic(exchange, path)
        } catch (error: ApiException) {
            sendError(exchange, error.status, error.code, error.message)
        } catch (error: JobBusyException) {
            sendError(exchange, 409, "job_busy", error.message ?: "Другое локальное задание ещё выполняется.")
        } catch (error: JobNotFoundException) {
            sendError(exchange, 404, "job_not_found", error.message ?: "Задание не найдено.")
        } catch (error: IndexNotReadyException) {
            sendError(exchange, 409, "index_not_ready", error.message ?: "Индекс недоступен.")
        } catch (error: ImportNotFoundException) {
            sendError(exchange, 404, "import_not_found", error.message ?: "Сеанс импорта не найден.")
        } catch (error: ImportInProgressException) {
            sendError(exchange, 409, "import_in_progress", error.message ?: "Файл ещё загружается.")
        } catch (error: SourceBusyException) {
            sendError(exchange, 409, "source_busy", error.message ?: "Источник индексируется.")
        } catch (error: SourceNotFoundException) {
            sendError(exchange, 404, "source_not_found", error.message ?: "Источник не найден.")
        } catch (error: LocalModelException) {
            sendError(exchange, 503, "local_model_unavailable", error.message ?: "Локальная модель недоступна.")
        } catch (error: CloudModelException) {
            sendError(exchange, 502, "cloud_model_unavailable", error.message ?: "Облачная модель недоступна.")
        } catch (error: IllegalArgumentException) {
            sendError(exchange, 400, "invalid_request", error.message ?: "Проверьте параметры запроса.")
        } catch (_: Exception) {
            sendError(exchange, 500, "internal_error", "Локальное приложение не смогло выполнить запрос. Проверьте его журнал.")
        } finally {
            exchange.close()
        }
    }

    private fun handleApi(exchange: HttpExchange, path: String) {
        val method = exchange.requestMethod.uppercase()
        val parts = path.split('/').filter(String::isNotEmpty)
        when {
            path == "/api/models" && method == "GET" -> sendJson(exchange, 200, application.models())
            path == "/api/models/selection" && method == "PUT" -> {
                requireJsonContentType(exchange)
                val request = readJson<ModelSelectionRequest>(exchange)
                sendJson(exchange, 200, application.selectModel(ModelSelection(request.providerId, request.modelId)))
            }
            path == "/api/chat/sessions" && method == "GET" -> sendJson(exchange, 200, application.chatSessions())
            path == "/api/chat/sessions" && method == "POST" -> {
                requireJsonContentType(exchange)
                consumeOptionalJson(exchange)
                sendJson(exchange, 201, application.createChatSession())
            }
            parts.size == 4 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "sessions" && method == "GET" ->
                sendJson(exchange, 200, application.chatSession(parts[3]))
            parts.size == 4 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "sessions" && method == "DELETE" -> {
                requireJsonContentType(exchange)
                val request = readJson<DeleteRequest>(exchange)
                application.deleteChatSession(parts[3], request.confirm)
                sendJson(exchange, 200, SavedResponse(true))
            }
            path == "/api/chat/memory/shared" && method == "GET" -> sendJson(exchange, 200, application.sharedChatFacts())
            parts.size == 5 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "memory" && parts[3] == "shared" && method == "DELETE" -> {
                requireJsonContentType(exchange)
                val request = readJson<DeleteRequest>(exchange)
                application.deleteSharedChatFact(parts[4], request.confirm)
                sendJson(exchange, 200, SavedResponse(true))
            }
            parts.size == 5 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "memory" && parts[3] == "shared" && method == "PUT" -> {
                requireJsonContentType(exchange)
                val request = readJson<ChatMemoryFactRequest>(exchange)
                sendJson(exchange, 200, application.updateSharedChatFact(parts[4], request.content))
            }
            parts.size == 5 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "sessions" && parts[4] == "turns" && method == "POST" -> {
                requireJsonContentType(exchange)
                val request = readJson<ChatTurnHttpRequest>(exchange)
                sendJson(exchange, 200, application.chatTurn(parts[3], request.turnId, request.question, request.strategy, request.topK))
            }
            parts.size == 5 && parts[0] == "api" && parts[1] == "chat" && parts[2] == "sessions" && parts[4] == "task-state" && method == "PUT" -> {
                requireJsonContentType(exchange)
                val state = readJson<dev.localrag.chat.ChatTaskState>(exchange)
                sendJson(exchange, 200, application.updateChatTaskState(parts[3], state))
            }
            path == "/api/status" && method == "GET" -> sendJson(exchange, 200, application.status())
            path == "/api/sources" && method == "GET" -> sendJson(exchange, 200, application.sources())
            path == "/api/imports" && method == "POST" -> {
                requireJsonContentType(exchange)
                val request = readJson<BeginImportRequest>(exchange)
                sendJson(exchange, 201, application.beginImport(request.files))
            }
            parts.size == 5 && parts[0] == "api" && parts[1] == "imports" && parts[3] == "files" && method == "PUT" -> {
                requireBinaryContentType(exchange)
                val contentLength = exchange.requestHeaders.getFirst("Content-Length")?.toLongOrNull()
                if (contentLength != null && contentLength > SourceCatalog.MAX_FILE_BYTES) {
                    val result = application.failImportedFile(parts[2], parts[4], "Файл превышает лимит ${SourceCatalog.MAX_FILE_MIB} MiB.")
                    sendJson(exchange, 200, result)
                } else {
                    sendJson(exchange, 200, application.writeImportedFile(parts[2], parts[4], exchange.requestBody))
                }
            }
            parts.size == 4 && parts[0] == "api" && parts[1] == "imports" && parts[3] == "complete" && method == "POST" -> {
                consumeOptionalJson(exchange)
                sendJson(exchange, 200, application.finishImport(parts[2]))
            }
            parts.size == 3 && parts[0] == "api" && parts[1] == "sources" && method == "DELETE" -> {
                requireJsonContentType(exchange)
                val request = readJson<DeleteRequest>(exchange)
                application.deleteSource(parts[2], request.confirm)
                sendJson(exchange, 200, SavedResponse(true))
            }
            path == "/api/index/pending" && method == "GET" -> sendJson(exchange, 200, application.pendingIndex())
            path == "/api/index" && method == "POST" -> {
                requireJsonContentType(exchange)
                val request = readJson<IndexRequest>(exchange)
                sendJson(exchange, 202, application.startIndex(request.confirm, request.sourceIds, request.totalBytes))
            }
            parts.size == 3 && parts[0] == "api" && parts[1] == "jobs" && method == "GET" -> {
                val job = application.job(parts[2]) ?: throw JobNotFoundException()
                sendJson(exchange, 200, job)
            }
            path == "/api/query" && method == "POST" -> {
                requireJsonContentType(exchange)
                val request = readJson<QueryRequest>(exchange)
                sendJson(exchange, 200, application.query(request.question, request.strategy, request.topK))
            }
            path == "/api/benchmark/questions" && method == "GET" -> sendJson(exchange, 200, application.benchmarkQuestions())
            path == "/api/benchmark/questions" && method == "PUT" -> {
                requireJsonContentType(exchange)
                val request = readJson<QuestionsRequest>(exchange)
                sendJson(exchange, 200, application.saveBenchmarkQuestions(request.questions))
            }
            path == "/api/benchmark/run" && method == "POST" -> {
                consumeOptionalJson(exchange)
                sendJson(exchange, 202, application.startBenchmark())
            }
            path == "/api/benchmark/results" && method == "GET" -> sendJson(exchange, 200, application.benchmarkResults())
            path == "/api/benchmark/review" && method == "POST" -> {
                requireJsonContentType(exchange)
                val review = readJson<ReviewRequest>(exchange)
                application.saveReview(review.questionId, review.strategy, review.baselineRating, review.ragRating, review.note)
                sendJson(exchange, 200, SavedResponse(true))
            }
            else -> {
                val status = if (path.startsWith("/api/") && method !in SAFE_METHODS) 405 else 404
                sendError(exchange, status, if (status == 405) "method_not_allowed" else "not_found", if (status == 405) "Метод запроса не поддерживается." else "Маршрут не найден.")
            }
        }
    }

    private fun handleStatic(exchange: HttpExchange, path: String) {
        if (exchange.requestMethod != "GET") {
            sendError(exchange, 405, "method_not_allowed", "Статическая страница доступна только методом GET.")
            return
        }
        if (path == "/favicon.ico") {
            exchange.responseHeaders.set("Cache-Control", "no-store")
            exchange.sendResponseHeaders(204, -1)
            return
        }
        if (path != "/" && path != "/index.html") {
            sendError(exchange, 404, "not_found", "Страница не найдена.")
            return
        }
        val page = LocalHttpServer::class.java.getResourceAsStream("/index.html")?.use { it.readBytes() }
            ?: throw ApiException(500, "ui_missing", "Файл локального интерфейса не найден.")
        exchange.responseHeaders.set("Content-Type", "text/html; charset=utf-8")
        exchange.responseHeaders.set("Cache-Control", "no-store")
        exchange.responseHeaders.set("X-Content-Type-Options", "nosniff")
        exchange.responseHeaders.set("X-Frame-Options", "DENY")
        exchange.responseHeaders.set("Referrer-Policy", "no-referrer")
        exchange.responseHeaders.set(
            "Content-Security-Policy",
            "default-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:",
        )
        exchange.sendResponseHeaders(200, page.size.toLong())
        exchange.responseBody.use { it.write(page) }
    }

    private fun isSafeLocalRequest(exchange: HttpExchange): Boolean {
        val expectedPort = port
        val host = exchange.requestHeaders.getFirst("Host")?.lowercase() ?: return false
        if (host != "127.0.0.1:$expectedPort" && host != "localhost:$expectedPort") return false
        if (exchange.requestMethod.uppercase() in MUTATING_METHODS) {
            val origin = exchange.requestHeaders.getFirst("Origin") ?: return true
            if (origin != "http://127.0.0.1:$expectedPort" && origin != "http://localhost:$expectedPort") return false
        }
        return true
    }

    private fun requireJsonContentType(exchange: HttpExchange) {
        val value = exchange.requestHeaders.getFirst("Content-Type")?.substringBefore(';')?.trim()
        if (!value.equals("application/json", ignoreCase = true)) {
            throw ApiException(415, "json_required", "Передайте тело запроса в формате application/json.")
        }
    }

    private fun requireBinaryContentType(exchange: HttpExchange) {
        val value = exchange.requestHeaders.getFirst("Content-Type")?.substringBefore(';')?.trim()
        if (!value.equals("application/octet-stream", ignoreCase = true)) {
            throw ApiException(415, "binary_required", "Передайте файл как application/octet-stream.")
        }
    }

    private inline fun <reified T> readJson(exchange: HttpExchange): T {
        val raw = readBody(exchange)
        return try {
            json.decodeFromString(raw)
        } catch (error: Exception) {
            throw ApiException(400, "invalid_json", "Тело запроса не соответствует ожидаемому JSON-формату.")
        }
    }

    private fun consumeOptionalJson(exchange: HttpExchange) {
        val raw = readBody(exchange).trim()
        if (raw.isNotEmpty() && raw != "{}") throw ApiException(400, "invalid_body", "Для этого действия тело запроса должно быть пустым или {}.")
    }

    private fun readBody(exchange: HttpExchange): String {
        val declaredSize = exchange.requestHeaders.getFirst("Content-Length")?.toLongOrNull()
        if (declaredSize != null && declaredSize > MAX_JSON_BYTES) {
            throw ApiException(413, "request_too_large", "JSON-запрос превышает допустимый размер.")
        }
        val bytes = exchange.requestBody.use { it.readNBytes(MAX_JSON_BYTES + 1) }
        if (bytes.size > MAX_JSON_BYTES) throw ApiException(413, "request_too_large", "JSON-запрос превышает допустимый размер.")
        return bytes.toString(StandardCharsets.UTF_8)
    }

    private fun sendError(exchange: HttpExchange, status: Int, code: String, message: String) {
        try {
            sendJson(exchange, status, ErrorPayload(ErrorDetail(code, message)))
        } catch (_: IOException) {
            exchange.close()
        }
    }

    private inline fun <reified T> sendJson(exchange: HttpExchange, status: Int, value: T) {
        val bytes = json.encodeToString(value).toByteArray(StandardCharsets.UTF_8)
        exchange.responseHeaders.set("Content-Type", "application/json; charset=utf-8")
        exchange.responseHeaders.set("Cache-Control", "no-store")
        exchange.responseHeaders.set("X-Content-Type-Options", "nosniff")
        exchange.sendResponseHeaders(status, bytes.size.toLong())
        exchange.responseBody.use { it.write(bytes) }
    }

    override fun close() {
        server.stop(0)
        workers.shutdownNow()
    }

    companion object {
        const val MAX_JSON_BYTES = 1024 * 1024
        private val MUTATING_METHODS = setOf("POST", "PUT", "DELETE")
        private val SAFE_METHODS = setOf("GET", "HEAD", "OPTIONS")
    }
}
