package dev.localrag.app
import dev.localrag.chat.ChatSession
import dev.localrag.chat.ChatSessionDetail
import dev.localrag.chat.ChatStore
import dev.localrag.chat.ChatMemoryFactSummary
import dev.localrag.chat.ChatService
import dev.localrag.chat.ChatTaskState
import dev.localrag.chat.ChatTurnResponse


import dev.localrag.benchmark.BenchmarkQuestion
import dev.localrag.benchmark.BenchmarkQuestionSet
import dev.localrag.benchmark.BenchmarkResult
import dev.localrag.benchmark.BenchmarkRunner
import dev.localrag.benchmark.BenchmarkStore
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.IndexRepository
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredSource
import dev.localrag.index.IndexWorkflow
import dev.localrag.domain.ModelSelection
import dev.localrag.generation.ModelConfiguration
import dev.localrag.generation.ModelConfigurationResponse
import dev.localrag.ollama.EMBEDDING_MODEL
import dev.localrag.ollama.OllamaApi
import dev.localrag.source.ImportBatch
import dev.localrag.source.ImportCompletion
import dev.localrag.source.ImportFileMetadata
import dev.localrag.source.ImportFileResult
import dev.localrag.source.SourceCatalog
import dev.localrag.source.SourceNotFoundException
import kotlinx.serialization.Serializable
import java.io.InputStream

@Serializable
data class ModelStatus(
    val embedding: String,
    val available: Boolean,
)

@Serializable
data class SourceSummary(
    val total: Int,
    val pending: Int,
    val ready: Int,
    val failed: Int,
    val totalBytes: Long,
)

@Serializable
data class StatusResponse(
    val models: ModelStatus,
    val sources: SourceSummary,
    val activeJob: JobSnapshot? = null,
    val benchmark: BenchmarkQuestionSet,
)

@Serializable
data class PendingIndexResponse(
    val files: List<SourceRecord>,
    val totalBytes: Long,
)

@Serializable
data class RetrievalSummary(
    val strategy: ChunkStrategy,
    val k: Int,
    val completedQuestions: Int,
    val rawHitAtK: Double?,
    val rawMrr: Double?,
    val enhancedHitAtK: Double?,
    val enhancedMrr: Double?,
)

@Serializable
data class BenchmarkSummary(
    val baselinePass: Int,
    val baselinePartial: Int,
    val baselineFail: Int,
    val ragPass: Int,
    val ragPartial: Int,
    val ragFail: Int,
    val retrieval: List<RetrievalSummary>,
)

@Serializable
data class BenchmarkResultsResponse(
    val results: List<BenchmarkResult>,
    val summary: BenchmarkSummary,
)

class ApplicationService(
    private val index: IndexRepository,
    private val store: BenchmarkStore,
    private val catalog: SourceCatalog,
    private val jobs: JobManager,
    private val indexing: IndexWorkflow,
    private val rag: RagService,
    private val benchmark: BenchmarkRunner,
    private val ollama: OllamaApi,
    private val modelConfiguration: ModelConfiguration,
    private val chatStore: ChatStore,
    private val chatService: ChatService,
) {

    private val operationLock = Any()
    fun status(): StatusResponse {
        val sources = catalog.sources()
        val availableModels = availableModels()
        return StatusResponse(
            models = ModelStatus(EMBEDDING_MODEL, EMBEDDING_MODEL in availableModels),
            sources = SourceSummary(
                total = sources.size,
                pending = sources.count { it.status == SourceStatus.PENDING },
                ready = sources.count { it.status == SourceStatus.READY },
                failed = sources.count { it.status in setOf(SourceStatus.FAILED, SourceStatus.UNSUPPORTED) },
                totalBytes = sources.sumOf(SourceRecord::sizeBytes),
            ),
            activeJob = jobs.active(),
            benchmark = benchmarkQuestions(),
        )
    }

    fun sources(): List<SourceRecord> = catalog.sources()
    fun chatSessions(): List<ChatSession> = chatStore.sessions()
    fun createChatSession(): ChatSessionDetail = chatStore.createSession()

    fun chatSession(sessionId: String): ChatSessionDetail =
        chatStore.sessionDetail(sessionId) ?: throw ApiException(404, "chat_session_not_found", "Сессия чата не найдена.")

    fun deleteChatSession(sessionId: String, confirmed: Boolean) {
        if (!confirmed) throw ApiException(400, "confirmation_required", "Удаление сессии требует отдельного подтверждения.")
        if (!chatStore.deleteSession(sessionId)) throw ApiException(404, "chat_session_not_found", "Сессия чата не найдена.")
    }

    fun chatTurn(
        sessionId: String,
        turnId: String,
        question: String,
        strategy: ChunkStrategy,
        topK: Int?,
    ): ChatTurnResponse {
        chatSession(sessionId)
        val configuration = modelConfiguration.options()
        return chatService.turn(
            sessionId,
            turnId,
            question,
            strategy,
            ModelSelection(configuration.selectedProviderId, configuration.selectedModelId),
            topK,
        )
    }


    fun sharedChatFacts(): List<ChatMemoryFactSummary> = chatService.sharedFacts()


    fun deleteSharedChatFact(memoryId: String, confirmed: Boolean) = chatService.deleteSharedFact(memoryId, confirmed)
    fun updateSharedChatFact(memoryId: String, content: String): ChatMemoryFactSummary =
        chatService.updateSharedFact(memoryId, content)

    fun updateChatTaskState(sessionId: String, taskState: ChatTaskState): ChatTaskState {
        chatSession(sessionId)
        return chatService.updateTaskState(sessionId, taskState)
    }
    fun models(): ModelConfigurationResponse = modelConfiguration.options()

    fun selectModel(selection: ModelSelection): ModelConfigurationResponse {
        try {
            modelConfiguration.select(selection)
        } catch (error: IllegalArgumentException) {
            throw ApiException(400, "invalid_model_selection", error.message ?: "Выберите модель из каталога.")
        }
        return modelConfiguration.options()
    }

    fun beginImport(files: List<ImportFileMetadata>): ImportBatch = catalog.beginImport(files)

    fun writeImportedFile(importId: String, fileId: String, input: InputStream): ImportFileResult =
        catalog.writeImportedFile(importId, fileId, input)

    fun failImportedFile(importId: String, fileId: String, message: String): ImportFileResult =
        catalog.failImportedFile(importId, fileId, message)

    fun finishImport(importId: String): ImportCompletion = catalog.finishImport(importId)

    fun deleteSource(sourceId: String, confirmed: Boolean) = synchronized(operationLock) {
        if (jobs.active() != null) throw JobBusyException()
        catalog.deleteSource(sourceId, confirmed)
    }

    fun pendingIndex(): PendingIndexResponse {
        val files = catalog.pendingSources()
        val totalBytes = files.fold(0L) { total, source ->
            require(source.sizeBytes <= Long.MAX_VALUE - total) { "Размер коллекции слишком велик." }
            total + source.sizeBytes
        }
        return PendingIndexResponse(files, totalBytes)
    }

    fun startIndex(confirmed: Boolean, sourceIds: List<String>, totalBytes: Long): JobSnapshot = synchronized(operationLock) {
        if (!confirmed) throw ApiException(400, "confirmation_required", "Индексация требует отдельного подтверждения.")
        val pending = index.pendingSources()
        if (pending.isEmpty()) throw ApiException(409, "nothing_pending", "Нет новых источников для индексации.")
        val expectedIds = pending.map { it.record.sourceId }.toSet()
        if (sourceIds.size != expectedIds.size || sourceIds.toSet() != expectedIds) {
            throw ApiException(409, "pending_collection_changed", "Состав ожидающих источников изменился. Обновите список перед подтверждением.")
        }
        val expectedBytes = pending.fold(0L) { total, source ->
            if (source.record.sizeBytes > Long.MAX_VALUE - total) {
                throw ApiException(409, "pending_size_too_large", "Общий размер ожидающих источников превышает допустимое значение.")
            }
            total + source.record.sizeBytes
        }
        if (totalBytes != expectedBytes) {
            throw ApiException(409, "pending_size_changed", "Общий размер источников изменился. Обновите список перед подтверждением.")
        }

        if (expectedBytes > SourceCatalog.MAX_IMPORT_BYTES) {
            throw ApiException(
                409,
                "pending_size_too_large",
                "Ожидающие индексации источники не должны превышать ${SourceCatalog.MAX_IMPORT_MIB} MiB за один запуск.",
            )
        }
        requireModels(EMBEDDING_MODEL)
        jobs.submit("INDEX", pending.size, expectedBytes) { progress ->
            indexing.index(pending) { update -> progress.update(update) }
        }
    }

    fun query(question: String, strategy: ChunkStrategy, topK: Int?): QueryResponse {
        val selection = modelConfiguration.currentSelection()
        requireModels(EMBEDDING_MODEL)
        modelConfiguration.requireCredential(selection)
        return try {
            rag.answer(question, strategy, topK, selection)
        } catch (error: IllegalArgumentException) {
            throw ApiException(400, "invalid_query", error.message ?: "Проверьте вопрос и Top-K.")
        }
    }

    fun job(jobId: String): JobSnapshot? = jobs.get(jobId)

    fun saveBenchmarkQuestions(questions: List<BenchmarkQuestion>): BenchmarkQuestionSet = synchronized(operationLock) {
        val candidate = benchmarkQuestionSet(questions)
        if (questions.size == BenchmarkStore.EXPECTED_QUESTION_COUNT && !candidate.runnable) {
            throw ApiException(400, "invalid_benchmark", candidate.errors.joinToString(" "))
        }
        try {
            store.saveQuestions(questions)
        } catch (error: IllegalArgumentException) {
            throw ApiException(400, "invalid_benchmark", error.message ?: "Проверьте JSON-набор вопросов.")
        }
        benchmarkQuestions()
    }

    fun benchmarkQuestions(): BenchmarkQuestionSet = benchmarkQuestionSet(store.questions())

    private fun benchmarkQuestionSet(questions: List<BenchmarkQuestion>): BenchmarkQuestionSet {
        val errors = mutableListOf<String>()
        if (questions.size != BenchmarkStore.EXPECTED_QUESTION_COUNT) {
            errors += "Нужно загрузить JSON ровно с ${BenchmarkStore.EXPECTED_QUESTION_COUNT} вопросами."
        }
        questions.forEach { question ->
            question.expectedSources.forEach { expected ->
                val source = index.source(expected.sourceId)
                when {
                    source == null -> errors += "Вопрос ${question.id}: источник ${expected.sourceId} отсутствует."
                    source.status != SourceStatus.READY -> errors += "Вопрос ${question.id}: источник «${source.name}» ещё не проиндексирован."
                    !validLocationType(source.type, expected) -> errors += "Вопрос ${question.id}: ссылка на «${source.name}» не соответствует типу источника."
                    !index.hasSourceLocation(expected.sourceId, expected.location, expected.section) ->
                        errors += "Вопрос ${question.id}: ожидаемая страница, строка или раздел не найдены в индексе «${source.name}»."
                }
            }
        }
        return BenchmarkQuestionSet(questions, errors.distinct(), errors.isEmpty())
    }

    fun startBenchmark(): JobSnapshot = synchronized(operationLock) {
        val selection = modelConfiguration.currentSelection()
        requireModels(EMBEDDING_MODEL)
        modelConfiguration.requireCredential(selection)
        val questionSet = benchmarkQuestions()
        if (!questionSet.runnable) {
            throw ApiException(409, "benchmark_not_ready", questionSet.errors.joinToString(" "))
        }
        if (index.sources(SourceStatus.READY).isEmpty()) {
            throw ApiException(409, "index_not_ready", "Сначала проиндексируйте источники коллекции.")
        }
        val questions = questionSet.questions
        val cases = questions.size * ChunkStrategy.entries.size
        jobs.submit("BENCHMARK", cases, 0) { progress ->
            benchmark.run(questions, selection) { update -> progress.update(update) }
        }
    }

    fun benchmarkResults(): BenchmarkResultsResponse {
        val rows = store.results()
        return BenchmarkResultsResponse(
            results = rows,
            summary = BenchmarkSummary(
                baselinePass = rows.count { it.baselineRating == "PASS" },
                baselinePartial = rows.count { it.baselineRating == "PARTIAL" },
                baselineFail = rows.count { it.baselineRating == "FAIL" },
                ragPass = rows.count { it.ragRating == "PASS" },
                ragPartial = rows.count { it.ragRating == "PARTIAL" },
                ragFail = rows.count { it.ragRating == "FAIL" },
                retrieval = ChunkStrategy.entries.map { strategy ->
                    val completed = rows.filter { it.strategy == strategy && it.rawRetrievedSources != null }
                    val count = completed.size
                    val denominator = BenchmarkStore.EXPECTED_QUESTION_COUNT.toDouble()
                    RetrievalSummary(
                        strategy = strategy,
                        k = RagService.DEFAULT_TOP_K,
                        completedQuestions = count,
                        rawHitAtK =
                            completed.count { row ->
                                row.rawExpectedSourceRank?.let { rank -> rank in 1..RagService.DEFAULT_TOP_K } == true
                            } / denominator,
                        rawMrr =
                            completed.sumOf { it.rawExpectedSourceRank?.let { rank -> 1.0 / rank } ?: 0.0 } / denominator,
                        enhancedHitAtK =
                            completed.count { row ->
                                row.expectedSourceRank?.let { rank -> rank in 1..RagService.DEFAULT_TOP_K } == true
                            } / denominator,
                        enhancedMrr =
                            completed.sumOf { it.expectedSourceRank?.let { rank -> 1.0 / rank } ?: 0.0 } / denominator,
                    )
                },
            ),
        )
    }

    fun saveReview(questionId: String, strategy: ChunkStrategy, baselineRating: String, ragRating: String, note: String?) {
        if (questionId.length !in 1..80) throw ApiException(400, "invalid_question_id", "Идентификатор вопроса некорректен.")
        if (note != null && note.length > MAX_NOTE_CHARS) throw ApiException(400, "note_too_long", "Заметка не должна превышать $MAX_NOTE_CHARS символов.")
        try {
            store.saveReview(questionId, strategy, baselineRating.uppercase(), ragRating.uppercase(), note?.trim()?.takeIf(String::isNotEmpty))
        } catch (error: IllegalArgumentException) {
            throw ApiException(400, "invalid_review", error.message ?: "Проверьте оценки и существование результата benchmark.")
        }
    }

    private fun validLocationType(sourceType: SourceType, expected: dev.localrag.benchmark.BenchmarkExpectedSource): Boolean = when (sourceType) {
        SourceType.PDF -> expected.location.pageStart != null && expected.location.lineStart == null
        SourceType.TEXT, SourceType.MARKDOWN, SourceType.CODE -> expected.location.lineStart != null || !expected.section.isNullOrBlank()
        SourceType.HTML -> !expected.section.isNullOrBlank()
    }

    private fun requireModels(vararg models: String) {
        val available = availableModels()
        val missing = models.filterNot(available::contains)
        if (missing.isNotEmpty()) {
            throw ApiException(503, "models_unavailable", "Запустите локальный Ollama и загрузите: ${missing.joinToString()}.")
        }
    }

    private fun availableModels(): Set<String> = try {
        ollama.availableModels()
    } catch (_: Exception) {
        emptySet()
    }

    companion object {
        private const val MAX_NOTE_CHARS = 2_000
    }
}

class ApiException(
    val status: Int,
    val code: String,
    override val message: String,
) : RuntimeException(message)
