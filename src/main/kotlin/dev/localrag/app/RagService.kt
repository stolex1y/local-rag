package dev.localrag.app

import dev.localrag.domain.ChatPort
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.EmbeddingPort
import dev.localrag.domain.IndexRepository
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.SourceCitation
import dev.localrag.domain.SourceStatus
import kotlinx.serialization.Serializable

class IndexNotReadyException(message: String) : RuntimeException(message)

@Serializable
data class AnswerEnvelope(val answer: String)

@Serializable
data class QueryAnswer(
    val answer: String,
    val sources: List<SourceCitation> = emptyList(),
)

@Serializable
data class QueryResponse(
    val baseline: AnswerEnvelope,
    val rag: QueryAnswer,
)

class RagService(
    private val index: IndexRepository,
    private val embeddings: EmbeddingPort,
    private val chat: ChatPort,
    private val defaultTopK: Int = 4,
    private val maxTopK: Int = 20,
) {
    init {
        require(defaultTopK in 1..maxTopK)
    }

    fun answer(question: String, strategy: ChunkStrategy, topK: Int?): QueryResponse {
        validateQuestion(question)
        requireReadySources()
        val count = validateTopK(topK)
        val baseline = baseline(question)
        val retrieved = retrieve(question, strategy, count)
        requireRetrieved(retrieved)
        return QueryResponse(
            baseline = AnswerEnvelope(baseline),
            rag = QueryAnswer(chat.answer(question, retrieved), retrieved.map { it.toCitation() }),
        )
    }

    fun baseline(question: String): String {
        validateQuestion(question)
        return chat.answer(question)
    }

    fun retrieveForStrategies(question: String, topK: Int = defaultTopK): Map<ChunkStrategy, List<ScoredChunk>> {
        validateQuestion(question)
        requireReadySources()
        val count = validateTopK(topK)
        val queryEmbedding = embeddings.embed(listOf(question)).singleOrNull()
            ?: throw IllegalStateException("Embedding-сервис должен вернуть один вектор запроса.")
        val results = ChunkStrategy.entries.associateWith { strategy ->
            index.search(strategy, queryEmbedding, embeddings.modelName, count)
        }
        results.values.forEach(::requireRetrieved)
        return results
    }

    fun retrieve(question: String, strategy: ChunkStrategy, topK: Int = defaultTopK): List<ScoredChunk> {
        validateQuestion(question)
        requireReadySources()
        val count = validateTopK(topK)
        val queryEmbedding = embeddings.embed(listOf(question)).singleOrNull()
            ?: throw IllegalStateException("Embedding-сервис должен вернуть один вектор запроса.")
        return index.search(strategy, queryEmbedding, embeddings.modelName, count).also(::requireRetrieved)
    }

    fun ragAnswer(question: String, sources: List<ScoredChunk>): String {
        validateQuestion(question)
        return chat.answer(question, sources)
    }

    private fun validateQuestion(question: String) {
        require(question.isNotBlank() && question.length <= MAX_QUESTION_LENGTH) {
            "Вопрос должен содержать от 1 до $MAX_QUESTION_LENGTH символов."
        }
    }

    private fun validateTopK(topK: Int?): Int {
        val value = topK ?: defaultTopK
        require(value in 1..maxTopK) { "Top-K должен быть от 1 до $maxTopK." }
        return value
    }

    private fun requireReadySources() {
        if (index.sources(SourceStatus.READY).isEmpty()) {
            throw IndexNotReadyException("В коллекции пока нет готовых источников. Добавьте файлы и подтвердите индексацию.")
        }
    }
    private fun requireRetrieved(sources: List<ScoredChunk>) {
        if (sources.isEmpty()) {
            throw IndexNotReadyException("Для текущей embedding-модели нет совместимых индексных фрагментов.")
        }
    }

    private fun ScoredChunk.toCitation() = SourceCitation(
        sourceId = chunk.draft.sourceId,
        source = chunk.draft.sourceName,
        section = chunk.draft.section,
        chunkId = chunk.draft.chunkId,
        location = chunk.draft.location,
        score = cosineScore,
    )

    companion object {
        const val MAX_QUESTION_LENGTH = 4_000
    }
}
