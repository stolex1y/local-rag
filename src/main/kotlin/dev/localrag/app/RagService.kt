package dev.localrag.app

import dev.localrag.domain.ChatPort
import dev.localrag.domain.RerankPort
import dev.localrag.domain.ModelSelection
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
    val rawSources: List<SourceCitation>,
    val modelSelection: ModelSelection,
)

internal data class RetrievalComparison(
    val raw: List<ScoredChunk>,
    val enhanced: List<ScoredChunk>,
)


class RagService(
    private val index: IndexRepository,
    private val embeddings: EmbeddingPort,
    private val chat: ChatPort,
    private val reranker: RerankPort,
    private val defaultTopK: Int = DEFAULT_TOP_K,
    private val maxTopK: Int = MAX_TOP_K,
) {
    init {
        require(defaultTopK in 1..maxTopK)
    }

    fun answer(question: String, strategy: ChunkStrategy, topK: Int?, selection: ModelSelection): QueryResponse {
        validateQuestion(question)
        requireReadySources()
        val count = validateTopK(topK)
        val baseline = baseline(question, selection)
        val comparison = retrieveComparison(question, strategy, count, selection)
        val ragAnswer = ragAnswer(question, comparison.enhanced, selection)
        return QueryResponse(
            baseline = AnswerEnvelope(baseline),
            rag = QueryAnswer(ragAnswer, comparison.enhanced.map { it.toCitation() }),
            rawSources = comparison.raw.map { it.toCitation() },
            modelSelection = selection,
        )
    }

    fun baseline(question: String, selection: ModelSelection): String {
        validateQuestion(question)
        return chat.answer(selection, question)
    }

    internal fun retrieveForStrategies(
        question: String,
        selection: ModelSelection,
        topK: Int = defaultTopK,
    ): Map<ChunkStrategy, RetrievalComparison> {
        validateQuestion(question)
        requireReadySources()
        val count = validateTopK(topK)
        val rawEmbedding = embedQuery(question)
        val normalized = QueryNormalizer.normalize(question)
        val enhancedEmbedding = if (normalized == question) rawEmbedding else embedQuery(normalized)
        return ChunkStrategy.entries.associateWith { strategy ->
            compare(strategy, question, count, selection, rawEmbedding, enhancedEmbedding)
        }
    }

    private fun retrieveComparison(
        question: String,
        strategy: ChunkStrategy,
        topK: Int,
        selection: ModelSelection,
    ): RetrievalComparison {
        val rawEmbedding = embedQuery(question)
        val normalized = QueryNormalizer.normalize(question)
        val enhancedEmbedding = if (normalized == question) rawEmbedding else embedQuery(normalized)
        return compare(strategy, question, topK, selection, rawEmbedding, enhancedEmbedding)
    }

    private fun compare(
        strategy: ChunkStrategy,
        question: String,
        topK: Int,
        selection: ModelSelection,
        rawEmbedding: List<Float>,
        enhancedEmbedding: List<Float>,
    ): RetrievalComparison {
        val raw = index.search(strategy, rawEmbedding, embeddings.modelName, topK)
        val candidates = filterRelevant(index.search(strategy, enhancedEmbedding, embeddings.modelName, topK))
        if (candidates.isEmpty()) return RetrievalComparison(raw, emptyList())

        val order = reranker.rerank(selection, question, candidates.map { it.chunk.draft.text })
        if (order.size != candidates.size) throw IllegalStateException("Reranker вернул некорректный порядок фрагментов.")
        val seen = BooleanArray(candidates.size)
        val enhanced = ArrayList<ScoredChunk>(candidates.size)
        for (candidateIndex in order) {
            if (candidateIndex !in candidates.indices || seen[candidateIndex]) {
                throw IllegalStateException("Reranker вернул некорректный порядок фрагментов.")
            }
            seen[candidateIndex] = true
            enhanced += candidates[candidateIndex]
        }
        return RetrievalComparison(raw, enhanced)
    }

    private fun embedQuery(query: String): List<Float> =
        embeddings.embed(listOf(query)).singleOrNull()
            ?: throw IllegalStateException("Embedding-сервис должен вернуть один вектор запроса.")


    fun ragAnswer(question: String, sources: List<ScoredChunk>, selection: ModelSelection): String {
        validateQuestion(question)
        if (sources.isEmpty()) return NO_CONTEXT_NOTICE
        return chat.answer(selection, question, sources)
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
    private fun filterRelevant(chunks: List<ScoredChunk>) =
        chunks.filter { it.cosineScore >= MIN_RETRIEVAL_SCORE }


    private fun ScoredChunk.toCitation() = SourceCitation(
        sourceId = chunk.draft.sourceId,
        source = chunk.draft.sourceName,
        section = chunk.draft.section,
        chunkId = chunk.draft.chunkId,
        location = chunk.draft.location,
        score = cosineScore,
        quote = chunk.draft.text.take(MAX_CITATION_QUOTE_LENGTH),
    )

    companion object {
        const val MAX_QUESTION_LENGTH = 4_000
        const val DEFAULT_TOP_K = 4
        const val MAX_TOP_K = 20
        private const val MAX_CITATION_QUOTE_LENGTH = 300
        private const val MIN_RETRIEVAL_SCORE = 0.20
        private const val NO_CONTEXT_NOTICE =
            "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник."
    }
}
