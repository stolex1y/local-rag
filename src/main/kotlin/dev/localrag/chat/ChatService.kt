package dev.localrag.chat

import dev.localrag.app.RagService
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.EmbeddingPort
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.SourceCitation
import java.util.Locale
import kotlin.math.sqrt

@kotlinx.serialization.Serializable
data class ChatTurnResponse(val message: ChatMessage, val taskState: ChatTaskState)

class ChatService(
    private val store: ChatStore,
    private val rag: RagService,
    private val generator: ChatTurnGenerator,
    private val embeddings: EmbeddingPort,
) {
    @Synchronized
    fun turn(
        sessionId: String,
        turnId: String,
        question: String,
        strategy: ChunkStrategy,
        selection: ModelSelection,
        topK: Int? = null,
    ): ChatTurnResponse {
        require(question.isNotBlank() && question.length <= ChatStore.MAX_MESSAGE_LENGTH) {
            "Вопрос должен содержать от 1 до ${ChatStore.MAX_MESSAGE_LENGTH} символов."
        }
        require(store.sessionExists(sessionId)) { "Сессия чата не найдена." }
        val previous = store.recentMessages(sessionId, ChatStore.MAX_HISTORY_MESSAGES + 1)
            .filterNot { it.turnId == turnId }
            .takeLast(ChatStore.MAX_HISTORY_MESSAGES)
        store.saveUserMessage(sessionId, turnId, question)
        val completed = store.assistantMessage(sessionId, turnId)
        if (completed != null) return ChatTurnResponse(completed, store.taskState(sessionId))
        val retrieved = rag.retrieveForChat(question, strategy, topK, selection).enhanced
        val sharedFacts = store.memoryFacts(sessionId).filter { it.scope == ChatMemoryScope.SHARED }
        val queryEmbedding = if (sharedFacts.isEmpty()) emptyList() else embedOne(question)
        val selectedSharedFacts = selectFacts(sharedFacts, ChatMemoryScope.SHARED, queryEmbedding)
        val state = store.taskState(sessionId)
        val result = generator.completeTurn(
            selection,
            ChatTurnRequest(
                question = question,
                previousMessages = previous,
                taskState = state,
                sharedFacts = selectedSharedFacts,
                documentChunks = retrieved.map { it.chunk.draft.text },
            ),
        )
        validateDelta(result.taskStateDelta)
        val nextState = merge(state, result.taskStateDelta)
        val newFacts = stateFacts(nextState)
        val factVectors = if (newFacts.isEmpty()) emptyList() else embeddings.embed(newFacts)
        require(factVectors.size == newFacts.size && factVectors.all { it.isNotEmpty() }) {
            "Embedding-сервис вернул неполный вектор для локальной памяти."
        }
        val factDrafts = newFacts.zip(factVectors).map { (text, vector) ->
            ChatMemoryFactDraft(text, vector, embeddings.modelName)
        }
        val hasSupportingEvidence = retrieved.isNotEmpty() || selectedSharedFacts.isNotEmpty()
        val references = if (hasSupportingEvidence) {
            selectedSharedFacts.map { fact -> ChatMemoryReference(fact.scope, fact.id, fact.text) }
        } else {
            emptyList()
        }
        val citations = if (hasSupportingEvidence) retrieved.map { it.toCitation() } else emptyList()
        val answer = if (hasSupportingEvidence) result.answer else ABSTENTION
        val message = store.completeTurn(
            sessionId = sessionId,
            turnId = turnId,
            answer = answer,
            citations = citations,
            memoryReferences = references,
            taskState = nextState,
            sessionFacts = factDrafts,
        )
        return ChatTurnResponse(message, store.taskState(sessionId))
    }

    @Synchronized
    fun updateTaskState(sessionId: String, state: ChatTaskState): ChatTaskState {
        require(store.sessionExists(sessionId)) { "Сессия чата не найдена." }
        state.goal?.let { requireStateString(it, "Цель") }
        require(state.clarifications.size <= MAX_TASK_STATE_ITEMS && state.constraints.size <= MAX_TASK_STATE_ITEMS && state.terms.size <= MAX_TASK_STATE_ITEMS) {
            "В каждой категории состояния задачи допускается не более $MAX_TASK_STATE_ITEMS записей."
        }
        state.clarifications.forEach { requireStateString(it, "Уточнение") }
        state.constraints.forEach { requireStateString(it, "Ограничение") }
        state.terms.forEach {
            requireStateString(it.term, "Термин")
            requireStateString(it.definition, "Определение")
        }
        val facts = stateFacts(state)
        val vectors = if (facts.isEmpty()) emptyList() else embeddings.embed(facts)
        require(vectors.size == facts.size && vectors.all { it.isNotEmpty() }) {
            "Embedding-сервис вернул неполный вектор для локальной памяти."
        }
        store.replaceTaskState(
            sessionId,
            state,
            facts.zip(vectors).map { (text, vector) -> ChatMemoryFactDraft(text, vector, embeddings.modelName) },
        )
        return state
    }

    fun promoteSessionFact(sessionId: String, memoryId: String, confirmed: Boolean): ChatMemoryFactSummary {
        require(confirmed) { "Копирование факта в общую память требует отдельного подтверждения." }
        val fact = store.memoryFacts(sessionId).firstOrNull {
            it.scope == ChatMemoryScope.SESSION && it.id == memoryId
        } ?: throw IllegalArgumentException("Факт сессии не найден.")
        return store.saveSharedFact(fact.text, fact.embedding, fact.embeddingModel).toSummary()
    }

    fun sharedFacts(): List<ChatMemoryFactSummary> = store.sharedFacts().map { it.toSummary() }

    fun sessionMemoryFacts(sessionId: String): List<ChatMemoryFactSummary> =
        store.memoryFacts(sessionId).filter { it.scope == ChatMemoryScope.SESSION }.map { it.toSummary() }

    private fun ChatMemoryFact.toSummary() = ChatMemoryFactSummary(scope, id, sessionId, text, createdAt)

    fun deleteSharedFact(memoryId: String, confirmed: Boolean) {
        require(confirmed) { "Удаление общей памяти требует отдельного подтверждения." }
        if (!store.deleteSharedFact(memoryId)) throw IllegalArgumentException("Общий факт не найден.")
    }

    fun updateSharedFact(memoryId: String, content: String): ChatMemoryFactSummary {
        require(content.isNotBlank() && content.length <= ChatStore.MAX_SHARED_FACT_LENGTH)
        val embedding = embedOne(content)
        if (!store.updateSharedFact(memoryId, content, embedding, embeddings.modelName)) {
            throw IllegalArgumentException("Общий факт не найден.")
        }
        return store.sharedFacts().first { it.id == memoryId }.toSummary()
    }

    private fun selectFacts(
        facts: List<ChatMemoryFact>,
        scope: ChatMemoryScope,
        query: List<Float>,
    ): List<ChatMemoryFact> {
        if (query.isEmpty()) return emptyList()
        return facts.asSequence()
            .filter { it.scope == scope && it.embeddingModel == embeddings.modelName && it.embedding.size == query.size }
            .map { it to cosine(query, it.embedding) }
            .filter { (_, score) -> score >= MIN_MEMORY_SCORE }
            .sortedWith(compareByDescending<Pair<ChatMemoryFact, Double>> { it.second }.thenBy { it.first.id })
            .take(MAX_SELECTED_FACTS)
            .map { it.first }
            .toList()
    }

    private fun embedOne(text: String): List<Float> = embeddings.embed(listOf(text)).singleOrNull()
        ?.takeIf(List<Float>::isNotEmpty)
        ?: throw IllegalStateException("Embedding-сервис должен вернуть один вектор запроса.")

    private fun stateFacts(state: ChatTaskState): List<String> = buildList {
        state.goal?.let { add("Цель задачи: ${it.trim()}") }
        state.clarifications.forEach { add("Уточнение: ${it.trim()}") }
        state.constraints.forEach { add("Ограничение: ${it.trim()}") }
        state.terms.forEach { add("Термин ${it.term.trim()}: ${it.definition.trim()}") }
    }.distinctBy(::normalize)

    private fun validateDelta(delta: ChatTaskStateDelta) {
        delta.goal?.let { requireStateString(it, "Цель") }
        require(delta.clarificationsToAdd.size <= MAX_TASK_STATE_ITEMS)
        require(delta.constraintsToAdd.size <= MAX_TASK_STATE_ITEMS)
        require(delta.termsToAdd.size <= MAX_TASK_STATE_ITEMS)
        delta.clarificationsToAdd.forEach { requireStateString(it, "Уточнение") }
        delta.constraintsToAdd.forEach { requireStateString(it, "Ограничение") }
        delta.termsToAdd.forEach {
            requireStateString(it.term, "Термин")
            requireStateString(it.definition, "Определение")
        }
    }

    private fun requireStateString(value: String, label: String) {
        require(value.isNotBlank() && value.length <= MAX_TASK_STATE_STRING_LENGTH) {
            "$label должно содержать от 1 до $MAX_TASK_STATE_STRING_LENGTH символов."
        }
    }

    private fun merge(state: ChatTaskState, delta: ChatTaskStateDelta): ChatTaskState {
        val merged = ChatTaskState(
            goal = delta.goal?.trim() ?: state.goal,
            clarifications = mergeStrings(state.clarifications, delta.clarificationsToAdd),
            constraints = mergeStrings(state.constraints, delta.constraintsToAdd),
            terms = buildList {
                addAll(state.terms)
                delta.termsToAdd.forEach { next ->
                    val key = normalize(next.term)
                    val existingIndex = indexOfFirst { normalize(it.term) == key }
                    val normalized = ChatTaskTerm(next.term.trim(), next.definition.trim())
                    if (existingIndex < 0) add(normalized) else this[existingIndex] = normalized
                }
            },
        )
        require(
            merged.clarifications.size <= MAX_TASK_STATE_ITEMS &&
                merged.constraints.size <= MAX_TASK_STATE_ITEMS &&
                merged.terms.size <= MAX_TASK_STATE_ITEMS,
        ) {
            "В каждой категории состояния задачи допускается не более $MAX_TASK_STATE_ITEMS записей. Удалите ненужные записи и повторите ход."
        }
        return merged
    }

    private fun mergeStrings(existing: List<String>, additions: List<String>): List<String> = buildList {
        addAll(existing)
        val known = existing.mapTo(HashSet(), ::normalize)
        additions.forEach { value ->
            val trimmed = value.trim()
            if (known.add(normalize(trimmed))) add(trimmed)
        }
    }

    private fun normalize(value: String): String = value.trim().lowercase(Locale.ROOT)

    private fun cosine(left: List<Float>, right: List<Float>): Double {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        left.indices.forEach { index ->
            val a = left[index].toDouble()
            val b = right[index].toDouble()
            dot += a * b
            leftNorm += a * a
            rightNorm += b * b
        }
        if (leftNorm == 0.0 || rightNorm == 0.0) return 0.0
        return dot / (sqrt(leftNorm) * sqrt(rightNorm))
    }

    private fun ScoredChunk.toCitation(): SourceCitation {
        val draft = chunk.draft
        return SourceCitation(
            sourceId = draft.sourceId,
            source = draft.sourceName,
            section = draft.section,
            chunkId = draft.chunkId,
            location = draft.location,
            score = cosineScore,
            quote = draft.text.take(MAX_CITATION_LENGTH),
        )
    }

    companion object {
        const val MAX_SELECTED_FACTS = 3
        const val MIN_MEMORY_SCORE = 0.20
        private const val MAX_TASK_STATE_STRING_LENGTH = 500
        private const val MAX_TASK_STATE_ITEMS = 20
        private const val MAX_CITATION_LENGTH = 300
        const val ABSTENTION = "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник."
    }
}
