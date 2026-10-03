package dev.localrag.chat

import dev.localrag.app.RagService
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.EmbeddingPort
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.RerankPort
import dev.localrag.index.SqliteIndexRepository
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ConcurrentLinkedQueue

class ChatServiceTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `turns classify session and shared memory automatically and bound history`() {
        val database = temporaryDirectory.resolve("chat.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val generator = FixtureGenerator()
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), generator, embeddings)
                val first = store.createSession().session.id
                val second = store.createSession().session.id

                generator.enqueue(
                    ChatTurnCompletion(
                        "Конечно, продолжим с новой целью.",
                        ChatTaskStateDelta(
                            goal = "Импортировать transaction-import",
                            clarificationsToAdd = listOf("Формат CSV"),
                            constraintsToAdd = listOf("Не изменять исходный файл"),
                            termsToAdd = listOf(ChatTaskTerm("transaction-import", "Импорт банковских операций")),
                        ),
                        memoryUpdates = listOf(
                            ChatMemoryUpdate(ChatMemoryScope.SESSION, "Ограничение: не употреблять орехи"),
                            ChatMemoryUpdate(ChatMemoryScope.SHARED, "Preferred currency is USD"),
                        ),
                    ),
                )
                val firstTurn = service.turn(first, UUID.randomUUID().toString(), "Как импортировать?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertEquals("Конечно, продолжим с новой целью.", firstTurn.message.content)
                assertEquals("Импортировать transaction-import", firstTurn.taskState.goal)
                assertTrue(store.memoryFacts(first).any { it.scope == ChatMemoryScope.SESSION && it.text == "Ограничение: не употреблять орехи" })
                assertTrue(store.sharedFacts().any { it.text == "Preferred currency is USD" })

                generator.enqueue(
                    ChatTurnCompletion(
                        "Shared-memory response",
                        ChatTaskStateDelta(),
                        memoryReferenceRefs = listOf("M1"),
                    ),
                )
                val secondTurn = service.turn(second, UUID.randomUUID().toString(), "Какая валюта предпочтительна?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertEquals("Shared-memory response", secondTurn.message.content)
                assertTrue(generator.requests.last().sessionFacts.isEmpty())
                assertEquals("Preferred currency is USD", generator.requests.last().sharedFacts.single().text)
                assertTrue(store.memoryFacts(second).none { it.text == "Ограничение: не употреблять орехи" })

                generator.enqueue(ChatTurnCompletion("Session answer", ChatTaskStateDelta()))
                service.turn(first, UUID.randomUUID().toString(), "Что было сказано в этой сессии?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertTrue(generator.requests.last().sessionFacts.any { it.text == "Ограничение: не употреблять орехи" })

                repeat(12) { indexTurn ->
                    generator.enqueue(ChatTurnCompletion("Bounded answer", ChatTaskStateDelta()))
                    service.turn(first, UUID.randomUUID().toString(), "Уточнение $indexTurn", ChunkStrategy.STRUCTURAL, SELECTION)
                }
                val lastRequest = generator.requests.last()
                assertEquals(ChatStore.MAX_HISTORY_MESSAGES, lastRequest.previousMessages.size)
                assertTrue(lastRequest.previousMessages.none { it.content == "Какая валюта предпочтительна?" })
                assertEquals(14, requireNotNull(store.sessionDetail(first)).session.messageCount / 2)
                service.updateTaskState(first, ChatTaskState(goal = "Ручная правка цели"))
                assertTrue(store.memoryFacts(first).any { it.text == "Ограничение: не употреблять орехи" })
            }
        }
    }

    @Test
    fun `invalid selected reference leaves completion updates atomic and retry reuses user turn`() {
        val database = temporaryDirectory.resolve("invalid-reference.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val generator = FixtureGenerator()
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), generator, embeddings)
                val sessionId = store.createSession().session.id
                val turnId = UUID.randomUUID().toString()
                generator.enqueue(
                    ChatTurnCompletion(
                        "Invalid completion",
                        ChatTaskStateDelta(goal = "Must not persist"),
                        memoryUpdates = listOf(ChatMemoryUpdate(ChatMemoryScope.SHARED, "Must not persist")),
                        memoryReferenceRefs = listOf("M99"),
                    ),
                )

                assertTrue(
                    runCatching {
                        service.turn(sessionId, turnId, "Atomic retry", ChunkStrategy.STRUCTURAL, SELECTION)
                    }.isFailure,
                )
                assertEquals(ChatTaskState(), store.taskState(sessionId))
                assertTrue(store.assistantMessage(sessionId, turnId) == null)
                assertTrue(store.sharedFacts().isEmpty())
                assertFalse(store.sessionTitleGenerated(sessionId))

                generator.enqueue(
                    ChatTurnCompletion(
                        "Valid retry",
                        ChatTaskStateDelta(),
                        memoryUpdates = listOf(ChatMemoryUpdate(ChatMemoryScope.SHARED, "Persisted after retry")),
                    ),
                )
                val response = service.turn(sessionId, turnId, "Atomic retry", ChunkStrategy.STRUCTURAL, SELECTION)
                assertEquals("Valid retry", response.message.content)
                val messages = requireNotNull(store.sessionDetail(sessionId)).messages
                assertEquals(1, messages.count { it.role == ChatMessageRole.USER })
                assertEquals(1, messages.count { it.role == ChatMessageRole.ASSISTANT })
                assertEquals("Persisted after retry", store.sharedFacts().single().text)
                assertTrue(store.sessionTitleGenerated(sessionId))
            }
        }
    }

    @Test
    fun `task state remains bounded and failed overflow keeps prior state and user turn`() {
        val database = temporaryDirectory.resolve("bounded-state.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val generator = FixtureGenerator()
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), generator, embeddings)
                val sessionId = store.createSession().session.id
                val state = ChatTaskState(clarifications = (1..20).map { "Clarification $it" })
                service.updateTaskState(sessionId, state)
                val turnId = UUID.randomUUID().toString()
                generator.enqueue(
                    ChatTurnCompletion(
                        "Synthetic answer",
                        ChatTaskStateDelta(clarificationsToAdd = listOf("Clarification 21")),
                    ),
                )

                val failure = runCatching {
                    service.turn(sessionId, turnId, "Add one more clarification.", ChunkStrategy.STRUCTURAL, SELECTION)
                }.exceptionOrNull()

                assertTrue(failure is IllegalArgumentException)
                assertTrue(failure?.message?.contains("20") == true)
                assertEquals(state, store.taskState(sessionId))
                assertTrue(store.assistantMessage(sessionId, turnId) == null)
                assertEquals(1, requireNotNull(store.sessionDetail(sessionId)).messages.size)
            }
        }
    }

    @Test
    fun `maximum valid term stays in task state without creating duplicate memory facts`() {
        val database = temporaryDirectory.resolve("maximum-term.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), FixtureGenerator(), embeddings)
                val sessionId = store.createSession().session.id
                val term = ChatTaskTerm("t".repeat(500), "d".repeat(500))

                val saved = service.updateTaskState(sessionId, ChatTaskState(terms = listOf(term)))

                assertEquals(term, saved.terms.single())
                assertTrue(store.memoryFacts(sessionId).isEmpty())
            }
        }
    }


    @Test
    fun `turn generation serializes state updates within the service`() {
        val database = temporaryDirectory.resolve("serialized-turns.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val firstEntered = java.util.concurrent.CountDownLatch(1)
                val secondEntered = java.util.concurrent.CountDownLatch(1)
                val secondStarted = java.util.concurrent.CountDownLatch(1)
                val releaseFirst = java.util.concurrent.CountDownLatch(1)
                val generator = object : ChatTurnGenerator {
                    override fun completeTurn(selection: ModelSelection, request: ChatTurnRequest): ChatTurnCompletion {
                        if (request.question == "First") {
                            firstEntered.countDown()
                            check(releaseFirst.await(5, java.util.concurrent.TimeUnit.SECONDS))
                        } else {
                            secondEntered.countDown()
                        }
                        return ChatTurnCompletion(
                            request.question,
                            ChatTaskStateDelta(clarificationsToAdd = listOf(request.question)),
                            sessionTitle = if (request.generateTitle) "Synthetic chat" else null,
                        )
                    }
                }
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), generator, embeddings)
                val sessionId = store.createSession().session.id
                val executor = java.util.concurrent.Executors.newFixedThreadPool(2)
                try {
                    val first = executor.submit {
                        service.turn(sessionId, UUID.randomUUID().toString(), "First", ChunkStrategy.STRUCTURAL, SELECTION)
                    }
                    assertTrue(firstEntered.await(5, java.util.concurrent.TimeUnit.SECONDS))
                    val second = executor.submit {
                        secondStarted.countDown()
                        service.turn(sessionId, UUID.randomUUID().toString(), "Second", ChunkStrategy.STRUCTURAL, SELECTION)
                    }
                    assertTrue(secondStarted.await(5, java.util.concurrent.TimeUnit.SECONDS))
                    assertFalse(secondEntered.await(100, java.util.concurrent.TimeUnit.MILLISECONDS))
                    releaseFirst.countDown()
                    first.get(5, java.util.concurrent.TimeUnit.SECONDS)
                    second.get(5, java.util.concurrent.TimeUnit.SECONDS)

                    assertEquals(setOf("First", "Second"), store.taskState(sessionId).clarifications.toSet())
                } finally {
                    releaseFirst.countDown()
                    executor.shutdownNow()
                }
            }
        }
    }
    private class FixtureEmbeddings : EmbeddingPort {
        override val modelName = "fixture-embedding"
        override fun embed(texts: List<String>): List<List<Float>> = texts.map { listOf(1f, 0f) }
    }

    private class FixtureGenerator : ChatTurnGenerator {
        val requests = CopyOnWriteArrayList<ChatTurnRequest>()
        private val responses = ConcurrentLinkedQueue<ChatTurnCompletion>()

        fun enqueue(response: ChatTurnCompletion) { responses.add(response) }

        override fun completeTurn(selection: ModelSelection, request: ChatTurnRequest): ChatTurnCompletion {
            requests += request
            val response = requireNotNull(responses.poll())
            return if (request.generateTitle && response.sessionTitle == null) {
                response.copy(sessionTitle = "Synthetic chat")
            } else {
                response
            }
        }
    }

    private object EmptyChatPort : dev.localrag.domain.ChatPort {
        override fun answer(
            selection: ModelSelection,
            question: String,
            context: List<dev.localrag.domain.ScoredChunk>,
        ): String = ""
    }
    private object EmptyReranker : RerankPort {
        override fun rerank(selection: ModelSelection, question: String, candidateTexts: List<String>): List<Int> = emptyList()
    }

    companion object {
        private val SELECTION = ModelSelection("fixture", "fixture-chat")
    }
}
