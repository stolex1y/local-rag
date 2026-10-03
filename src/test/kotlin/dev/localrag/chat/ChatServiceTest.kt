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
    fun `session facts stay isolated until explicit shared promotion and history is bounded`() {
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
                        "Synthetic answer",
                        ChatTaskStateDelta(
                            goal = "Импортировать transaction-import",
                            clarificationsToAdd = listOf("Формат CSV"),
                            constraintsToAdd = listOf("Не изменять исходный файл"),
                            termsToAdd = listOf(ChatTaskTerm("transaction-import", "Импорт банковских операций")),
                        ),
                    ),
                )
                val firstTurn = service.turn(first, UUID.randomUUID().toString(), "Как импортировать?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertEquals(ChatService.ABSTENTION, firstTurn.message.content)
                assertEquals("Импортировать transaction-import", firstTurn.taskState.goal)
                assertTrue(service.sessionMemoryFacts(first).any { it.text == "Цель задачи: Импортировать transaction-import" })

                generator.enqueue(ChatTurnCompletion("Unsupported prose", ChatTaskStateDelta()))
                val sessionOnlyTurn = service.turn(
                    first,
                    UUID.randomUUID().toString(),
                    "Что следует делать дальше?",
                    ChunkStrategy.STRUCTURAL,
                    SELECTION,
                )
                assertEquals(ChatService.ABSTENTION, sessionOnlyTurn.message.content)
                assertTrue(sessionOnlyTurn.message.documentCitations.isEmpty())
                assertTrue(sessionOnlyTurn.message.memoryReferences.isEmpty())
                assertTrue(generator.requests.last().sharedFacts.isEmpty())
                val goalFact = service.sessionMemoryFacts(first)
                    .single { it.text == "Цель задачи: Импортировать transaction-import" }

                generator.enqueue(ChatTurnCompletion("Second session answer", ChatTaskStateDelta()))
                service.turn(second, UUID.randomUUID().toString(), "С чего начать?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertTrue(generator.requests.last().sharedFacts.isEmpty())

                assertTrue(runCatching { service.promoteSessionFact(first, goalFact.id, false) }.isFailure)
                service.promoteSessionFact(first, goalFact.id, true)

                generator.enqueue(ChatTurnCompletion("Shared memory answer", ChatTaskStateDelta()))
                val sharedTurn = service.turn(second, UUID.randomUUID().toString(), "Что означает transaction-import?", ChunkStrategy.STRUCTURAL, SELECTION)
                assertEquals("Shared memory answer", sharedTurn.message.content)
                assertEquals("Цель задачи: Импортировать transaction-import", generator.requests.last().sharedFacts.single().text)

                repeat(12) { indexTurn ->
                    generator.enqueue(ChatTurnCompletion("Bounded answer", ChatTaskStateDelta()))
                    service.turn(first, UUID.randomUUID().toString(), "Уточнение $indexTurn", ChunkStrategy.STRUCTURAL, SELECTION)
                }
                val lastRequest = generator.requests.last()
                assertEquals(ChatStore.MAX_HISTORY_MESSAGES, lastRequest.previousMessages.size)
                assertTrue(lastRequest.previousMessages.none { it.content == "С чего начать?" })
                assertFalse(lastRequest.previousMessages.any { it.content.startsWith("Shared memory answer") })
                assertEquals(14, requireNotNull(store.sessionDetail(first)).session.messageCount / 2)
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
    fun `maximum valid term and definition fit in session memory`() {
        val database = temporaryDirectory.resolve("maximum-term.sqlite")
        ChatStore(database).use { store ->
            SqliteIndexRepository(database).use { index ->
                val embeddings = FixtureEmbeddings()
                val service = ChatService(store, RagService(index, embeddings, EmptyChatPort, EmptyReranker), FixtureGenerator(), embeddings)
                val sessionId = store.createSession().session.id
                val term = ChatTaskTerm("t".repeat(500), "d".repeat(500))

                val saved = service.updateTaskState(sessionId, ChatTaskState(terms = listOf(term)))

                assertEquals(term, saved.terms.single())
                assertEquals(1_009, service.sessionMemoryFacts(sessionId).single().text.length)
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
            return requireNotNull(responses.poll())
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
