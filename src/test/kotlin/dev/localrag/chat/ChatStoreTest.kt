package dev.localrag.chat

import dev.localrag.benchmark.BenchmarkQuestion
import dev.localrag.benchmark.BenchmarkExpectedSource
import dev.localrag.benchmark.BenchmarkStore
import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredChunk
import dev.localrag.domain.StoredSource
import dev.localrag.index.SqliteIndexRepository
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import java.sql.DriverManager
import java.util.UUID

class ChatStoreTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `sessions persist user turns and keep new titles neutral until generation`() {
        val database = temporaryDirectory.resolve("chat.sqlite")
        ChatStore(database).use { store ->
            val first = store.createSession()
            val second = store.createSession()
            val turnId = UUID.randomUUID().toString()

            val saved = store.saveUserMessage(first.session.id, turnId, "  Импортируй выписку  ")
            val repeated = store.saveUserMessage(first.session.id, turnId, "  Импортируй выписку  ")
            assertEquals(saved.id, repeated.id)
            assertEquals(1, store.sessionDetail(first.session.id)?.session?.messageCount)
            assertEquals(ChatStore.NEW_SESSION_TITLE, store.sessionDetail(first.session.id)?.session?.title)
            assertEquals("Новая сессия", store.sessionDetail(second.session.id)?.session?.title)
            assertEquals(1, store.sessions().first().messageCount)
            assertTrue(runCatching { store.saveUserMessage(first.session.id, turnId, "изменённый текст") }.isFailure)

            assertTrue(store.deleteSession(first.session.id))
            assertNull(store.sessionDetail(first.session.id))
            assertFalse(store.deleteSession(first.session.id))
            assertEquals(second.session.id, store.sessions().single().id)
        }
    }

    @Test
    fun `chat schema migration preserves existing source chunks benchmark review and selection`() {
        val database = temporaryDirectory.resolve("existing.sqlite")
        val sourceId = UUID.randomUUID().toString()
        val questions = (1..10).map { index ->
            BenchmarkQuestion(
                UUID.randomUUID().toString(),
                "Вопрос $index?",
                listOf("Факт $index"),
                listOf(BenchmarkExpectedSource(sourceId, section = "Раздел $index")),
            )
        }
        val legacyChunk = StoredChunk(
            ChunkDraft(
                strategy = ChunkStrategy.STRUCTURAL,
                chunkId = "legacy-chunk",
                sourceId = sourceId,
                sourceName = "fixture.txt",
                section = "Legacy",
                location = SourceLocation(lineStart = 1, lineEnd = 1),
                tokenUnits = 3,
                text = "Legacy chunk text",
            ),
            embedding = listOf(1f, 0f),
            embeddingModel = "fixture-embedding",
        )
        SqliteIndexRepository(database).use { index ->
            index.registerSource(
                StoredSource(
                    SourceRecord(
                        sourceId = sourceId,
                        name = "fixture.txt",
                        type = SourceType.TEXT,
                        sizeBytes = 10,
                        status = SourceStatus.PENDING,
                        createdAt = "2026-10-02T00:00:00Z",
                    ),
                    null,
                ),
            )
            index.markSourceStatus(sourceId, SourceStatus.INDEXING)
            index.saveChunks(sourceId, listOf(legacyChunk))
            index.markSourceStatus(sourceId, SourceStatus.READY)
        }
        val expectedResult = BenchmarkStore(database).use { store ->
            store.saveQuestions(questions)
            val question = questions.first()
            store.saveResult(
                question = question,
                strategy = ChunkStrategy.STRUCTURAL,
                rawSources = emptyList(),
                sources = emptyList(),
                baselineAnswer = "Legacy baseline",
                ragAnswer = "Legacy RAG",
                rawExpectedSourceRank = null,
                expectedSourceRank = null,
                selection = ModelSelection("deepseek", "deepseek-flash"),
            )
            store.saveReview(question.id, ChunkStrategy.STRUCTURAL, "PASS", "PARTIAL", "Keep this review.")
            store.results().single()
        }
        val modelSelectionPath = temporaryDirectory.resolve("model-selection.json")
        val savedModelSelection = """{"providerId":"deepseek","modelId":"deepseek-v4-pro"}"""
        java.nio.file.Files.writeString(modelSelectionPath, savedModelSelection)

        ChatStore(database).use { store ->
            assertTrue(store.sessions().isEmpty())
            assertEquals(ChatStore.CHAT_SCHEMA_VERSION, userVersion(database))
        }

        SqliteIndexRepository(database).use { index ->
            assertEquals(sourceId, index.source(sourceId)?.sourceId)
            val restoredChunk = index.search(
                ChunkStrategy.STRUCTURAL,
                listOf(1f, 0f),
                "fixture-embedding",
                20,
            ).single().chunk
            assertEquals(legacyChunk, restoredChunk)
            BenchmarkStore(database).use { store ->
                assertEquals(questions, store.questions())
                assertEquals(expectedResult, store.results().single())
            }
        }
        assertEquals(savedModelSelection, java.nio.file.Files.readString(modelSelectionPath))
        assertEquals(ChatStore.CHAT_SCHEMA_VERSION, userVersion(database))
    }

    @Test
    fun `title migration preserves legacy names and generated titles freeze after first completion`() {
        val database = temporaryDirectory.resolve("legacy-title.sqlite")
        val legacySessionId = UUID.randomUUID().toString()
        DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
            connection.createStatement().use { statement ->
                statement.execute(
                    "CREATE TABLE chat_sessions(session_id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)",
                )
            }
            connection.prepareStatement(
                "INSERT INTO chat_sessions(session_id,title,created_at,updated_at) VALUES (?,?,?,?)",
            ).use { statement ->
                statement.setString(1, legacySessionId)
                statement.setString(2, "Legacy user-visible title")
                statement.setString(3, "2026-10-01T00:00:00Z")
                statement.setString(4, "2026-10-01T00:00:00Z")
                statement.executeUpdate()
            }
        }

        ChatStore(database).use { store ->
            assertTrue(store.sessionTitleGenerated(legacySessionId))
            store.saveUserMessage(legacySessionId, UUID.randomUUID().toString(), "Legacy conversation")
            assertEquals("Legacy user-visible title", store.sessionDetail(legacySessionId)?.session?.title)

            val created = store.createSession().session.id
            assertFalse(store.sessionTitleGenerated(created))
            val turnId = UUID.randomUUID().toString()
            store.saveUserMessage(created, turnId, "Initial request that must not become the title")
            assertEquals(ChatStore.NEW_SESSION_TITLE, store.sessionDetail(created)?.session?.title)
            store.completeTurn(
                sessionId = created,
                turnId = turnId,
                answer = "Synthetic answer",
                citations = emptyList(),
                memoryReferences = emptyList(),
                taskState = ChatTaskState(),
                sessionTitle = "Context-based synthetic title",
            )
            assertTrue(store.sessionTitleGenerated(created))
            assertEquals("Context-based synthetic title", store.sessionDetail(created)?.session?.title)
        }
    }

    @Test
    fun `user message content is restored after store reopen`() {
        val database = temporaryDirectory.resolve("restart.sqlite")
        val sessionId: String
        val turnId = UUID.randomUUID().toString()
        ChatStore(database).use { store ->
            sessionId = store.createSession().session.id
            store.saveUserMessage(sessionId, turnId, "Срок оплаты — пятнадцатое число")
        }
        ChatStore(database).use { store ->
            val restored = requireNotNull(store.sessionDetail(sessionId))
            assertEquals("Срок оплаты — пятнадцатое число", restored.messages.single().content)
            assertEquals(ChatTaskState(), restored.taskState)
        }
    }

    private fun userVersion(database: Path): Int = DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
        connection.createStatement().use { statement ->
            statement.executeQuery("PRAGMA user_version").use { rows ->
                assertTrue(rows.next())
                rows.getInt(1)
            }
        }
    }
}
