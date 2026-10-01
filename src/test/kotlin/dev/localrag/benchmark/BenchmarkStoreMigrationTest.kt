package dev.localrag.benchmark

import java.nio.file.Path
import java.sql.DriverManager
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import org.junit.jupiter.api.io.TempDir

class BenchmarkStoreMigrationTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `adding model and retrieval fields preserves previous answers ratings and notes`() {
        val database = temporaryDirectory.resolve("legacy.sqlite")
        DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
            connection.createStatement().use { statement ->
                statement.execute(
                    """CREATE TABLE benchmark_questions (
                        question_id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        expected_facts_json TEXT NOT NULL,
                        expected_sources_json TEXT NOT NULL,
                        position INTEGER NOT NULL UNIQUE
                    )""",
                )
                statement.execute(
                    """CREATE TABLE benchmark_results (
                        question_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        question TEXT NOT NULL,
                        expected_facts_json TEXT NOT NULL,
                        expected_sources_json TEXT NOT NULL,
                        retrieved_sources_json TEXT NOT NULL,
                        expected_source_hit INTEGER NOT NULL,
                        baseline_answer TEXT NOT NULL,
                        rag_answer TEXT NOT NULL,
                        baseline_rating TEXT,
                        rag_rating TEXT,
                        note TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(question_id,strategy)
                    )""",
                )
            }
            connection.prepareStatement(
                "INSERT INTO benchmark_questions VALUES (?,?,?,?,?)",
            ).use { statement ->
                statement.setString(1, "legacy-question")
                statement.setString(2, "Legacy question")
                statement.setString(3, "[\"legacy expected fact\"]")
                statement.setString(4, "[]")
                statement.setInt(5, 0)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                """INSERT INTO benchmark_results(
                    question_id,strategy,question,expected_facts_json,expected_sources_json,retrieved_sources_json,
                    expected_source_hit,baseline_answer,rag_answer,baseline_rating,rag_rating,note
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            ).use { statement ->
                statement.setString(1, "legacy-question")
                statement.setString(2, "FIXED_SIZE")
                statement.setString(3, "Legacy question")
                statement.setString(4, "[\"legacy expected fact\"]")
                statement.setString(5, "[]")
                statement.setString(6, "[]")
                statement.setInt(7, 1)
                statement.setString(8, "Legacy baseline")
                statement.setString(9, "Legacy RAG")
                statement.setString(10, "PASS")
                statement.setString(11, "PARTIAL")
                statement.setString(12, "Keep this note")
                statement.executeUpdate()
            }
        }

        BenchmarkStore(database).use { store ->
            val result = store.results().single()
            assertEquals("Legacy baseline", result.baselineAnswer)
            assertEquals("Legacy RAG", result.ragAnswer)
            assertEquals("PASS", result.baselineRating)
            assertEquals("PARTIAL", result.ragRating)
            assertEquals("Keep this note", result.note)
            assertNull(result.providerId)
            assertNull(result.modelId)
            assertNull(result.rawRetrievedSources)
            assertNull(result.rawExpectedSourceRank)
            assertNull(result.expectedSourceRank)
        }
    }
}
