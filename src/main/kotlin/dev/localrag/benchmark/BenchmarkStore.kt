package dev.localrag.benchmark

import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.SourceCitation
import dev.localrag.domain.SourceLocation
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.util.UUID

@Serializable
data class BenchmarkExpectedSource(
    val sourceId: String,
    val location: SourceLocation = SourceLocation(),
    val section: String? = null,
)

@Serializable
data class BenchmarkQuestion(
    val id: String,
    val question: String,
    val expectedFacts: List<String>,
    val expectedSources: List<BenchmarkExpectedSource>,
)

@Serializable
data class BenchmarkQuestionSet(
    val questions: List<BenchmarkQuestion>,
    val errors: List<String> = emptyList(),
    val runnable: Boolean = false,
)

@Serializable
data class BenchmarkResult(
    val questionId: String,
    val question: String,
    val expectedFacts: List<String>,
    val expectedSources: List<BenchmarkExpectedSource>,
    val strategy: ChunkStrategy,
    val retrievedSources: List<SourceCitation>,
    val expectedSourceHit: Boolean,
    val baselineAnswer: String,
    val ragAnswer: String,
    val providerId: String? = null,
    val modelId: String? = null,
    val baselineRating: String? = null,
    val ragRating: String? = null,
    val note: String? = null,
    val rawRetrievedSources: List<SourceCitation>? = null,
    val rawExpectedSourceRank: Int? = null,
    val expectedSourceRank: Int? = null,
)

class BenchmarkStore(databasePath: Path) : AutoCloseable {
    private val connection: Connection = DriverManager.getConnection("jdbc:sqlite:${databasePath.toAbsolutePath().normalize()}")

    init {
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA foreign_keys=ON")
            statement.execute("PRAGMA busy_timeout=5000")
            statement.execute("PRAGMA journal_mode=WAL")
            statement.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_questions (
                    question_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_facts_json TEXT NOT NULL,
                    expected_sources_json TEXT NOT NULL,
                    position INTEGER NOT NULL UNIQUE
                )""".trimIndent(),
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_results (
                    question_id TEXT NOT NULL REFERENCES benchmark_questions(question_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL CHECK(strategy IN ('FIXED_SIZE','STRUCTURAL')),
                    provider_id TEXT,
                    model_id TEXT,
                    question TEXT NOT NULL,
                    expected_facts_json TEXT NOT NULL,
                    expected_sources_json TEXT NOT NULL,
                    retrieved_sources_json TEXT NOT NULL,
                    raw_retrieved_sources_json TEXT,
                    expected_source_hit INTEGER NOT NULL,
                    raw_expected_source_rank INTEGER,
                    expected_source_rank INTEGER,
                    baseline_answer TEXT NOT NULL,
                    rag_answer TEXT NOT NULL,
                    baseline_rating TEXT CHECK(baseline_rating IS NULL OR baseline_rating IN ('PASS','PARTIAL','FAIL')),
                    rag_rating TEXT CHECK(rag_rating IS NULL OR rag_rating IN ('PASS','PARTIAL','FAIL')),
                    note TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(question_id,strategy)
                )""".trimIndent(),
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_result_sources (
                    question_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    PRIMARY KEY(question_id,strategy,source_id),
                    FOREIGN KEY(question_id,strategy) REFERENCES benchmark_results(question_id,strategy) ON DELETE CASCADE
                )""".trimIndent(),
            )
            statement.execute("CREATE INDEX IF NOT EXISTS benchmark_source_refs_idx ON benchmark_result_sources(source_id)")
        }
        ensureBenchmarkResultColumns()
    }

    private fun ensureBenchmarkResultColumns() {
        val columns = connection.createStatement().use { statement ->
            statement.executeQuery("PRAGMA table_info(benchmark_results)").use { rows ->
                buildSet {
                    while (rows.next()) add(rows.getString("name"))
                }
            }
        }
        connection.createStatement().use { statement ->
            if ("provider_id" !in columns) statement.execute("ALTER TABLE benchmark_results ADD COLUMN provider_id TEXT")
            if ("model_id" !in columns) statement.execute("ALTER TABLE benchmark_results ADD COLUMN model_id TEXT")
            if ("raw_retrieved_sources_json" !in columns) {
                statement.execute("ALTER TABLE benchmark_results ADD COLUMN raw_retrieved_sources_json TEXT")
            }
            if ("raw_expected_source_rank" !in columns) {
                statement.execute("ALTER TABLE benchmark_results ADD COLUMN raw_expected_source_rank INTEGER")
            }
            if ("expected_source_rank" !in columns) {
                statement.execute("ALTER TABLE benchmark_results ADD COLUMN expected_source_rank INTEGER")
            }
        }
    }

    @Synchronized
    fun saveQuestions(questions: List<BenchmarkQuestion>) {
        validateQuestions(questions)
        transaction {
            connection.createStatement().use { it.executeUpdate("DELETE FROM benchmark_questions") }
            connection.prepareStatement(
                "INSERT INTO benchmark_questions(question_id,question,expected_facts_json,expected_sources_json,position) VALUES (?,?,?,?,?)",
            ).use { statement ->
                questions.forEachIndexed { position, question ->
                    statement.setString(1, question.id)
                    statement.setString(2, question.question)
                    statement.setString(3, Json.encodeToString(question.expectedFacts))
                    statement.setString(4, Json.encodeToString(question.expectedSources))
                    statement.setInt(5, position)
                    statement.addBatch()
                }
                statement.executeBatch()
            }
        }
    }

    @Synchronized
    fun questions(): List<BenchmarkQuestion> = connection.prepareStatement(
        "SELECT question_id,question,expected_facts_json,expected_sources_json FROM benchmark_questions ORDER BY position",
    ).use { statement ->
        statement.executeQuery().use { rows ->
            buildList {
                while (rows.next()) {
                    add(
                        BenchmarkQuestion(
                            id = rows.getString(1),
                            question = rows.getString(2),
                            expectedFacts = Json.decodeFromString(rows.getString(3)),
                            expectedSources = Json.decodeFromString(rows.getString(4)),
                        ),
                    )
                }
            }
        }
    }

    @Synchronized
    fun saveResult(
        question: BenchmarkQuestion,
        strategy: ChunkStrategy,
        rawSources: List<ScoredChunk>,
        sources: List<ScoredChunk>,
        baselineAnswer: String,
        ragAnswer: String,
        rawExpectedSourceRank: Int?,
        expectedSourceRank: Int?,
        selection: ModelSelection,
    ) {
        val rawCitations = rawSources.map { it.toCitation() }
        val citations = sources.map { it.toCitation() }
        val hit = expectedSourceRank != null
        transaction {
            connection.prepareStatement("DELETE FROM benchmark_result_sources WHERE question_id=? AND strategy=?").use { statement ->
                statement.setString(1, question.id)
                statement.setString(2, strategy.name)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                """INSERT INTO benchmark_results(
                    question_id,strategy,provider_id,model_id,question,expected_facts_json,expected_sources_json,
                    retrieved_sources_json,raw_retrieved_sources_json,expected_source_hit,raw_expected_source_rank,
                    expected_source_rank,baseline_answer,rag_answer,baseline_rating,rag_rating,note,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,CURRENT_TIMESTAMP)
                ON CONFLICT(question_id,strategy) DO UPDATE SET
                    provider_id=excluded.provider_id,
                    model_id=excluded.model_id,
                    question=excluded.question,
                    expected_facts_json=excluded.expected_facts_json,
                    expected_sources_json=excluded.expected_sources_json,
                    retrieved_sources_json=excluded.retrieved_sources_json,
                    raw_retrieved_sources_json=excluded.raw_retrieved_sources_json,
                    expected_source_hit=excluded.expected_source_hit,
                    raw_expected_source_rank=excluded.raw_expected_source_rank,
                    expected_source_rank=excluded.expected_source_rank,
                    baseline_answer=excluded.baseline_answer,
                    rag_answer=excluded.rag_answer,
                    baseline_rating=NULL,
                    rag_rating=NULL,
                    note=NULL,
                    updated_at=CURRENT_TIMESTAMP""".trimIndent(),
            ).use { statement ->
                statement.setString(1, question.id)
                statement.setString(2, strategy.name)
                statement.setString(3, selection.providerId)
                statement.setString(4, selection.modelId)
                statement.setString(5, question.question)
                statement.setString(6, Json.encodeToString(question.expectedFacts))
                statement.setString(7, Json.encodeToString(question.expectedSources))
                statement.setString(8, Json.encodeToString(citations))
                statement.setString(9, Json.encodeToString(rawCitations))
                statement.setInt(10, if (hit) 1 else 0)
                if (rawExpectedSourceRank == null) statement.setNull(11, java.sql.Types.INTEGER)
                else statement.setInt(11, rawExpectedSourceRank)
                if (expectedSourceRank == null) statement.setNull(12, java.sql.Types.INTEGER)
                else statement.setInt(12, expectedSourceRank)
                statement.setString(13, baselineAnswer)
                statement.setString(14, ragAnswer)
                statement.executeUpdate()
            }
            val sourceIds = (
                question.expectedSources.map(BenchmarkExpectedSource::sourceId) +
                    rawCitations.map(SourceCitation::sourceId) +
                    citations.map(SourceCitation::sourceId)
                ).distinct()
            connection.prepareStatement("INSERT INTO benchmark_result_sources(question_id,strategy,source_id) VALUES (?,?,?)").use { statement ->
                sourceIds.forEach { sourceId ->
                    statement.setString(1, question.id)
                    statement.setString(2, strategy.name)
                    statement.setString(3, sourceId)
                    statement.addBatch()
                }
                statement.executeBatch()
            }
        }
    }

    @Synchronized
    fun results(): List<BenchmarkResult> = connection.prepareStatement(
        """SELECT question_id,question,expected_facts_json,expected_sources_json,strategy,provider_id,model_id,
            retrieved_sources_json,raw_retrieved_sources_json,expected_source_hit,raw_expected_source_rank,
            expected_source_rank,baseline_answer,rag_answer,baseline_rating,rag_rating,note
            FROM benchmark_results ORDER BY question_id,CASE strategy WHEN 'FIXED_SIZE' THEN 0 ELSE 1 END""".trimIndent(),
    ).use { statement ->
        statement.executeQuery().use { rows ->
            buildList {
                while (rows.next()) {
                    add(
                        BenchmarkResult(
                            questionId = rows.getString(1),
                            question = rows.getString(2),
                            expectedFacts = Json.decodeFromString(rows.getString(3)),
                            expectedSources = Json.decodeFromString(rows.getString(4)),
                            strategy = ChunkStrategy.valueOf(rows.getString(5)),
                            providerId = rows.getString(6),
                            modelId = rows.getString(7),
                            retrievedSources = Json.decodeFromString(rows.getString(8)),
                            expectedSourceHit = rows.getInt(10) == 1,
                            baselineAnswer = rows.getString(13),
                            ragAnswer = rows.getString(14),
                            baselineRating = rows.getString(15),
                            ragRating = rows.getString(16),
                            note = rows.getString(17),
                            rawRetrievedSources = rows.getString(9)?.let { Json.decodeFromString<List<SourceCitation>>(it) },
                            rawExpectedSourceRank = rows.getInt(11).let { if (rows.wasNull()) null else it },
                            expectedSourceRank = rows.getInt(12).let { if (rows.wasNull()) null else it },
                        ),
                    )
                }
            }
        }
    }

    @Synchronized
    fun saveReview(questionId: String, strategy: ChunkStrategy, baselineRating: String, ragRating: String, note: String?) {
        require(baselineRating in RATINGS && ragRating in RATINGS) { "Оценка должна быть PASS, PARTIAL или FAIL." }
        require(note == null || note.length <= MAX_NOTE_CHARS) { "Заметка не должна превышать $MAX_NOTE_CHARS символов." }
        connection.prepareStatement(
            "UPDATE benchmark_results SET baseline_rating=?,rag_rating=?,note=?,updated_at=CURRENT_TIMESTAMP WHERE question_id=? AND strategy=?",
        ).use { statement ->
            statement.setString(1, baselineRating)
            statement.setString(2, ragRating)
            statement.setString(3, note)
            statement.setString(4, questionId)
            statement.setString(5, strategy.name)
            require(statement.executeUpdate() == 1) { "Результат benchmark для этого вопроса и стратегии не найден." }
        }
    }

    @Synchronized
    fun deleteSourceReferences(sourceId: String) {
        transaction {
            connection.prepareStatement(
                """DELETE FROM benchmark_results WHERE EXISTS (
                    SELECT 1 FROM benchmark_result_sources r
                    WHERE r.question_id=benchmark_results.question_id
                    AND r.strategy=benchmark_results.strategy AND r.source_id=?
                )""".trimIndent(),
            ).use { statement ->
                statement.setString(1, sourceId)
                statement.executeUpdate()
            }
        }
    }

    private fun validateQuestions(questions: List<BenchmarkQuestion>) {
        require(questions.size == EXPECTED_QUESTION_COUNT) { "JSON benchmark должен содержать ровно $EXPECTED_QUESTION_COUNT вопросов." }
        require(questions.map(BenchmarkQuestion::id).toSet().size == questions.size) { "ID вопросов должны быть уникальными." }
        questions.forEach { question ->
            require(question.id.length in 1..MAX_ID_CHARS && question.id.all { it.isLetterOrDigit() || it in "-_" }) {
                "ID вопроса должен содержать только буквы, цифры, дефис или подчёркивание."
            }
            require(question.question.isNotBlank() && question.question.length <= MAX_QUESTION_CHARS) {
                "Текст каждого вопроса должен содержать от 1 до $MAX_QUESTION_CHARS символов."
            }
            require(question.expectedFacts.isNotEmpty() && question.expectedFacts.size <= MAX_EXPECTED_FACTS &&
                question.expectedFacts.all { it.isNotBlank() && it.length <= MAX_EXPECTATION_CHARS }) {
                "Каждый вопрос должен содержать ожидаемые факты."
            }
            require(question.expectedSources.isNotEmpty() && question.expectedSources.size <= MAX_EXPECTED_SOURCES) {
                "Каждый вопрос должен ссылаться хотя бы на один ожидаемый источник."
            }
            question.expectedSources.forEach { expected ->
                require(isUuid(expected.sourceId)) { "Ожидаемый источник должен указывать sourceId из коллекции." }
                require(expected.location.isDefined || !expected.section.isNullOrBlank()) {
                    "Ожидаемый источник должен содержать диапазон страницы/строк или название раздела."
                }
                require(expected.section == null || expected.section.length <= MAX_SECTION_CHARS) {
                    "Ожидаемый раздел не должен превышать $MAX_SECTION_CHARS символов."
                }
            }
        }
    }


    private fun ScoredChunk.toCitation() = SourceCitation(
        sourceId = chunk.draft.sourceId,
        source = chunk.draft.sourceName,
        section = chunk.draft.section,
        chunkId = chunk.draft.chunkId,
        location = chunk.draft.location,
        score = cosineScore,
        quote = chunk.draft.text.take(MAX_CITATION_QUOTE_LENGTH),
    )

    private inline fun <T> transaction(block: () -> T): T {
        val oldAutoCommit = connection.autoCommit
        connection.autoCommit = false
        try {
            val value = block()
            connection.commit()
            return value
        } catch (error: Exception) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = oldAutoCommit
        }
    }

    private fun isUuid(value: String): Boolean = runCatching { UUID.fromString(value).toString() == value }.getOrDefault(false)

    @Synchronized
    override fun close() = connection.close()

    companion object {
        const val EXPECTED_QUESTION_COUNT = 10
        private const val MAX_ID_CHARS = 80
        private const val MAX_QUESTION_CHARS = 4_000
        private const val MAX_EXPECTED_FACTS = 20
        private const val MAX_EXPECTED_SOURCES = 20
        private const val MAX_EXPECTATION_CHARS = 2_000
        private const val MAX_SECTION_CHARS = 160
        private const val MAX_NOTE_CHARS = 2_000
        private val RATINGS = setOf("PASS", "PARTIAL", "FAIL")
        private const val MAX_CITATION_QUOTE_LENGTH = 300
    }
}
