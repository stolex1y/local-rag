package dev.localrag.index

import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.IndexRepository
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredChunk
import dev.localrag.domain.StoredSource
import dev.localrag.source.LocalRagPaths
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.decodeFromString
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet
import java.sql.SQLException
import java.sql.Types
import java.util.PriorityQueue
import java.util.UUID
import kotlin.math.sqrt

class SqliteIndexRepository(databasePath: Path = defaultDatabasePath()) : IndexRepository {
    private val connection: Connection

    init {
        databasePath.toAbsolutePath().parent?.let(dev.localrag.source.PrivateLocalStorage::prepareDirectory)
        dev.localrag.source.PrivateLocalStorage.prepareFile(databasePath)
        connection = DriverManager.getConnection("jdbc:sqlite:${databasePath.toAbsolutePath().normalize()}")
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA foreign_keys=ON")
            statement.execute("PRAGMA busy_timeout=5000")
            statement.execute("PRAGMA journal_mode=WAL")
            statement.execute(
                """CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('PDF','TEXT','MARKDOWN','HTML','CODE')),
                    size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                    status TEXT NOT NULL CHECK(status IN ('PENDING','INDEXING','READY','FAILED','UNSUPPORTED','DELETING')),
                    error TEXT,
                    created_at TEXT NOT NULL,
                    storage_key TEXT,
                    unsearchable_pages_json TEXT NOT NULL DEFAULT '[]'
                )""".trimIndent(),
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL CHECK(strategy IN ('FIXED_SIZE','STRUCTURAL')),
                    section TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    line_start INTEGER,
                    line_end INTEGER,
                    token_units INTEGER NOT NULL CHECK(token_units > 0),
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL CHECK(embedding_dimension > 0),
                    CHECK ((page_start IS NULL) = (page_end IS NULL)),
                    CHECK ((line_start IS NULL) = (line_end IS NULL)),
                    CHECK (page_start IS NULL OR line_start IS NULL),
                    CHECK (page_start IS NULL OR (page_start > 0 AND page_end >= page_start)),
                    CHECK (line_start IS NULL OR (line_start > 0 AND line_end >= line_start))
                )""".trimIndent(),
            )
            statement.execute("CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks(strategy, embedding_model, source_id)")
            statement.execute("CREATE INDEX IF NOT EXISTS sources_status_idx ON sources(status, created_at)")
            statement.execute("PRAGMA user_version=11")
        }
    }

    @Synchronized
    override fun registerSource(source: StoredSource) {
        val record = source.record
        require(isUuid(record.sourceId)) { "Source IDs must be app-generated UUIDs." }
        require(record.name.isNotBlank() && record.name.length <= 255) { "Source name is invalid." }
        require(record.sizeBytes > 0) { "Source size must be positive." }
        require(record.status in INITIAL_SOURCE_STATUSES) { "A new source must be pending or report an import failure." }
        require(source.storageKey == null || source.storageKey == "${record.sourceId}.source") {
            "Source storage key must be generated from its ID."
        }
        connection.prepareStatement(
            """INSERT INTO sources(source_id,name,type,size_bytes,status,error,created_at,storage_key,unsearchable_pages_json)
                VALUES (?,?,?,?,?,?,?,?,?)""".trimIndent(),
        ).use { statement ->
            statement.setString(1, record.sourceId)
            statement.setString(2, record.name)
            statement.setString(3, record.type.name)
            statement.setLong(4, record.sizeBytes)
            statement.setString(5, record.status.name)
            statement.setString(6, record.error)
            statement.setString(7, record.createdAt)
            statement.setString(8, source.storageKey)
            statement.setString(9, Json.encodeToString(record.unsearchablePages))
            statement.executeUpdate()
        }
    }

    @Synchronized
    override fun source(sourceId: String): SourceRecord? = connection.prepareStatement(
        "SELECT source_id,name,type,size_bytes,status,error,created_at,unsearchable_pages_json FROM sources WHERE source_id=?",
    ).use { statement ->
        statement.setString(1, sourceId)
        statement.executeQuery().use { rows -> if (rows.next()) rows.toSourceRecord() else null }
    }

    @Synchronized
    override fun storedSource(sourceId: String): StoredSource? = connection.prepareStatement(
        "SELECT source_id,name,type,size_bytes,status,error,created_at,unsearchable_pages_json,storage_key FROM sources WHERE source_id=?",
    ).use { statement ->
        statement.setString(1, sourceId)
        statement.executeQuery().use { rows -> if (rows.next()) rows.toStoredSource() else null }
    }
    @Synchronized
    override fun storedSources(): List<StoredSource> = connection.prepareStatement(
        "SELECT source_id,name,type,size_bytes,status,error,created_at,unsearchable_pages_json,storage_key FROM sources ORDER BY created_at,source_id",
    ).use { statement ->
        statement.executeQuery().use { rows -> buildList { while (rows.next()) add(rows.toStoredSource()) } }
    }


    @Synchronized
    override fun sources(): List<SourceRecord> = connection.prepareStatement(
        "SELECT source_id,name,type,size_bytes,status,error,created_at,unsearchable_pages_json FROM sources ORDER BY created_at,source_id",
    ).use { statement ->
        statement.executeQuery().use { rows -> buildList { while (rows.next()) add(rows.toSourceRecord()) } }
    }

    @Synchronized
    override fun sources(status: SourceStatus): List<StoredSource> = connection.prepareStatement(
        "SELECT source_id,name,type,size_bytes,status,error,created_at,unsearchable_pages_json,storage_key FROM sources WHERE status=? ORDER BY created_at,source_id",
    ).use { statement ->
        statement.setString(1, status.name)
        statement.executeQuery().use { rows -> buildList { while (rows.next()) add(rows.toStoredSource()) } }
    }

    @Synchronized
    override fun pendingSources(): List<StoredSource> = sources(SourceStatus.PENDING)
    @Synchronized
    override fun hasSourceLocation(sourceId: String, location: SourceLocation, section: String?): Boolean {
        val source = source(sourceId) ?: return false
        val pageStart = location.pageStart
        val pageEnd = location.pageEnd
        if (pageStart != null && source.unsearchablePages.any { it in pageStart..pageEnd!! }) return false
        val sectionValue = section?.trim()?.takeIf(String::isNotEmpty)
        val rangeCondition = when {
            location.pageStart != null -> "page_start<=? AND page_end>=?"
            location.lineStart != null -> "line_start<=? AND line_end>=?"
            else -> null
        }
        if (rangeCondition == null && sectionValue == null) return false
        val sectionCondition = if (sectionValue == null) "" else " AND instr(lower(section),lower(?))>0"
        val query = "SELECT 1 FROM chunks WHERE source_id=?${if (rangeCondition == null) "" else " AND $rangeCondition"}$sectionCondition LIMIT 1"
        return connection.prepareStatement(query).use { statement ->
            statement.setString(1, sourceId)
            var parameter = 2
            when {
                location.pageStart != null -> {
                    statement.setInt(parameter++, location.pageEnd!!)
                    statement.setInt(parameter++, location.pageStart)
                }
                location.lineStart != null -> {
                    statement.setInt(parameter++, location.lineEnd!!)
                    statement.setInt(parameter++, location.lineStart)
                }
            }
            if (sectionValue != null) statement.setString(parameter, sectionValue.lowercase())
            statement.executeQuery().use { it.next() }
        }
    }


    @Synchronized
    override fun markSourceStatus(
        sourceId: String,
        status: SourceStatus,
        error: String?,
        unsearchablePages: List<Int>,
    ) {
        require(unsearchablePages.all { it > 0 } && unsearchablePages.distinct().size == unsearchablePages.size) {
            "Unsearchable PDF page numbers must be unique and positive."
        }
        val allowedPrevious = when (status) {
            SourceStatus.INDEXING -> "'PENDING'"
            SourceStatus.READY -> "'INDEXING'"
            SourceStatus.FAILED -> "'PENDING','INDEXING'"
            SourceStatus.DELETING -> "'PENDING','READY','FAILED','UNSUPPORTED','DELETING'"
            SourceStatus.PENDING, SourceStatus.UNSUPPORTED -> "''"
        }
        connection.prepareStatement(
            "UPDATE sources SET status=?,error=?,unsearchable_pages_json=? WHERE source_id=? AND status IN ($allowedPrevious)",
        ).use { statement ->
            statement.setString(1, status.name)
            statement.setString(2, error?.take(MAX_ERROR_LENGTH))
            statement.setString(3, Json.encodeToString(unsearchablePages))
            statement.setString(4, sourceId)
            if (statement.executeUpdate() != 1) throw IllegalArgumentException("Состояние источника изменилось или источник не найден.")
        }
    }

    @Synchronized
    override fun clearChunks(sourceId: String) {
        connection.prepareStatement("DELETE FROM chunks WHERE source_id=?").use { statement ->
            statement.setString(1, sourceId)
            statement.executeUpdate()
        }
    }

    @Synchronized
    override fun saveChunks(sourceId: String, chunks: List<StoredChunk>) {
        if (chunks.isEmpty()) return
        transaction {
            requireSourceStatus(sourceId, SourceStatus.INDEXING)
            connection.prepareStatement(
                """INSERT INTO chunks(
                    chunk_id,source_id,strategy,section,page_start,page_end,line_start,line_end,
                    token_units,text,embedding,embedding_model,embedding_dimension
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""".trimIndent(),
            ).use { statement ->
                chunks.forEach { chunk ->
                    val draft = chunk.draft
                    require(draft.sourceId == sourceId) { "A chunk batch cannot contain another source." }
                    require(draft.chunkId.isNotBlank() && draft.section.isNotBlank() && draft.text.isNotBlank()) {
                        "A stored chunk must have an ID, section and text."
                    }
                    require(draft.tokenUnits == TOKEN_UNIT.findAll(draft.text).count()) {
                        "Stored chunk token count must match whitespace units."
                    }
                    require(chunk.embeddingModel.isNotBlank() && chunk.embedding.isNotEmpty() && chunk.embedding.all(Float::isFinite)) {
                        "A chunk must have a finite embedding and model name."
                    }
                    statement.setString(1, draft.chunkId)
                    statement.setString(2, sourceId)
                    statement.setString(3, draft.strategy.name)
                    statement.setString(4, draft.section)
                    statement.setNullableInt(5, draft.location.pageStart)
                    statement.setNullableInt(6, draft.location.pageEnd)
                    statement.setNullableInt(7, draft.location.lineStart)
                    statement.setNullableInt(8, draft.location.lineEnd)
                    statement.setInt(9, draft.tokenUnits)
                    statement.setString(10, draft.text)
                    statement.setBytes(11, encodeEmbedding(chunk.embedding))
                    statement.setString(12, chunk.embeddingModel)
                    statement.setInt(13, chunk.embedding.size)
                    statement.addBatch()
                }
                statement.executeBatch()
            }
        }
    }

    @Synchronized
    override fun search(
        strategy: ChunkStrategy,
        queryEmbedding: List<Float>,
        embeddingModel: String,
        limit: Int,
    ): List<ScoredChunk> {
        require(queryEmbedding.isNotEmpty() && queryEmbedding.all(Float::isFinite)) { "Query embedding is invalid." }
        require(embeddingModel.isNotBlank()) { "Embedding model is required." }
        require(limit in 1..MAX_SEARCH_RESULTS) { "Search limit must be between 1 and $MAX_SEARCH_RESULTS." }
        val query = FloatArray(queryEmbedding.size) { queryEmbedding[it] }
        val queryNorm = norm(query)
        if (queryNorm == 0.0) return emptyList()
        val comparator = compareBy<ScoredChunk> { it.cosineScore }.thenByDescending { it.chunk.draft.chunkId }
        val best = PriorityQueue(comparator)
        connection.prepareStatement(
            """SELECT c.chunk_id,c.strategy,c.section,c.page_start,c.page_end,c.line_start,c.line_end,
                c.token_units,c.text,c.embedding,c.embedding_dimension,c.embedding_model,s.source_id,s.name
                FROM chunks c JOIN sources s ON s.source_id=c.source_id
                WHERE c.strategy=? AND c.embedding_model=? AND c.embedding_dimension=? AND s.status='READY'""".trimIndent(),
        ).use { statement ->
            statement.setString(1, strategy.name)
            statement.setString(2, embeddingModel)
            statement.setInt(3, query.size)
            statement.executeQuery().use { rows ->
                while (rows.next()) {
                    val vectorBytes = rows.getBytes(10)
                    if (vectorBytes.size != rows.getInt(11) * Float.SIZE_BYTES) {
                        throw SQLException("Stored embedding has an invalid byte length.")
                    }
                    val vector = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN)
                    var dot = 0.0
                    var vectorNormSquared = 0.0
                    for (index in query.indices) {
                        val value = vector.float.toDouble()
                        dot += value * query[index]
                        vectorNormSquared += value * value
                    }
                    if (vectorNormSquared == 0.0) continue
                    val score = dot / (sqrt(vectorNormSquared) * queryNorm)
                    if (best.size == limit && score < best.peek().cosineScore) continue
                    val scored = rows.toScoredChunk(score, vectorBytes)
                    if (best.size < limit) best.add(scored)
                    else if (comparator.compare(scored, best.peek()) > 0) {
                        best.poll()
                        best.add(scored)
                    }
                }
            }
        }
        return best.toList().sortedWith(compareByDescending<ScoredChunk> { it.cosineScore }.thenBy { it.chunk.draft.chunkId })
    }

    @Synchronized
    override fun chunkCount(sourceId: String, strategy: ChunkStrategy): Int = connection.prepareStatement(
        "SELECT COUNT(*) FROM chunks WHERE source_id=? AND strategy=?",
    ).use { statement ->
        statement.setString(1, sourceId)
        statement.setString(2, strategy.name)
        statement.executeQuery().use { rows -> rows.next(); rows.getInt(1) }
    }

    @Synchronized
    override fun deleteSource(sourceId: String) {
        connection.prepareStatement("DELETE FROM sources WHERE source_id=?").use { statement ->
            statement.setString(1, sourceId)
            if (statement.executeUpdate() != 1) throw IllegalArgumentException("Источник не найден.")
        }
    }

    @Synchronized
    override fun recoverInterruptedIndexes() {
        transaction {
            connection.prepareStatement("DELETE FROM chunks WHERE source_id IN (SELECT source_id FROM sources WHERE status='INDEXING')").use { it.executeUpdate() }
            connection.prepareStatement(
                "UPDATE sources SET status='FAILED',error=?,unsearchable_pages_json='[]' WHERE status='INDEXING'",
            ).use { statement ->
                statement.setString(1, "Индексация была прервана при остановке приложения.")
                statement.executeUpdate()
            }
        }
    }

    private fun requireSourceStatus(sourceId: String, expected: SourceStatus) {
        val status = connection.prepareStatement("SELECT status FROM sources WHERE source_id=?").use { statement ->
            statement.setString(1, sourceId)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getString(1) else null }
        }
        require(status == expected.name) { "Источник должен находиться в состоянии $expected." }
    }

    private inline fun <T> transaction(block: () -> T): T {
        val previousAutoCommit = connection.autoCommit
        connection.autoCommit = false
        try {
            val result = block()
            connection.commit()
            return result
        } catch (error: Exception) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = previousAutoCommit
        }
    }

    private fun encodeEmbedding(values: List<Float>): ByteArray {
        val buffer = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putFloat)
        return buffer.array()
    }

    private fun ResultSet.toSourceRecord(): SourceRecord = SourceRecord(
        sourceId = getString(1),
        name = getString(2),
        type = SourceType.valueOf(getString(3)),
        sizeBytes = getLong(4),
        status = SourceStatus.valueOf(getString(5)),
        error = getString(6),
        createdAt = getString(7),
        unsearchablePages = Json.decodeFromString<List<Int>>(getString(8)),
    )

    private fun ResultSet.toStoredSource(): StoredSource = StoredSource(toSourceRecord(), getString(9))

    private fun ResultSet.toScoredChunk(score: Double, vectorBytes: ByteArray): ScoredChunk {
        val location = SourceLocation(
            pageStart = getNullableInt(4),
            pageEnd = getNullableInt(5),
            lineStart = getNullableInt(6),
            lineEnd = getNullableInt(7),
        )
        val buffer = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN)
        val embedding = List(vectorBytes.size / Float.SIZE_BYTES) { buffer.float }
        val draft = ChunkDraft(
            strategy = ChunkStrategy.valueOf(getString(2)),
            chunkId = getString(1),
            sourceId = getString(13),
            sourceName = getString(14),
            section = getString(3),
            location = location,
            tokenUnits = getInt(8),
            text = getString(9),
        )
        return ScoredChunk(StoredChunk(draft, embedding, getString(12)), score)
    }

    private fun ResultSet.getNullableInt(column: Int): Int? {
        val value = getInt(column)
        return if (wasNull()) null else value
    }

    private fun java.sql.PreparedStatement.setNullableInt(index: Int, value: Int?) {
        if (value == null) setNull(index, Types.INTEGER) else setInt(index, value)
    }

    private fun norm(vector: FloatArray): Double = sqrt(vector.sumOf { value -> value.toDouble() * value })

    private fun isUuid(value: String): Boolean = runCatching { UUID.fromString(value).toString() == value }.getOrDefault(false)

    @Synchronized
    override fun close() = connection.close()

    companion object {
        const val MAX_SEARCH_RESULTS = 100
        private const val MAX_ERROR_LENGTH = 240
        private val TOKEN_UNIT = Regex("\\S+")
        private val INITIAL_SOURCE_STATUSES = setOf(SourceStatus.PENDING, SourceStatus.FAILED, SourceStatus.UNSUPPORTED)

        fun defaultDatabasePath(): Path = LocalRagPaths.current().database
    }
}
