package dev.localrag.chat

import dev.localrag.domain.SourceCitation
import dev.localrag.source.PrivateLocalStorage
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.Path
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant
import java.util.UUID

@Serializable
data class ChatSession(
    val id: String,
    val title: String,
    val createdAt: String,
    val updatedAt: String,
    val messageCount: Int,
)

@Serializable
enum class ChatMessageRole {
    USER,
    ASSISTANT,
}

@Serializable
data class ChatTaskTerm(val term: String, val definition: String)

@Serializable
data class ChatTaskState(
    val goal: String? = null,
    val clarifications: List<String> = emptyList(),
    val constraints: List<String> = emptyList(),
    val terms: List<ChatTaskTerm> = emptyList(),
)

@Serializable
enum class ChatMemoryScope {
    SESSION,
    SHARED,
}

@Serializable
data class ChatMemoryReference(
    val scope: ChatMemoryScope,
    val id: String,
    val text: String,
)

@Serializable
data class ChatMemoryFact(
    val scope: ChatMemoryScope,
    val id: String,
    val sessionId: String?,
    val text: String,
    val embedding: List<Float>,
    val embeddingModel: String,
    val createdAt: String,
)

@kotlinx.serialization.Serializable
data class ChatMemoryFactSummary(
    val scope: ChatMemoryScope,
    val id: String,
    val sessionId: String?,
    val text: String,
    val createdAt: String,
)

data class ChatMemoryFactDraft(val content: String, val embedding: List<Float>, val embeddingModel: String)

@Serializable
data class ChatTaskStateDelta(
    val goal: String? = null,
    val clarificationsToAdd: List<String> = emptyList(),
    val constraintsToAdd: List<String> = emptyList(),
    val termsToAdd: List<ChatTaskTerm> = emptyList(),
)

data class ChatTurnRequest(
    val question: String,
    val previousMessages: List<ChatMessage>,
    val taskState: ChatTaskState,
    val sessionFacts: List<ChatMemoryFact>,
    val sharedFacts: List<ChatMemoryFact>,
    val documentChunks: List<String>,
)

data class ChatTurnCompletion(val answer: String, val taskStateDelta: ChatTaskStateDelta)

interface ChatTurnGenerator {
    fun completeTurn(selection: dev.localrag.domain.ModelSelection, request: ChatTurnRequest): ChatTurnCompletion
}

@Serializable
data class ChatMessage(
    val id: String,
    val turnId: String,
    val role: ChatMessageRole,
    val content: String,
    val documentCitations: List<SourceCitation> = emptyList(),
    val memoryReferences: List<ChatMemoryReference> = emptyList(),
    val createdAt: String,
)

@Serializable
data class ChatSessionDetail(
    val session: ChatSession,
    val messages: List<ChatMessage>,
    val taskState: ChatTaskState,
)

class ChatStore(databasePath: Path) : AutoCloseable {
    private val json = Json { encodeDefaults = true }
    private val connection: Connection

    init {
        databasePath.toAbsolutePath().parent?.let(PrivateLocalStorage::prepareDirectory)
        PrivateLocalStorage.prepareFile(databasePath)
        connection = DriverManager.getConnection("jdbc:sqlite:${databasePath.toAbsolutePath().normalize()}")
        try {
            connection.createStatement().use { statement ->
                statement.execute("PRAGMA foreign_keys=ON")
                statement.execute("PRAGMA busy_timeout=5000")
                statement.execute("PRAGMA journal_mode=WAL")
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        turn_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('USER','ASSISTANT')),
                        content TEXT NOT NULL,
                        document_citations_json TEXT NOT NULL DEFAULT '[]',
                        memory_references_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        UNIQUE(session_id,turn_id,role)
                    )""".trimIndent(),
                )
                statement.execute("CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages(session_id,created_at,message_id)")
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS chat_task_states (
                        session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        goal TEXT,
                        clarifications_json TEXT NOT NULL DEFAULT '[]',
                        constraints_json TEXT NOT NULL DEFAULT '[]',
                        terms_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS chat_shared_facts (
                        memory_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS chat_session_facts (
                        memory_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(session_id,content)
                    )""".trimIndent(),
                )
                val version = statement.executeQuery("PRAGMA user_version").use { rows ->
                    if (rows.next()) rows.getInt(1) else 0
                }
                if (version < CHAT_SCHEMA_VERSION) statement.execute("PRAGMA user_version=$CHAT_SCHEMA_VERSION")
            }
        } catch (error: Throwable) {
            connection.close()
            throw error
        }
    }

    @Synchronized
    fun createSession(): ChatSessionDetail = transaction {
        val now = Instant.now().toString()
        val sessionId = UUID.randomUUID().toString()
        connection.prepareStatement(
            "INSERT INTO chat_sessions(session_id,title,created_at,updated_at) VALUES (?,?,?,?)",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, NEW_SESSION_TITLE)
            statement.setString(3, now)
            statement.setString(4, now)
            statement.executeUpdate()
        }
        connection.prepareStatement(
            "INSERT INTO chat_task_states(session_id,goal,clarifications_json,constraints_json,terms_json,updated_at) VALUES (?,NULL,'[]','[]','[]',?)",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, now)
            statement.executeUpdate()
        }
        requireNotNull(sessionDetail(sessionId))
    }

    @Synchronized
    fun sessionExists(sessionId: String): Boolean {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement("SELECT 1 FROM chat_sessions WHERE session_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { it.next() }
        }
    }

    @Synchronized
    fun sessions(): List<ChatSession> = connection.prepareStatement(
        """SELECT s.session_id,s.title,s.created_at,s.updated_at,COUNT(m.message_id)
            FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id=s.session_id
            GROUP BY s.session_id ORDER BY s.updated_at DESC,s.session_id""".trimIndent(),
    ).use { statement ->
        statement.executeQuery().use { rows ->
            buildList {
                while (rows.next()) add(
                    ChatSession(
                        id = rows.getString(1),
                        title = rows.getString(2),
                        createdAt = rows.getString(3),
                        updatedAt = rows.getString(4),
                        messageCount = rows.getInt(5),
                    ),
                )
            }
        }
    }

    @Synchronized
    fun sessionDetail(sessionId: String): ChatSessionDetail? {
        requireUuid(sessionId, "session ID")
        val session = connection.prepareStatement(
            """SELECT s.session_id,s.title,s.created_at,s.updated_at,COUNT(m.message_id)
                FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id=s.session_id
                WHERE s.session_id=? GROUP BY s.session_id""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatSession(
                    id = rows.getString(1),
                    title = rows.getString(2),
                    createdAt = rows.getString(3),
                    updatedAt = rows.getString(4),
                    messageCount = rows.getInt(5),
                )
            }
        } ?: return null

        val messages = connection.prepareStatement(
            """SELECT message_id,turn_id,role,content,document_citations_json,memory_references_json,created_at
                FROM chat_messages WHERE session_id=? ORDER BY created_at,message_id""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) add(
                        ChatMessage(
                            id = rows.getString(1),
                            turnId = rows.getString(2),
                            role = ChatMessageRole.valueOf(rows.getString(3)),
                            content = rows.getString(4),
                            documentCitations = json.decodeFromString(rows.getString(5)),
                            memoryReferences = json.decodeFromString(rows.getString(6)),
                            createdAt = rows.getString(7),
                        ),
                    )
                }
            }
        }
        val taskState = connection.prepareStatement(
            "SELECT goal,clarifications_json,constraints_json,terms_json FROM chat_task_states WHERE session_id=?",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) ChatTaskState() else ChatTaskState(
                    goal = rows.getString(1),
                    clarifications = json.decodeFromString(rows.getString(2)),
                    constraints = json.decodeFromString(rows.getString(3)),
                    terms = json.decodeFromString(rows.getString(4)),
                )
            }
        }
        return ChatSessionDetail(session, messages, taskState)
    }

    @Synchronized
    fun saveUserMessage(sessionId: String, turnId: String, content: String): ChatMessage = transaction {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        require(content.isNotBlank() && content.length <= MAX_MESSAGE_LENGTH) {
            "Сообщение должно содержать от 1 до $MAX_MESSAGE_LENGTH символов."
        }
        val title = connection.prepareStatement("SELECT title FROM chat_sessions WHERE session_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) throw ChatSessionNotFoundException()
                rows.getString(1)
            }
        }
        val existing = connection.prepareStatement(
            "SELECT message_id,content,created_at FROM chat_messages WHERE session_id=? AND turn_id=? AND role='USER'",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, turnId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else Triple(rows.getString(1), rows.getString(2), rows.getString(3))
            }
        }
        if (existing != null) {
            require(existing.second == content) { "Повторный turn ID не может менять текст сообщения." }
            return@transaction ChatMessage(existing.first, turnId, ChatMessageRole.USER, content, createdAt = existing.third)
        }

        val messageId = UUID.randomUUID().toString()
        val now = Instant.now().toString()
        connection.prepareStatement(
            "INSERT INTO chat_messages(message_id,session_id,turn_id,role,content,created_at) VALUES (?,?,?,'USER',?,?)",
        ).use { statement ->
            statement.setString(1, messageId)
            statement.setString(2, sessionId)
            statement.setString(3, turnId)
            statement.setString(4, content)
            statement.setString(5, now)
            statement.executeUpdate()
        }
        val nextTitle = if (title == NEW_SESSION_TITLE) content.trim().replace(WHITESPACE, " ").take(MAX_TITLE_LENGTH) else title
        connection.prepareStatement("UPDATE chat_sessions SET title=?,updated_at=? WHERE session_id=?").use { statement ->
            statement.setString(1, nextTitle)
            statement.setString(2, now)
            statement.setString(3, sessionId)
            statement.executeUpdate()
        }
        ChatMessage(messageId, turnId, ChatMessageRole.USER, content, createdAt = now)
    }

    @Synchronized
    fun assistantMessage(sessionId: String, turnId: String): ChatMessage? {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        return connection.prepareStatement(
            """SELECT message_id,content,document_citations_json,memory_references_json,created_at
                FROM chat_messages WHERE session_id=? AND turn_id=? AND role='ASSISTANT'""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, turnId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatMessage(
                    id = rows.getString(1),
                    turnId = turnId,
                    role = ChatMessageRole.ASSISTANT,
                    content = rows.getString(2),
                    documentCitations = json.decodeFromString(rows.getString(3)),
                    memoryReferences = json.decodeFromString(rows.getString(4)),
                    createdAt = rows.getString(5),
                )
            }
        }
    }

    @Synchronized
    fun recentMessages(sessionId: String, limit: Int): List<ChatMessage> {
        requireUuid(sessionId, "session ID")
        require(limit in 0..MAX_HISTORY_MESSAGES + 1)
        return connection.prepareStatement(
            """SELECT message_id,turn_id,role,content,document_citations_json,memory_references_json,created_at
                FROM chat_messages WHERE session_id=? ORDER BY created_at DESC,message_id DESC LIMIT ?""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setInt(2, limit)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) add(
                        ChatMessage(
                            id = rows.getString(1),
                            turnId = rows.getString(2),
                            role = ChatMessageRole.valueOf(rows.getString(3)),
                            content = rows.getString(4),
                            documentCitations = json.decodeFromString(rows.getString(5)),
                            memoryReferences = json.decodeFromString(rows.getString(6)),
                            createdAt = rows.getString(7),
                        ),
                    )
                }.asReversed()
            }
        }
    }

    @Synchronized
    fun taskState(sessionId: String): ChatTaskState {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement(
            "SELECT goal,clarifications_json,constraints_json,terms_json FROM chat_task_states WHERE session_id=?",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) throw ChatSessionNotFoundException()
                ChatTaskState(
                    goal = rows.getString(1),
                    clarifications = json.decodeFromString(rows.getString(2)),
                    constraints = json.decodeFromString(rows.getString(3)),
                    terms = json.decodeFromString(rows.getString(4)),
                )
            }
        }
    }

    @Synchronized
    fun completeTurn(
        sessionId: String,
        turnId: String,
        answer: String,
        citations: List<SourceCitation>,
        memoryReferences: List<ChatMemoryReference>,
        taskState: ChatTaskState,
        sessionFacts: List<ChatMemoryFactDraft> = emptyList(),
    ): ChatMessage = transaction {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        require(answer.isNotBlank() && answer.length <= MAX_MESSAGE_LENGTH)
        val existing = connection.prepareStatement(
            """SELECT message_id,content,document_citations_json,memory_references_json,created_at
                FROM chat_messages WHERE session_id=? AND turn_id=? AND role='ASSISTANT'""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, turnId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatMessage(
                    id = rows.getString(1),
                    turnId = turnId,
                    role = ChatMessageRole.ASSISTANT,
                    content = rows.getString(2),
                    documentCitations = json.decodeFromString(rows.getString(3)),
                    memoryReferences = json.decodeFromString(rows.getString(4)),
                    createdAt = rows.getString(5),
                )
            }
        }
        if (existing != null) return@transaction existing
        connection.prepareStatement(
            "SELECT 1 FROM chat_messages WHERE session_id=? AND turn_id=? AND role='USER'",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, turnId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) throw IllegalArgumentException("Пользовательское сообщение для этого turn ID не найдено.")
            }
        }
        val now = Instant.now().toString()
        val messageId = UUID.randomUUID().toString()
        connection.prepareStatement(
            """INSERT INTO chat_messages(
                message_id,session_id,turn_id,role,content,document_citations_json,memory_references_json,created_at
            ) VALUES (?,?,?,'ASSISTANT',?,?,?,?)""".trimIndent(),
        ).use { statement ->
            statement.setString(1, messageId)
            statement.setString(2, sessionId)
            statement.setString(3, turnId)
            statement.setString(4, answer)
            statement.setString(5, json.encodeToString(citations))
            statement.setString(6, json.encodeToString(memoryReferences))
            statement.setString(7, now)
            statement.executeUpdate()
        }
        connection.prepareStatement(
            """UPDATE chat_task_states SET goal=?,clarifications_json=?,constraints_json=?,terms_json=?,updated_at=?
                WHERE session_id=?""".trimIndent(),
        ).use { statement ->
            statement.setString(1, taskState.goal)
            statement.setString(2, json.encodeToString(taskState.clarifications))
            statement.setString(3, json.encodeToString(taskState.constraints))
            statement.setString(4, json.encodeToString(taskState.terms))
            statement.setString(5, now)
            statement.setString(6, sessionId)
            require(statement.executeUpdate() == 1) { "Состояние задачи для сессии не найдено." }
        }
        connection.prepareStatement("DELETE FROM chat_session_facts WHERE session_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.executeUpdate()
        }
        sessionFacts.forEach { fact ->
            saveMemoryFact(ChatMemoryScope.SESSION, sessionId, fact.content, fact.embedding, fact.embeddingModel)
        }
        connection.prepareStatement("UPDATE chat_sessions SET updated_at=? WHERE session_id=?").use { statement ->
            statement.setString(1, now)
            statement.setString(2, sessionId)
            statement.executeUpdate()
        }
        ChatMessage(messageId, turnId, ChatMessageRole.ASSISTANT, answer, citations, memoryReferences, now)
    }

    @Synchronized
    fun saveSessionFact(sessionId: String, content: String, embedding: List<Float>, embeddingModel: String): ChatMemoryFact =
        saveMemoryFact(ChatMemoryScope.SESSION, sessionId, content, embedding, embeddingModel)

    @Synchronized
    fun saveSharedFact(content: String, embedding: List<Float>, embeddingModel: String): ChatMemoryFact =
        saveMemoryFact(ChatMemoryScope.SHARED, null, content, embedding, embeddingModel)

    @Synchronized
    fun updateSharedFact(memoryId: String, content: String, embedding: List<Float>, embeddingModel: String): Boolean = transaction {
        requireUuid(memoryId, "memory ID")
        require(content.isNotBlank() && content.length <= MAX_SHARED_FACT_LENGTH)
        require(embedding.isNotEmpty() && embedding.all(Float::isFinite))
        require(embeddingModel.isNotBlank() && embeddingModel.length <= MAX_EMBEDDING_MODEL_LENGTH)
        connection.prepareStatement(
            "UPDATE chat_shared_facts SET content=?,embedding=?,embedding_model=?,updated_at=? WHERE memory_id=?",
        ).use { statement ->
            statement.setString(1, content.trim())
            statement.setBytes(2, encodeEmbedding(embedding))
            statement.setString(3, embeddingModel)
            statement.setString(4, Instant.now().toString())
            statement.setString(5, memoryId)
            statement.executeUpdate() == 1
        }
    }

    @Synchronized
    fun memoryFacts(sessionId: String): List<ChatMemoryFact> {
        requireUuid(sessionId, "session ID")
        val sessionFacts = queryMemoryFacts(
            """SELECT memory_id,session_id,content,embedding,embedding_model,created_at
                FROM chat_session_facts WHERE session_id=?""".trimIndent(),
            sessionId,
            ChatMemoryScope.SESSION,
        )
        return sessionFacts + queryMemoryFacts(
            """SELECT memory_id,NULL,content,embedding,embedding_model,created_at
                FROM chat_shared_facts ORDER BY created_at,memory_id""".trimIndent(),
            null,
            ChatMemoryScope.SHARED,
        )
    }

    @Synchronized
    fun sharedFacts(): List<ChatMemoryFact> = queryMemoryFacts(
        """SELECT memory_id,NULL,content,embedding,embedding_model,created_at
            FROM chat_shared_facts ORDER BY created_at,memory_id""".trimIndent(),
        null,
        ChatMemoryScope.SHARED,
    )

    @Synchronized
    fun replaceTaskState(sessionId: String, taskState: ChatTaskState, sessionFacts: List<ChatMemoryFactDraft>) = transaction {
        requireUuid(sessionId, "session ID")
        val now = Instant.now().toString()
        val updated = connection.prepareStatement(
            """UPDATE chat_task_states SET goal=?,clarifications_json=?,constraints_json=?,terms_json=?,updated_at=?
                WHERE session_id=?""".trimIndent(),
        ).use { statement ->
            statement.setString(1, taskState.goal)
            statement.setString(2, json.encodeToString(taskState.clarifications))
            statement.setString(3, json.encodeToString(taskState.constraints))
            statement.setString(4, json.encodeToString(taskState.terms))
            statement.setString(5, now)
            statement.setString(6, sessionId)
            statement.executeUpdate()
        }
        if (updated != 1) throw ChatSessionNotFoundException()
        connection.prepareStatement("DELETE FROM chat_session_facts WHERE session_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.executeUpdate()
        }
        sessionFacts.forEach { fact ->
            saveMemoryFact(ChatMemoryScope.SESSION, sessionId, fact.content, fact.embedding, fact.embeddingModel)
        }
        connection.prepareStatement("UPDATE chat_sessions SET updated_at=? WHERE session_id=?").use { statement ->
            statement.setString(1, now)
            statement.setString(2, sessionId)
            statement.executeUpdate()
        }
    }

    @Synchronized
    fun deleteSharedFact(memoryId: String): Boolean {
        requireUuid(memoryId, "memory ID")
        return connection.prepareStatement("DELETE FROM chat_shared_facts WHERE memory_id=?").use { statement ->
            statement.setString(1, memoryId)
            statement.executeUpdate() == 1
        }
    }

    @Synchronized
    fun deleteSession(sessionId: String): Boolean {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement("DELETE FROM chat_sessions WHERE session_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.executeUpdate() == 1
        }
    }

    private fun saveMemoryFact(
        scope: ChatMemoryScope,
        sessionId: String?,
        content: String,
        embedding: List<Float>,
        embeddingModel: String,
    ): ChatMemoryFact = transaction {
        require(content.isNotBlank() && content.length <= MAX_MEMORY_FACT_LENGTH)
        require(embedding.isNotEmpty() && embedding.all(Float::isFinite))
        require(embeddingModel.isNotBlank() && embeddingModel.length <= MAX_EMBEDDING_MODEL_LENGTH)
        if (scope == ChatMemoryScope.SESSION) requireUuid(requireNotNull(sessionId), "session ID")
        else require(sessionId == null)
        val table = if (scope == ChatMemoryScope.SESSION) "chat_session_facts" else "chat_shared_facts"
        val existing = if (scope == ChatMemoryScope.SESSION) {
            connection.prepareStatement(
                "SELECT memory_id,created_at FROM $table WHERE session_id=? AND lower(trim(content))=lower(trim(?))",
            ).use { statement ->
                statement.setString(1, sessionId)
                statement.setString(2, content)
                statement.executeQuery().use { rows -> if (rows.next()) rows.getString(1) to rows.getString(2) else null }
            }
        } else {
            connection.prepareStatement(
                "SELECT memory_id,created_at FROM $table WHERE lower(trim(content))=lower(trim(?))",
            ).use { statement ->
                statement.setString(1, content)
                statement.executeQuery().use { rows -> if (rows.next()) rows.getString(1) to rows.getString(2) else null }
            }
        }
        val id = existing?.first ?: UUID.randomUUID().toString()
        val createdAt = existing?.second ?: Instant.now().toString()
        val now = Instant.now().toString()
        if (existing != null) {
            val sql = if (scope == ChatMemoryScope.SESSION) {
                "UPDATE $table SET content=?,embedding=?,embedding_model=? WHERE memory_id=?"
            } else {
                "UPDATE $table SET content=?,embedding=?,embedding_model=?,updated_at=? WHERE memory_id=?"
            }
            connection.prepareStatement(sql).use { statement ->
                statement.setString(1, content.trim())
                statement.setBytes(2, encodeEmbedding(embedding))
                statement.setString(3, embeddingModel)
                if (scope == ChatMemoryScope.SESSION) statement.setString(4, id)
                else {
                    statement.setString(4, now)
                    statement.setString(5, id)
                }
                statement.executeUpdate()
            }
        } else {
            val sql = if (scope == ChatMemoryScope.SESSION) {
                "INSERT INTO $table(memory_id,session_id,content,embedding,embedding_model,created_at) VALUES (?,?,?,?,?,?)"
            } else {
                "INSERT INTO $table(memory_id,content,embedding,embedding_model,created_at,updated_at) VALUES (?,?,?,?,?,?)"
            }
            connection.prepareStatement(sql).use { statement ->
                statement.setString(1, id)
                var index = 2
                if (scope == ChatMemoryScope.SESSION) statement.setString(index++, sessionId)
                statement.setString(index++, content.trim())
                statement.setBytes(index++, encodeEmbedding(embedding))
                statement.setString(index++, embeddingModel)
                statement.setString(index++, now)
                if (scope == ChatMemoryScope.SHARED) statement.setString(index, now)
                statement.executeUpdate()
            }
        }
        ChatMemoryFact(scope, id, sessionId, content.trim(), embedding, embeddingModel, createdAt)
    }

    private fun queryMemoryFacts(sql: String, sessionId: String?, scope: ChatMemoryScope): List<ChatMemoryFact> =
        connection.prepareStatement(sql).use { statement ->
            if (sessionId != null) statement.setString(1, sessionId)
            statement.executeQuery().use { rows ->
                buildList {
                    while (rows.next()) add(
                        ChatMemoryFact(
                            scope = scope,
                            id = rows.getString(1),
                            sessionId = rows.getString(2),
                            text = rows.getString(3),
                            embedding = decodeEmbedding(rows.getBytes(4)),
                            embeddingModel = rows.getString(5),
                            createdAt = rows.getString(6),
                        ),
                    )
                }
            }
        }

    private fun encodeEmbedding(vector: List<Float>): ByteArray =
        ByteBuffer.allocate(vector.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).apply {
            vector.forEach(::putFloat)
        }.array()

    private fun decodeEmbedding(bytes: ByteArray): List<Float> {
        require(bytes.isNotEmpty() && bytes.size % Float.SIZE_BYTES == 0) { "Повреждён вектор локальной памяти." }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return buildList(bytes.size / Float.SIZE_BYTES) {
            while (buffer.hasRemaining()) add(buffer.float)
        }
    }

    override fun close() = connection.close()

    private inline fun <T> transaction(block: () -> T): T {
        val previousAutoCommit = connection.autoCommit
        if (!previousAutoCommit) return block()
        connection.autoCommit = false
        try {
            val result = block()
            connection.commit()
            return result
        } catch (error: Throwable) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = previousAutoCommit
        }
    }

    private fun requireUuid(value: String, label: String) {
        require(runCatching { UUID.fromString(value).toString() == value }.getOrDefault(false)) {
            "Некорректный $label."
        }
    }

    class ChatSessionNotFoundException : RuntimeException("Сессия чата не найдена.")

    companion object {
        const val CHAT_SCHEMA_VERSION = 13
        const val MAX_MESSAGE_LENGTH = 4_000
        const val MAX_HISTORY_MESSAGES = 8
        const val MAX_MEMORY_FACT_LENGTH = 600
        const val MAX_SHARED_FACT_LENGTH = MAX_MEMORY_FACT_LENGTH
        const val MAX_EMBEDDING_MODEL_LENGTH = 120
        const val MAX_TITLE_LENGTH = 60
        const val NEW_SESSION_TITLE = "Новая сессия"
        private val WHITESPACE = Regex("\\s+")
    }
}
