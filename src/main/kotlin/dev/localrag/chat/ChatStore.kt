package dev.localrag.chat

import dev.localrag.domain.ModelSelection
import dev.localrag.domain.SourceCitation
import dev.localrag.domain.ChunkStrategy
import dev.localrag.source.PrivateLocalStorage
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Path
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
    val modelSelection: ModelSelection? = null,
    val modelSelectionRequired: Boolean = false,
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

data class ChatMemoryCandidate(val refId: String, val scope: ChatMemoryScope, val text: String)

data class ChatDocumentCandidate(val refId: String, val text: String)

data class ChatMemoryUpdate(val scope: ChatMemoryScope, val content: String)

data class ChatTurnRequest(
    val question: String,
    val previousMessages: List<ChatMessage>,
    val taskState: ChatTaskState,
    val sessionFacts: List<ChatMemoryCandidate>,
    val sharedFacts: List<ChatMemoryCandidate>,
    val documentCandidates: List<ChatDocumentCandidate>,
    val generateTitle: Boolean,
)

data class ChatTurnCompletion(
    val answer: String,
    val taskStateDelta: ChatTaskStateDelta,
    val memoryUpdates: List<ChatMemoryUpdate> = emptyList(),
    val documentCitationRefs: List<String> = emptyList(),
    val memoryReferenceRefs: List<String> = emptyList(),
    val sessionTitle: String? = null,
)

interface ChatTurnGenerator {
    fun completeTurn(selection: ModelSelection, request: ChatTurnRequest): ChatTurnCompletion
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
    val modelSelection: ModelSelection? = null,
    val elapsedMs: Long? = null,
)

@Serializable
data class ChatTurnSnapshot(
    val turnId: String,
    val state: String,
    val error: String? = null,
)

@Serializable
data class ChatTurnJob(
    val turnId: String,
    val sessionId: String,
    val question: String,
    val strategy: ChunkStrategy,
    val topK: Int?,
    val selection: ModelSelection,
    val state: String,
    val error: String?,
    val createdAt: String,
    val updatedAt: String,
)

@Serializable
data class ChatSessionDetail(
    val session: ChatSession,
    val messages: List<ChatMessage>,
    val taskState: ChatTaskState,
    val activeTurn: ChatTurnSnapshot? = null,
)

class ChatTurnCancelledException : RuntimeException("Ход отменён.")

class ChatActiveTurnException : RuntimeException("В этой сессии уже есть незавершённый ход.")

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
            }
            connection.autoCommit = false
            try {
                initializeSchema()
                connection.commit()
            } catch (error: Throwable) {
                connection.rollback()
                throw error
            } finally {
                connection.autoCommit = true
            }
        } catch (error: Throwable) {
            connection.close()
            throw error
        }
    }

    private fun initializeSchema() {
        connection.createStatement().use { statement ->
            statement.execute(
                """CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    title_generated INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provider_id TEXT,
                    model_id TEXT,
                    selection_required INTEGER NOT NULL DEFAULT 0,
                    workspace_id TEXT NOT NULL DEFAULT 'legacy-owner'
                )""".trimIndent(),
            )
            val hasTitleGenerated = statement.executeQuery("PRAGMA table_info(chat_sessions)").use { rows ->
                var found = false
                while (rows.next()) if (rows.getString("name") == "title_generated") found = true
                found
            }
            if (!hasTitleGenerated) {
                statement.execute("ALTER TABLE chat_sessions ADD COLUMN title_generated INTEGER NOT NULL DEFAULT 1")
            }
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
                    provider_id TEXT,
                    model_id TEXT,
                    elapsed_ms INTEGER,
                    UNIQUE(session_id,turn_id,role)
                )""".trimIndent(),
            )
            ensureColumn("chat_sessions", "provider_id", "TEXT")
            ensureColumn("chat_sessions", "model_id", "TEXT")
            ensureColumn("chat_sessions", "selection_required", "INTEGER NOT NULL DEFAULT 0")
            ensureColumn("chat_sessions", "workspace_id", "TEXT NOT NULL DEFAULT 'legacy-owner'")
            ensureColumn("chat_messages", "provider_id", "TEXT")
            ensureColumn("chat_messages", "model_id", "TEXT")
            ensureColumn("chat_messages", "elapsed_ms", "INTEGER")
            statement.execute("CREATE INDEX IF NOT EXISTS chat_sessions_workspace_idx ON chat_sessions(workspace_id,updated_at)")
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
                    updated_at TEXT NOT NULL,
                    workspace_id TEXT NOT NULL DEFAULT 'legacy-owner'
                )""".trimIndent(),
            )
            ensureColumn("chat_shared_facts", "workspace_id", "TEXT NOT NULL DEFAULT 'legacy-owner'")
            statement.execute("CREATE INDEX IF NOT EXISTS chat_shared_facts_workspace_idx ON chat_shared_facts(workspace_id,created_at)")
            statement.execute(
                """CREATE TABLE IF NOT EXISTS chat_turn_jobs (
                    turn_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL,
                    top_k INTEGER,
                    provider_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('queued','running','completed','failed','cancelled','interrupted')),
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id,turn_id)
                )""".trimIndent(),
            )
            statement.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS chat_turn_jobs_one_active_session_idx
                    ON chat_turn_jobs(workspace_id,session_id)
                    WHERE state IN ('queued','running')""".trimIndent(),
            )
            statement.execute("CREATE INDEX IF NOT EXISTS chat_turn_jobs_workspace_idx ON chat_turn_jobs(workspace_id,created_at)")
            statement.execute("UPDATE chat_turn_jobs SET state='interrupted',updated_at=datetime('now') WHERE state IN ('queued','running')")
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
    }

    @Synchronized
    fun createSession(
        modelSelection: ModelSelection? = null,
        modelSelectionRequired: Boolean = false,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatSessionDetail = transaction {
        val now = Instant.now().toString()
        val sessionId = UUID.randomUUID().toString()
        connection.prepareStatement(
            """INSERT INTO chat_sessions(
                session_id,title,title_generated,created_at,updated_at,provider_id,model_id,selection_required,workspace_id
            ) VALUES (?,?,0,?,?,?,?,?,?)""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, NEW_SESSION_TITLE)
            statement.setString(3, now)
            statement.setString(4, now)
            statement.setString(5, modelSelection?.providerId)
            statement.setString(6, modelSelection?.modelId)
            statement.setInt(7, if (modelSelectionRequired) 1 else 0)
            statement.setString(8, workspaceId)
            statement.executeUpdate()
        }
        connection.prepareStatement(
            "INSERT INTO chat_task_states(session_id,goal,clarifications_json,constraints_json,terms_json,updated_at) VALUES (?,NULL,'[]','[]','[]',?)",
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, now)
            statement.executeUpdate()
        }
        requireNotNull(sessionDetail(sessionId, workspaceId))
    }

    @Synchronized
    fun initializeModelSelection(
        sessionId: String,
        modelSelection: ModelSelection,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatSessionDetail = transaction {
        requireUuid(sessionId, "session ID")
        connection.prepareStatement(
            "UPDATE chat_sessions SET provider_id=?,model_id=?,selection_required=0 WHERE session_id=? AND workspace_id=? AND provider_id IS NULL AND model_id IS NULL AND selection_required=0",
        ).use { statement ->
            statement.setString(1, modelSelection.providerId)
            statement.setString(2, modelSelection.modelId)
            statement.setString(3, sessionId)
            statement.setString(4, workspaceId)
            statement.executeUpdate()
        }
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
        requireNotNull(sessionDetail(sessionId, workspaceId))
    }

    @Synchronized
    fun updateModelSelection(
        sessionId: String,
        modelSelection: ModelSelection,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatSessionDetail = transaction {
        requireUuid(sessionId, "session ID")
        connection.prepareStatement(
            "UPDATE chat_sessions SET provider_id=?,model_id=?,selection_required=0 WHERE session_id=? AND workspace_id=?",
        ).use { statement ->
            statement.setString(1, modelSelection.providerId)
            statement.setString(2, modelSelection.modelId)
            statement.setString(3, sessionId)
            statement.setString(4, workspaceId)
            if (statement.executeUpdate() != 1) throw ChatSessionNotFoundException()
        }
        requireNotNull(sessionDetail(sessionId, workspaceId))
    }

    @Synchronized
    fun sessionExists(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): Boolean {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement("SELECT 1 FROM chat_sessions WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, workspaceId)
            statement.executeQuery().use { it.next() }
        }
    }

    @Synchronized
    fun sessionTitleGenerated(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): Boolean {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement("SELECT title_generated FROM chat_sessions WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, workspaceId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) throw ChatSessionNotFoundException()
                rows.getInt(1) != 0
            }
        }
    }

    @Synchronized
    fun sessions(workspaceId: String = DEFAULT_WORKSPACE_ID): List<ChatSession> = connection.prepareStatement(
        """SELECT s.session_id,s.title,s.created_at,s.updated_at,COUNT(m.message_id),s.provider_id,s.model_id,s.selection_required
            FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id=s.session_id
            WHERE s.workspace_id=?
            GROUP BY s.session_id ORDER BY s.updated_at DESC,s.session_id""".trimIndent(),
    ).use { statement ->
        statement.setString(1, workspaceId)
        statement.executeQuery().use { rows ->
            buildList {
                while (rows.next()) add(
                    ChatSession(
                        id = rows.getString(1),
                        title = rows.getString(2),
                        createdAt = rows.getString(3),
                        updatedAt = rows.getString(4),
                        messageCount = rows.getInt(5),
                        modelSelection = readModelSelection(rows.getString(6), rows.getString(7)),
                        modelSelectionRequired = rows.getInt(8) != 0,
                    ),
                )
            }
        }
    }

    @Synchronized
    fun sessionDetail(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): ChatSessionDetail? {
        requireUuid(sessionId, "session ID")
        val session = connection.prepareStatement(
            """SELECT s.session_id,s.title,s.created_at,s.updated_at,COUNT(m.message_id),s.provider_id,s.model_id,s.selection_required
                FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id=s.session_id
                WHERE s.session_id=? AND s.workspace_id=? GROUP BY s.session_id""".trimIndent(),
        ).use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, workspaceId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatSession(
                    id = rows.getString(1),
                    title = rows.getString(2),
                    createdAt = rows.getString(3),
                    updatedAt = rows.getString(4),
                    messageCount = rows.getInt(5),
                    modelSelection = readModelSelection(rows.getString(6), rows.getString(7)),
                    modelSelectionRequired = rows.getInt(8) != 0,
                )
            }
        } ?: return null

        val messages = connection.prepareStatement(
            """SELECT message_id,turn_id,role,content,document_citations_json,memory_references_json,created_at,provider_id,model_id,elapsed_ms
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
                            modelSelection = readModelSelection(rows.getString(8), rows.getString(9)),
                            elapsedMs = rows.getLong(10).let { if (rows.wasNull()) null else it },
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
        val latestTurn = connection.prepareStatement(
            "SELECT turn_id,state,error FROM chat_turn_jobs WHERE workspace_id=? AND session_id=? ORDER BY created_at DESC LIMIT 1",
        ).use { statement ->
            statement.setString(1, workspaceId)
            statement.setString(2, sessionId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatTurnSnapshot(rows.getString(1), rows.getString(2), rows.getString(3))
            }
        }?.takeUnless { it.state == "completed" }
        return ChatSessionDetail(session, messages, taskState, latestTurn)
    }

    @Synchronized
    fun saveUserMessage(
        sessionId: String,
        turnId: String,
        content: String,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatMessage = transaction { saveUserMessageInTransaction(sessionId, turnId, content, workspaceId) }

    private fun saveUserMessageInTransaction(
        sessionId: String,
        turnId: String,
        content: String,
        workspaceId: String,
    ): ChatMessage {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        require(sessionExists(sessionId, workspaceId)) { "Сессия чата не найдена." }
        require(content.isNotBlank() && content.length <= MAX_MESSAGE_LENGTH) {
            "Сообщение должно содержать от 1 до $MAX_MESSAGE_LENGTH символов."
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
            return ChatMessage(existing.first, turnId, ChatMessageRole.USER, content, createdAt = existing.third)
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
        connection.prepareStatement("UPDATE chat_sessions SET updated_at=? WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, now)
            statement.setString(2, sessionId)
            statement.setString(3, workspaceId)
            statement.executeUpdate()
        }
        return ChatMessage(messageId, turnId, ChatMessageRole.USER, content, createdAt = now)
    }

    @Synchronized
    fun enqueueTurn(
        workspaceId: String,
        sessionId: String,
        turnId: String,
        question: String,
        strategy: ChunkStrategy,
        topK: Int?,
        selection: ModelSelection,
    ): ChatTurnJob = transaction {
        requireUuid(turnId, "turn ID")
        require(sessionExists(sessionId, workspaceId)) { "Сессия чата не найдена." }
        val existing = readTurnJob(workspaceId, sessionId, turnId)
        if (existing != null && existing.question != question) {
            throw IllegalArgumentException("Повторный turn ID не может менять текст сообщения.")
        }
        if (existing?.state in setOf("queued", "running", "completed")) return@transaction requireNotNull(existing)
        val active = connection.prepareStatement(
            "SELECT turn_id FROM chat_turn_jobs WHERE workspace_id=? AND session_id=? AND state IN ('queued','running') LIMIT 1",
        ).use { statement ->
            statement.setString(1, workspaceId)
            statement.setString(2, sessionId)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getString(1) else null }
        }
        if (active != null && active != turnId) throw ChatActiveTurnException()
        saveUserMessageInTransaction(sessionId, turnId, question, workspaceId)
        val now = Instant.now().toString()
        if (existing == null) {
            connection.prepareStatement(
                """INSERT INTO chat_turn_jobs(
                    turn_id,workspace_id,session_id,strategy,top_k,provider_id,model_id,state,error,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,'queued',NULL,?,?)""".trimIndent(),
            ).use { statement ->
                statement.setString(1, turnId)
                statement.setString(2, workspaceId)
                statement.setString(3, sessionId)
                statement.setString(4, strategy.name)
                if (topK == null) statement.setNull(5, java.sql.Types.INTEGER) else statement.setInt(5, topK)
                statement.setString(6, selection.providerId)
                statement.setString(7, selection.modelId)
                statement.setString(8, now)
                statement.setString(9, now)
                statement.executeUpdate()
            }
        } else {
            connection.prepareStatement(
                """UPDATE chat_turn_jobs SET strategy=?,top_k=?,provider_id=?,model_id=?,state='queued',error=NULL,updated_at=?
                    WHERE workspace_id=? AND session_id=? AND turn_id=?""".trimIndent(),
            ).use { statement ->
                statement.setString(1, strategy.name)
                if (topK == null) statement.setNull(2, java.sql.Types.INTEGER) else statement.setInt(2, topK)
                statement.setString(3, selection.providerId)
                statement.setString(4, selection.modelId)
                statement.setString(5, now)
                statement.setString(6, workspaceId)
                statement.setString(7, sessionId)
                statement.setString(8, turnId)
                statement.executeUpdate()
            }
        }
        requireNotNull(readTurnJob(workspaceId, sessionId, turnId))
    }

    @Synchronized
    fun turnJob(workspaceId: String, sessionId: String, turnId: String): ChatTurnJob? =
        readTurnJob(workspaceId, sessionId, turnId)

    @Synchronized
    fun markTurnRunning(workspaceId: String, sessionId: String, turnId: String): Boolean =
        updateTurnState(workspaceId, sessionId, turnId, "queued", "running", null)

    @Synchronized
    fun markTurnFailed(workspaceId: String, sessionId: String, turnId: String, error: String): Boolean =
        updateTurnState(workspaceId, sessionId, turnId, "running", "failed", error.take(MAX_TURN_ERROR_LENGTH))

    @Synchronized
    fun markTurnCompleted(workspaceId: String, sessionId: String, turnId: String): Boolean =
        updateTurnState(workspaceId, sessionId, turnId, "running", "completed", null)

    @Synchronized
    fun cancelTurn(workspaceId: String, sessionId: String, turnId: String): ChatTurnJob? = transaction {
        val existing = readTurnJob(workspaceId, sessionId, turnId) ?: return@transaction null
        if (existing.state in setOf("queued", "running")) {
            connection.prepareStatement(
                """UPDATE chat_turn_jobs SET state='cancelled',error=NULL,updated_at=?
                    WHERE workspace_id=? AND session_id=? AND turn_id=? AND state IN ('queued','running')""".trimIndent(),
            ).use { statement ->
                statement.setString(1, Instant.now().toString())
                statement.setString(2, workspaceId)
                statement.setString(3, sessionId)
                statement.setString(4, turnId)
                statement.executeUpdate()
            }
        }
        readTurnJob(workspaceId, sessionId, turnId)
    }

    private fun updateTurnState(
        workspaceId: String,
        sessionId: String,
        turnId: String,
        from: String,
        to: String,
        error: String?,
    ): Boolean = connection.prepareStatement(
        "UPDATE chat_turn_jobs SET state=?,error=?,updated_at=? WHERE workspace_id=? AND session_id=? AND turn_id=? AND state=?",
    ).use { statement ->
        statement.setString(1, to)
        statement.setString(2, error)
        statement.setString(3, Instant.now().toString())
        statement.setString(4, workspaceId)
        statement.setString(5, sessionId)
        statement.setString(6, turnId)
        statement.setString(7, from)
        statement.executeUpdate() == 1
    }

    private fun readTurnJob(workspaceId: String, sessionId: String, turnId: String): ChatTurnJob? =
        connection.prepareStatement(
            """SELECT j.turn_id,j.session_id,m.content,j.strategy,j.top_k,j.provider_id,j.model_id,j.state,j.error,j.created_at,j.updated_at
                FROM chat_turn_jobs j
                JOIN chat_messages m ON m.session_id=j.session_id AND m.turn_id=j.turn_id AND m.role='USER'
                WHERE j.workspace_id=? AND j.session_id=? AND j.turn_id=?""".trimIndent(),
        ).use { statement ->
            statement.setString(1, workspaceId)
            statement.setString(2, sessionId)
            statement.setString(3, turnId)
            statement.executeQuery().use { rows ->
                if (!rows.next()) null else ChatTurnJob(
                    turnId = rows.getString(1),
                    sessionId = rows.getString(2),
                    question = rows.getString(3),
                    strategy = ChunkStrategy.valueOf(rows.getString(4)),
                    topK = rows.getInt(5).let { if (rows.wasNull()) null else it },
                    selection = ModelSelection(rows.getString(6), rows.getString(7)),
                    state = rows.getString(8),
                    error = rows.getString(9),
                    createdAt = rows.getString(10),
                    updatedAt = rows.getString(11),
                )
            }
        }
    @Synchronized
    fun assistantMessage(sessionId: String, turnId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): ChatMessage? {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
        return connection.prepareStatement(
            """SELECT message_id,content,document_citations_json,memory_references_json,created_at,provider_id,model_id,elapsed_ms
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
                    modelSelection = readModelSelection(rows.getString(6), rows.getString(7)),
                    elapsedMs = rows.getLong(8).let { if (rows.wasNull()) null else it },
                )
            }
        }
    }

    @Synchronized
    fun recentMessages(
        sessionId: String,
        limit: Int,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): List<ChatMessage> {
        requireUuid(sessionId, "session ID")
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
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
    fun taskState(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): ChatTaskState {
        requireUuid(sessionId, "session ID")
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
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
        sharedFacts: List<ChatMemoryFactDraft> = emptyList(),
        sessionTitle: String? = null,
        modelSelection: ModelSelection? = null,
        elapsedMs: Long? = null,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatMessage = transaction {
        requireUuid(sessionId, "session ID")
        requireUuid(turnId, "turn ID")
        require(sessionExists(sessionId, workspaceId)) { "Сессия чата не найдена." }
        require(answer.isNotBlank() && answer.length <= MAX_MESSAGE_LENGTH)
        require((modelSelection == null) == (elapsedMs == null) && (elapsedMs == null || elapsedMs >= 0)) {
            "Для ответа нужны согласованные модель и время обработки."
        }
        val existing = connection.prepareStatement(
            """SELECT message_id,content,document_citations_json,memory_references_json,created_at,provider_id,model_id,elapsed_ms
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
                    modelSelection = readModelSelection(rows.getString(6), rows.getString(7)),
                    elapsedMs = rows.getLong(8).let { if (rows.wasNull()) null else it },
                )
            }
        }
        if (existing != null) return@transaction existing
        val jobState = connection.prepareStatement(
            "SELECT state FROM chat_turn_jobs WHERE workspace_id=? AND session_id=? AND turn_id=?",
        ).use { statement ->
            statement.setString(1, workspaceId)
            statement.setString(2, sessionId)
            statement.setString(3, turnId)
            statement.executeQuery().use { rows -> if (rows.next()) rows.getString(1) else null }
        }
        if (jobState != null && jobState != "running") throw ChatTurnCancelledException()
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
                message_id,session_id,turn_id,role,content,document_citations_json,memory_references_json,created_at,
                provider_id,model_id,elapsed_ms
            ) VALUES (?,?,?,'ASSISTANT',?,?,?,?,?,?,?)""".trimIndent(),
        ).use { statement ->
            statement.setString(1, messageId)
            statement.setString(2, sessionId)
            statement.setString(3, turnId)
            statement.setString(4, answer)
            statement.setString(5, json.encodeToString(citations))
            statement.setString(6, json.encodeToString(memoryReferences))
            statement.setString(7, now)
            statement.setString(8, modelSelection?.providerId)
            statement.setString(9, modelSelection?.modelId)
            if (elapsedMs == null) statement.setNull(10, java.sql.Types.INTEGER) else statement.setLong(10, elapsedMs)
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
            saveMemoryFact(ChatMemoryScope.SESSION, sessionId, fact.content, fact.embedding, fact.embeddingModel, workspaceId)
        }
        sharedFacts.forEach { fact ->
            saveMemoryFact(ChatMemoryScope.SHARED, null, fact.content, fact.embedding, fact.embeddingModel, workspaceId)
        }
        if (sessionTitle != null) {
            connection.prepareStatement(
                "UPDATE chat_sessions SET title=?,title_generated=1,updated_at=? WHERE session_id=? AND workspace_id=? AND title_generated=0",
            ).use { statement ->
                statement.setString(1, sessionTitle)
                statement.setString(2, now)
                statement.setString(3, sessionId)
                statement.setString(4, workspaceId)
                require(statement.executeUpdate() == 1) { "Название сессии уже создано." }
            }
        }
        connection.prepareStatement("UPDATE chat_sessions SET updated_at=? WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, now)
            statement.setString(2, sessionId)
            statement.setString(3, workspaceId)
            statement.executeUpdate()
        }
        if (jobState != null) {
            connection.prepareStatement(
                "UPDATE chat_turn_jobs SET state='completed',error=NULL,updated_at=? WHERE workspace_id=? AND session_id=? AND turn_id=? AND state='running'",
            ).use { statement ->
                statement.setString(1, now)
                statement.setString(2, workspaceId)
                statement.setString(3, sessionId)
                statement.setString(4, turnId)
                if (statement.executeUpdate() != 1) throw ChatTurnCancelledException()
            }
        }
        ChatMessage(
            messageId,
            turnId,
            ChatMessageRole.ASSISTANT,
            answer,
            citations,
            memoryReferences,
            now,
            modelSelection,
            elapsedMs,
        )
    }

    @Synchronized
    fun saveSessionFact(
        sessionId: String,
        content: String,
        embedding: List<Float>,
        embeddingModel: String,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatMemoryFact = saveMemoryFact(ChatMemoryScope.SESSION, sessionId, content, embedding, embeddingModel, workspaceId)

    @Synchronized
    fun saveSharedFact(
        content: String,
        embedding: List<Float>,
        embeddingModel: String,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatMemoryFact = saveMemoryFact(ChatMemoryScope.SHARED, null, content, embedding, embeddingModel, workspaceId)

    @Synchronized
    fun updateSharedFact(
        memoryId: String,
        content: String,
        embedding: List<Float>,
        embeddingModel: String,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): Boolean = transaction {
        requireUuid(memoryId, "memory ID")
        require(content.isNotBlank() && content.length <= MAX_SHARED_FACT_LENGTH)
        require(embedding.isNotEmpty() && embedding.all(Float::isFinite))
        require(embeddingModel.isNotBlank() && embeddingModel.length <= MAX_EMBEDDING_MODEL_LENGTH)
        connection.prepareStatement(
            "UPDATE chat_shared_facts SET content=?,embedding=?,embedding_model=?,updated_at=? WHERE memory_id=? AND workspace_id=?",
        ).use { statement ->
            statement.setString(1, content.trim())
            statement.setBytes(2, encodeEmbedding(embedding))
            statement.setString(3, embeddingModel)
            statement.setString(4, Instant.now().toString())
            statement.setString(5, memoryId)
            statement.setString(6, workspaceId)
            statement.executeUpdate() == 1
        }
    }

    @Synchronized
    fun memoryFacts(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): List<ChatMemoryFact> {
        requireUuid(sessionId, "session ID")
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
        val sessionFacts = queryMemoryFacts(
            """SELECT memory_id,session_id,content,embedding,embedding_model,created_at
                FROM chat_session_facts WHERE session_id=?""".trimIndent(),
            sessionId,
            ChatMemoryScope.SESSION,
            workspaceId,
        )
        return sessionFacts + queryMemoryFacts(
            """SELECT memory_id,NULL,content,embedding,embedding_model,created_at
                FROM chat_shared_facts WHERE workspace_id=? ORDER BY created_at,memory_id""".trimIndent(),
            null,
            ChatMemoryScope.SHARED,
            workspaceId,
        )
    }

    @Synchronized
    fun sharedFacts(workspaceId: String = DEFAULT_WORKSPACE_ID): List<ChatMemoryFact> = queryMemoryFacts(
        """SELECT memory_id,NULL,content,embedding,embedding_model,created_at
            FROM chat_shared_facts WHERE workspace_id=? ORDER BY created_at,memory_id""".trimIndent(),
        null,
        ChatMemoryScope.SHARED,
        workspaceId,
    )

    @Synchronized
    fun replaceTaskState(
        sessionId: String,
        taskState: ChatTaskState,
        sessionFacts: List<ChatMemoryFactDraft>,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ) = transaction {
        requireUuid(sessionId, "session ID")
        if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
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
            saveMemoryFact(ChatMemoryScope.SESSION, sessionId, fact.content, fact.embedding, fact.embeddingModel, workspaceId)
        }
        connection.prepareStatement("UPDATE chat_sessions SET updated_at=? WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, now)
            statement.setString(2, sessionId)
            statement.setString(3, workspaceId)
            statement.executeUpdate()
        }
    }

    @Synchronized
    fun deleteSharedFact(memoryId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): Boolean {
        requireUuid(memoryId, "memory ID")
        return connection.prepareStatement("DELETE FROM chat_shared_facts WHERE memory_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, memoryId)
            statement.setString(2, workspaceId)
            statement.executeUpdate() == 1
        }
    }

    @Synchronized
    fun deleteSession(sessionId: String, workspaceId: String = DEFAULT_WORKSPACE_ID): Boolean {
        requireUuid(sessionId, "session ID")
        return connection.prepareStatement("DELETE FROM chat_sessions WHERE session_id=? AND workspace_id=?").use { statement ->
            statement.setString(1, sessionId)
            statement.setString(2, workspaceId)
            statement.executeUpdate() == 1
        }
    }

    private fun saveMemoryFact(
        scope: ChatMemoryScope,
        sessionId: String?,
        content: String,
        embedding: List<Float>,
        embeddingModel: String,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): ChatMemoryFact = transaction {
        require(content.isNotBlank() && content.length <= MAX_MEMORY_FACT_LENGTH)
        require(embedding.isNotEmpty() && embedding.all(Float::isFinite))
        require(embeddingModel.isNotBlank() && embeddingModel.length <= MAX_EMBEDDING_MODEL_LENGTH)
        if (scope == ChatMemoryScope.SESSION) {
            requireUuid(requireNotNull(sessionId), "session ID")
            if (!sessionExists(sessionId, workspaceId)) throw ChatSessionNotFoundException()
        } else {
            require(sessionId == null)
        }
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
                "SELECT memory_id,created_at FROM $table WHERE workspace_id=? AND lower(trim(content))=lower(trim(?))",
            ).use { statement ->
                statement.setString(1, workspaceId)
                statement.setString(2, content)
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
                "UPDATE $table SET content=?,embedding=?,embedding_model=?,updated_at=? WHERE memory_id=? AND workspace_id=?"
            }
            connection.prepareStatement(sql).use { statement ->
                statement.setString(1, content.trim())
                statement.setBytes(2, encodeEmbedding(embedding))
                statement.setString(3, embeddingModel)
                if (scope == ChatMemoryScope.SESSION) statement.setString(4, id)
                else {
                    statement.setString(4, now)
                    statement.setString(5, id)
                    statement.setString(6, workspaceId)
                }
                statement.executeUpdate()
            }
        } else {
            val sql = if (scope == ChatMemoryScope.SESSION) {
                "INSERT INTO $table(memory_id,session_id,content,embedding,embedding_model,created_at) VALUES (?,?,?,?,?,?)"
            } else {
                "INSERT INTO $table(memory_id,content,embedding,embedding_model,created_at,updated_at,workspace_id) VALUES (?,?,?,?,?,?,?)"
            }
            connection.prepareStatement(sql).use { statement ->
                statement.setString(1, id)
                var index = 2
                if (scope == ChatMemoryScope.SESSION) statement.setString(index++, sessionId)
                statement.setString(index++, content.trim())
                statement.setBytes(index++, encodeEmbedding(embedding))
                statement.setString(index++, embeddingModel)
                statement.setString(index++, now)
                if (scope == ChatMemoryScope.SHARED) {
                    statement.setString(index++, now)
                    statement.setString(index, workspaceId)
                }
                statement.executeUpdate()
            }
        }
        ChatMemoryFact(scope, id, sessionId, content.trim(), embedding, embeddingModel, createdAt)
    }

    private fun queryMemoryFacts(
        sql: String,
        sessionId: String?,
        scope: ChatMemoryScope,
        workspaceId: String = DEFAULT_WORKSPACE_ID,
    ): List<ChatMemoryFact> =
        connection.prepareStatement(sql).use { statement ->
            if (sessionId != null) statement.setString(1, sessionId)
            else if (scope == ChatMemoryScope.SHARED) statement.setString(1, workspaceId)
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
    private fun ensureColumn(table: String, column: String, declaration: String) {
        val exists = connection.createStatement().use { statement ->
            statement.executeQuery("PRAGMA table_info($table)").use { rows ->
                var found = false
                while (rows.next()) if (rows.getString("name") == column) found = true
                found
            }
        }
        if (!exists) {
            connection.createStatement().use { statement ->
                statement.execute("ALTER TABLE $table ADD COLUMN $column $declaration")
            }
        }
    }

    private fun readModelSelection(providerId: String?, modelId: String?): ModelSelection? = when {
        providerId == null && modelId == null -> null
        providerId != null && modelId != null -> ModelSelection(providerId, modelId)
        else -> error("В хранилище чата сохранена неполная модель.")
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
        const val CHAT_SCHEMA_VERSION = 17
        const val LEGACY_OWNER_WORKSPACE_ID = "legacy-owner"
        const val DEFAULT_WORKSPACE_ID = LEGACY_OWNER_WORKSPACE_ID
        const val MAX_MESSAGE_LENGTH = 4_000
        const val MAX_HISTORY_MESSAGES = 8
        const val MAX_MEMORY_FACT_LENGTH = 1_100
        const val MAX_SHARED_FACT_LENGTH = MAX_MEMORY_FACT_LENGTH
        const val MAX_TURN_ERROR_LENGTH = 1_000
        const val MAX_EMBEDDING_MODEL_LENGTH = 120
        const val MAX_TITLE_LENGTH = 60
        const val NEW_SESSION_TITLE = "Новая сессия"
        private val WHITESPACE = Regex("\\s+")
    }
}
