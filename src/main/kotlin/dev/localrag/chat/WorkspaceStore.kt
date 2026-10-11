package dev.localrag.chat

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import java.security.MessageDigest
import java.security.SecureRandom
import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.Base64
import java.util.UUID

/** Persistent, opaque browser credentials mapped to isolated chat workspaces. */
class WorkspaceStore(databasePath: Path) : AutoCloseable {
    private val connection: Connection = DriverManager.getConnection(
        "jdbc:sqlite:${databasePath.toAbsolutePath().normalize()}",
    )

    init {
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA foreign_keys=ON")
            statement.execute("PRAGMA busy_timeout=5000")
        }
        connection.autoCommit = false
        try {
            connection.createStatement().use { statement ->
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS browser_workspaces (
                        workspace_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS browser_workspace_sessions (
                        token_hash TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL REFERENCES browser_workspaces(workspace_id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        authenticated INTEGER NOT NULL DEFAULT 0
                    )""".trimIndent(),
                )
                val columns = statement.executeQuery("PRAGMA table_info(browser_workspace_sessions)").use { rows ->
                    buildSet { while (rows.next()) add(rows.getString("name")) }
                }
                if ("authenticated" !in columns) {
                    statement.execute(
                        "ALTER TABLE browser_workspace_sessions ADD COLUMN authenticated INTEGER NOT NULL DEFAULT 0",
                    )
                }
                statement.execute(
                    """CREATE TABLE IF NOT EXISTS browser_workspace_settings (
                        setting_key TEXT PRIMARY KEY,
                        setting_value TEXT NOT NULL
                    )""".trimIndent(),
                )
            }
            connection.commit()
        } catch (error: Throwable) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = true
        }
    }

    @Synchronized
    fun resolveOrCreate(token: String?, mayClaimLegacyWorkspace: Boolean): WorkspaceSession {
        val existing = token?.let(::workspaceForToken)
        if (token != null && existing != null) return WorkspaceSession(existing, token)

        val workspaceId = transaction {
            if (mayClaimLegacyWorkspace && !legacyWorkspaceClaimed()) {
                connection.prepareStatement(
                    "INSERT INTO browser_workspace_settings(setting_key,setting_value) VALUES('legacy_owner_claimed','1')",
                ).use { it.executeUpdate() }
                ChatStore.LEGACY_OWNER_WORKSPACE_ID
            } else {
                UUID.randomUUID().toString().also(::insertWorkspace)
            }
        }
        if (workspaceId == ChatStore.LEGACY_OWNER_WORKSPACE_ID) insertWorkspaceIfMissing(workspaceId)
        return issueSession(workspaceId)
    }

    @Synchronized
    fun findWorkspace(token: String?): String? = token?.let(::workspaceForToken)

    @Synchronized
    fun isAuthenticated(token: String?): Boolean = token?.let(::authenticatedForToken) ?: false

    @Synchronized
    fun markAuthenticated(token: String): Boolean {
        if (token.length !in 32..128) return false
        val now = Instant.now().toString()
        return connection.prepareStatement(
            "UPDATE browser_workspace_sessions SET authenticated=1,expires_at=? WHERE token_hash=? AND expires_at>?",
        ).use { statement ->
            statement.setString(1, Instant.now().plus(SESSION_LIFETIME_DAYS, ChronoUnit.DAYS).toString())
            statement.setString(2, tokenHash(token))
            statement.setString(3, now)
            statement.executeUpdate() == 1
        }
    }

    @Synchronized
    fun createRemoteWorkspace(): WorkspaceSession {
        val workspaceId = UUID.randomUUID().toString()
        insertWorkspaceIfMissing(workspaceId)
        return issueSession(workspaceId)
    }

    private fun workspaceForToken(token: String): String? {
        if (token.length !in 32..128) return null
        val hash = tokenHash(token)
        val now = Instant.now()
        val (workspaceId, expiresAt) = connection.prepareStatement(
            "SELECT workspace_id,expires_at FROM browser_workspace_sessions WHERE token_hash=?",
        ).use { statement ->
            statement.setString(1, hash)
            statement.executeQuery().use { rows ->
                if (!rows.next()) return null
                rows.getString(1) to Instant.parse(rows.getString(2))
            }
        }
        if (!expiresAt.isAfter(now)) return null
        if (!expiresAt.isAfter(now.plus(SESSION_REFRESH_THRESHOLD_DAYS, ChronoUnit.DAYS))) {
            connection.prepareStatement(
                "UPDATE browser_workspace_sessions SET expires_at=? WHERE token_hash=? AND expires_at=?",
            ).use { statement ->
                statement.setString(1, now.plus(SESSION_LIFETIME_DAYS, ChronoUnit.DAYS).toString())
                statement.setString(2, hash)
                statement.setString(3, expiresAt.toString())
                statement.executeUpdate()
            }
        }
        return workspaceId
    }

    private fun authenticatedForToken(token: String): Boolean {
        if (token.length !in 32..128) return false
        val now = Instant.now()
        return connection.prepareStatement(
            "SELECT authenticated FROM browser_workspace_sessions WHERE token_hash=? AND expires_at>?",
        ).use { statement ->
            statement.setString(1, tokenHash(token))
            statement.setString(2, now.toString())
            statement.executeQuery().use { rows -> rows.next() && rows.getInt(1) == 1 }
        }
    }

    private fun issueSession(workspaceId: String): WorkspaceSession {
        val bytes = ByteArray(TOKEN_BYTES).also(SECURE_RANDOM::nextBytes)
        val token = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
        val now = Instant.now()
        connection.prepareStatement(
            "INSERT INTO browser_workspace_sessions(token_hash,workspace_id,created_at,expires_at) VALUES(?,?,?,?)",
        ).use { statement ->
            statement.setString(1, tokenHash(token))
            statement.setString(2, workspaceId)
            statement.setString(3, now.toString())
            statement.setString(4, now.plus(SESSION_LIFETIME_DAYS, ChronoUnit.DAYS).toString())
            statement.executeUpdate()
        }
        return WorkspaceSession(workspaceId, token)
    }

    private fun insertWorkspaceIfMissing(workspaceId: String) {
        connection.prepareStatement(
            "INSERT OR IGNORE INTO browser_workspaces(workspace_id,created_at) VALUES(?,?)",
        ).use { statement ->
            statement.setString(1, workspaceId)
            statement.setString(2, Instant.now().toString())
            statement.executeUpdate()
        }
    }

    private fun insertWorkspace(workspaceId: String) = insertWorkspaceIfMissing(workspaceId)

    private fun legacyWorkspaceClaimed(): Boolean = connection.prepareStatement(
        "SELECT 1 FROM browser_workspace_settings WHERE setting_key='legacy_owner_claimed'",
    ).use { statement -> statement.executeQuery().use { it.next() } }

    private inline fun <T> transaction(block: () -> T): T {
        val previous = connection.autoCommit
        connection.autoCommit = false
        try {
            val result = block()
            connection.commit()
            return result
        } catch (error: Throwable) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = previous
        }
    }


    override fun close() = connection.close()

    private fun tokenHash(token: String): String = Base64.getEncoder().encodeToString(
        MessageDigest.getInstance("SHA-256").digest(token.toByteArray(UTF_8)),
    )

    data class WorkspaceSession(val workspaceId: String, val token: String)

    companion object {
        const val SESSION_COOKIE = "local_rag_workspace"
        const val SESSION_LIFETIME_DAYS = 30L
        private const val SESSION_REFRESH_THRESHOLD_DAYS = SESSION_LIFETIME_DAYS - 1
        private const val TOKEN_BYTES = 32
        private val SECURE_RANDOM = SecureRandom()
    }
}
