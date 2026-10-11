package dev.localrag.chat

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import java.sql.DriverManager
import java.time.Instant
import java.time.temporal.ChronoUnit

class WorkspaceStoreTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `workspace expiry slides near expiration without writes on routine requests`() {
        val database = temporaryDirectory.resolve("workspace.sqlite")
        val session = WorkspaceStore(database).use { it.resolveOrCreate(null, mayClaimLegacyWorkspace = false) }
        val initialExpiry = expiry(database)

        repeat(5) { assertEquals(session.workspaceId, WorkspaceStore(database).use { it.findWorkspace(session.token) }) }
        assertEquals(initialExpiry, expiry(database))

        val nearExpiry = Instant.now().plus(2, ChronoUnit.DAYS).toString()
        DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
            connection.prepareStatement("UPDATE browser_workspace_sessions SET expires_at=?").use { statement ->
                statement.setString(1, nearExpiry)
                assertEquals(1, statement.executeUpdate())
            }
        }

        WorkspaceStore(database).use { assertEquals(session.workspaceId, it.findWorkspace(session.token)) }
        val extendedExpiry = expiry(database)
        assertTrue(Instant.parse(extendedExpiry).isAfter(Instant.now().plus(29, ChronoUnit.DAYS)))
        repeat(5) { assertEquals(session.workspaceId, WorkspaceStore(database).use { it.findWorkspace(session.token) }) }
        assertEquals(extendedExpiry, expiry(database))
    }

    private fun expiry(database: Path): String = DriverManager.getConnection(
        "jdbc:sqlite:${database.toAbsolutePath()}",
    ).use { connection ->
        connection.createStatement().use { statement ->
            statement.executeQuery("SELECT expires_at FROM browser_workspace_sessions").use { rows ->
                assertTrue(rows.next())
                rows.getString(1)
            }
        }
    }
}
