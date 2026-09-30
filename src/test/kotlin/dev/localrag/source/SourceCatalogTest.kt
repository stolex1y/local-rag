package dev.localrag.source

import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredSource
import dev.localrag.index.SqliteIndexRepository
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.sql.DriverManager
import java.nio.file.Path
import java.nio.file.attribute.PosixFileAttributeView
import java.nio.file.attribute.PosixFilePermissions
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.io.TempDir

class SourceCatalogTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `mixed upload keeps supported source while reporting invalid UTF8 per file`() {
        val database = temporaryDirectory.resolve("index.sqlite")
        val sourcesPath = temporaryDirectory.resolve("sources")
        val repository = SqliteIndexRepository(database)
        try {
            val catalog = SourceCatalog(sourcesPath, repository)
            val text = "Orbit markers are listed here.\n".toByteArray()
            val batch = catalog.beginImport(
                listOf(
                    ImportFileMetadata("../../guide.txt", text.size.toLong()),
                    ImportFileMetadata("damaged.bin", 2),
                ),
            )
            val bytes = mapOf(batch.files[0].fileId to text, batch.files[1].fileId to byteArrayOf(0, 0xff.toByte()))
            batch.files.forEach { slot ->
                catalog.writeImportedFile(batch.importId, slot.fileId, ByteArrayInputStream(bytes.getValue(slot.fileId)))
            }

            val completion = catalog.finishImport(batch.importId)
            val results = completion.files.associateBy { it.source.name }
            val imported = results.getValue("guide.txt").source
            val rejected = results.getValue("damaged.bin").source
            val stored = assertNotNull(repository.storedSource(imported.sourceId))
            val storedPath = sourcesPath.resolve(assertNotNull(stored.storageKey))

            assertEquals(SourceStatus.PENDING, imported.status)
            assertEquals(SourceStatus.UNSUPPORTED, rejected.status)
            assertTrue(rejected.error.orEmpty().contains("UTF-8"))
            assertEquals(text.toList(), Files.readAllBytes(storedPath).toList())
            assertTrue(storedPath.parent == sourcesPath)
            assertEquals(2, completion.sources.size)
            assertPrivateModes(sourcesPath, storedPath)
        } finally {
            repository.close()
        }
    }

    @Test
    fun `import limits reject announced file and batch sizes before opening uploads`() {
        val repository = SqliteIndexRepository(temporaryDirectory.resolve("limits.sqlite"))
        try {
            val catalog = SourceCatalog(temporaryDirectory.resolve("sources"), repository)

            assertFailsWith<IllegalArgumentException> {
                catalog.beginImport(listOf(ImportFileMetadata("too-large.txt", SourceCatalog.MAX_FILE_BYTES + 1)))
            }
            assertFailsWith<IllegalArgumentException> {
                catalog.beginImport(
                    listOf(
                        ImportFileMetadata("one.txt", 200L * 1024 * 1024),
                        ImportFileMetadata("two.txt", 200L * 1024 * 1024),
                        ImportFileMetadata("three.txt", 200L * 1024 * 1024),
                    ),
                )
            }
            assertTrue(catalog.sources().isEmpty())
        } finally {
            repository.close()
        }
    }

    @Test
    fun `pending sources and active imports share the 512 MiB indexing limit`() {
        val repository = SqliteIndexRepository(temporaryDirectory.resolve("index-run-limit.sqlite"))
        try {
            val catalog = SourceCatalog(temporaryDirectory.resolve("sources"), repository)
            val pendingBytes = SourceCatalog.MAX_IMPORT_BYTES - SourceCatalog.MAX_FILE_BYTES
            val pending = SourceRecord(
                sourceId = java.util.UUID.randomUUID().toString(),
                name = "existing.txt",
                type = SourceType.TEXT,
                sizeBytes = pendingBytes,
                status = SourceStatus.PENDING,
                createdAt = "2026-09-30T00:00:00Z",
            )
            repository.registerSource(StoredSource(pending, null))

            val activeBatch = catalog.beginImport(
                listOf(ImportFileMetadata("next.txt", SourceCatalog.MAX_FILE_BYTES)),
            )
            assertFailsWith<IllegalArgumentException> {
                catalog.beginImport(listOf(ImportFileMetadata("overflow.txt", 1)))
            }
            assertEquals(1, catalog.sources().size)

            catalog.finishImport(activeBatch.importId)
            val exactBatch = catalog.beginImport(
                listOf(ImportFileMetadata("exact-fit.txt", SourceCatalog.MAX_FILE_BYTES)),
            )
            assertEquals(SourceCatalog.MAX_FILE_BYTES, exactBatch.totalBytes)
        } finally {
            repository.close()
        }
    }

    @Test
    fun `actual byte mismatch becomes a failed source without a stored copy`() {
        val repository = SqliteIndexRepository(temporaryDirectory.resolve("mismatch.sqlite"))
        try {
            val catalog = SourceCatalog(temporaryDirectory.resolve("sources"), repository)
            val batch = catalog.beginImport(listOf(ImportFileMetadata("guide.md", 4)))
            catalog.writeImportedFile(batch.importId, batch.files.single().fileId, ByteArrayInputStream("too-long".toByteArray()))

            val completion = catalog.finishImport(batch.importId)
            val source = completion.files.single().source

            assertEquals(SourceStatus.FAILED, source.status)
            assertTrue(source.error.orEmpty().contains("превысил объявленный лимит"))
            assertEquals(null, repository.storedSource(source.sourceId)?.storageKey)
            assertFalse(Files.list(temporaryDirectory.resolve("sources")).use { it.findAny().isPresent })
        } finally {
            repository.close()
        }
    }

    @Test
    fun `source deletion requires confirmation and removes both copy and database record`() {
        val repository = SqliteIndexRepository(temporaryDirectory.resolve("delete.sqlite"))
        try {
            var removedReference: String? = null
            val sourcesPath = temporaryDirectory.resolve("sources")
            val catalog = SourceCatalog(sourcesPath, repository) { removedReference = it }
            val batch = catalog.beginImport(listOf(ImportFileMetadata("guide.txt", 5)))
            catalog.writeImportedFile(batch.importId, batch.files.single().fileId, ByteArrayInputStream("orbit".toByteArray()))
            val source = catalog.finishImport(batch.importId).files.single().source
            val storageKey = assertNotNull(repository.storedSource(source.sourceId)?.storageKey)
            val copy = sourcesPath.resolve(storageKey)

            assertFailsWith<IllegalArgumentException> { catalog.deleteSource(source.sourceId, confirmed = false) }
            assertTrue(Files.exists(copy))
            assertNotNull(repository.source(source.sourceId))

            catalog.deleteSource(source.sourceId, confirmed = true)

            assertFalse(Files.exists(copy))
            assertEquals(null, repository.source(source.sourceId))
            assertEquals(source.sourceId, removedReference)
        } finally {
            repository.close()
        }
    }

    @Test
    fun `source records and app-managed copies survive catalog restart`() {
        val database = temporaryDirectory.resolve("restart.sqlite")
        val sourcesPath = temporaryDirectory.resolve("sources")
        var repository = SqliteIndexRepository(database)
        val batch: ImportBatch
        val sourceId: String
        try {
            val catalog = SourceCatalog(sourcesPath, repository)
            batch = catalog.beginImport(listOf(ImportFileMetadata("guide.md", 5)))
            sourceId = catalog.writeImportedFile(
                batch.importId,
                batch.files.single().fileId,
                ByteArrayInputStream("orbit".toByteArray()),
            ).source.sourceId
            catalog.finishImport(batch.importId)
        } finally {
            repository.close()
        }

        repository = SqliteIndexRepository(database)
        try {
            val catalog = SourceCatalog(sourcesPath, repository)
            val source = catalog.sources().single()

            assertEquals(sourceId, source.sourceId)
            assertEquals(SourceStatus.PENDING, source.status)
            assertTrue(Files.exists(sourcesPath.resolve(assertNotNull(repository.storedSource(sourceId)?.storageKey))))
        } finally {
            repository.close()
        }
    }

    @Test
    fun `identical reimport requeues a failed source and restores a missing stored copy`() {
        val repository = SqliteIndexRepository(temporaryDirectory.resolve("failed-reimport.sqlite"))
        try {
            val sourcesPath = temporaryDirectory.resolve("sources")
            val bytes = "Retryable source.\n".toByteArray()
            val catalog = SourceCatalog(sourcesPath, repository)
            val firstBatch = catalog.beginImport(listOf(ImportFileMetadata("guide.md", bytes.size.toLong())))
            val firstSource = catalog.writeImportedFile(
                firstBatch.importId,
                firstBatch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(firstBatch.importId)
            val stored = assertNotNull(repository.storedSource(firstSource.sourceId))
            val copy = sourcesPath.resolve(assertNotNull(stored.storageKey))

            repository.markSourceStatus(firstSource.sourceId, SourceStatus.INDEXING)
            repository.markSourceStatus(
                firstSource.sourceId,
                SourceStatus.FAILED,
                error = "Embedding failed.",
                unsearchablePages = listOf(2),
            )
            val failed = assertNotNull(repository.source(firstSource.sourceId))
            assertEquals(SourceStatus.FAILED, failed.status)
            assertEquals("Embedding failed.", failed.error)
            assertEquals(listOf(2), failed.unsearchablePages)
            Files.delete(copy)
            assertFalse(Files.exists(copy))

            val retryBatch = catalog.beginImport(listOf(ImportFileMetadata("renamed.md", bytes.size.toLong())))
            val retried = catalog.writeImportedFile(
                retryBatch.importId,
                retryBatch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(retryBatch.importId)

            assertEquals(firstSource.sourceId, retried.sourceId)
            assertEquals(SourceStatus.PENDING, retried.status)
            assertEquals(null, retried.error)
            assertTrue(retried.unsearchablePages.isEmpty())
            assertEquals(SourceStatus.PENDING, repository.source(firstSource.sourceId)?.status)
            assertEquals(listOf(firstSource.sourceId), catalog.pendingSources().map { it.sourceId })
            assertTrue(Files.isRegularFile(copy))
            assertTrue(Files.readAllBytes(copy).contentEquals(bytes))
        } finally {
            repository.close()
        }
    }

    @Test
    fun `reimport after failed benchmark cleanup creates a fresh source`() {
        val database = temporaryDirectory.resolve("deleting-reimport.sqlite")
        val repository = SqliteIndexRepository(database)
        try {
            val sourcesPath = temporaryDirectory.resolve("sources")
            val bytes = "Orbiting source.\n".toByteArray()
            val catalog = SourceCatalog(sourcesPath, repository) {
                throw IllegalStateException("Benchmark cleanup failed.")
            }
            val firstBatch = catalog.beginImport(listOf(ImportFileMetadata("guide.txt", bytes.size.toLong())))
            val firstSource = catalog.writeImportedFile(
                firstBatch.importId,
                firstBatch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(firstBatch.importId)
            val firstCopy = sourcesPath.resolve(assertNotNull(repository.storedSource(firstSource.sourceId)?.storageKey))

            assertFailsWith<IllegalStateException> { catalog.deleteSource(firstSource.sourceId, confirmed = true) }
            assertFalse(Files.exists(firstCopy))
            assertEquals(SourceStatus.DELETING, repository.source(firstSource.sourceId)?.status)
            val originalHash = assertNotNull(repository.storedSource(firstSource.sourceId)?.contentSha256)
            assertEquals(null, repository.sourceByContentSha256(originalHash))
            DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
                connection.prepareStatement("UPDATE sources SET content_sha256=NULL WHERE source_id=?").use { statement ->
                    statement.setString(1, firstSource.sourceId)
                    assertEquals(1, statement.executeUpdate())
                }
            }
            assertFalse(repository.unhashedSourcesOfSize(bytes.size.toLong()).any { it.record.sourceId == firstSource.sourceId })
            assertTrue(catalog.pendingSources().isEmpty())

            val retryBatch = catalog.beginImport(listOf(ImportFileMetadata("guide.txt", bytes.size.toLong())))
            val retried = catalog.writeImportedFile(
                retryBatch.importId,
                retryBatch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(retryBatch.importId)

            assertTrue(firstSource.sourceId != retried.sourceId)
            assertEquals(SourceStatus.DELETING, repository.source(firstSource.sourceId)?.status)
            assertEquals(SourceStatus.PENDING, retried.status)
            assertEquals(listOf(retried.sourceId), catalog.pendingSources().map { it.sourceId })
            val retryCopy = sourcesPath.resolve(assertNotNull(repository.storedSource(retried.sourceId)?.storageKey))
            assertTrue(Files.isRegularFile(retryCopy))
            assertTrue(Files.readAllBytes(retryCopy).contentEquals(bytes))
            assertFalse(Files.exists(firstCopy))
        } finally {
            repository.close()
        }
    }

    @Test
    fun `identical bytes reuse existing source across restart while distinct same-name bytes remain separate`() {
        val database = temporaryDirectory.resolve("deduplicate.sqlite")
        val sourcesPath = temporaryDirectory.resolve("sources")
        val bytes = "Calibration instructions.\n".toByteArray()
        var repository = SqliteIndexRepository(database)
        val original = try {
            val catalog = SourceCatalog(sourcesPath, repository)
            val batch = catalog.beginImport(listOf(ImportFileMetadata("guide.txt", bytes.size.toLong())))
            val source = catalog.writeImportedFile(
                batch.importId,
                batch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(batch.importId)
            source
        } finally {
            repository.close()
        }

        DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
            connection.createStatement().use { statement ->
                statement.execute("DROP INDEX IF EXISTS sources_content_sha256_idx")
                statement.execute("ALTER TABLE sources DROP COLUMN content_sha256")
                statement.execute("PRAGMA user_version=11")
            }
        }

        repository = SqliteIndexRepository(database)
        try {
            val catalog = SourceCatalog(sourcesPath, repository)
            val duplicateBatch = catalog.beginImport(
                listOf(ImportFileMetadata("renamed-copy.txt", bytes.size.toLong())),
            )
            val duplicate = catalog.writeImportedFile(
                duplicateBatch.importId,
                duplicateBatch.files.single().fileId,
                ByteArrayInputStream(bytes),
            ).source
            catalog.finishImport(duplicateBatch.importId)

            assertEquals(original.sourceId, duplicate.sourceId)
            assertEquals(1, catalog.sources().size)
            assertNotNull(repository.storedSource(original.sourceId)?.contentSha256)

            val differentBytes = "Calibration instructions, revised.\n".toByteArray()
            val differentBatch = catalog.beginImport(
                listOf(ImportFileMetadata("guide.txt", differentBytes.size.toLong())),
            )
            val different = catalog.writeImportedFile(
                differentBatch.importId,
                differentBatch.files.single().fileId,
                ByteArrayInputStream(differentBytes),
            ).source
            catalog.finishImport(differentBatch.importId)

            assertTrue(different.sourceId != original.sourceId)
            assertEquals(2, catalog.sources().size)
        } finally {
            repository.close()
        }
    }

    private fun assertPrivateModes(directory: Path, file: Path) {
        if (Files.getFileAttributeView(directory, PosixFileAttributeView::class.java) != null) {
            assertEquals(PosixFilePermissions.fromString("rwx------"), Files.getPosixFilePermissions(directory))
            assertEquals(PosixFilePermissions.fromString("rw-------"), Files.getPosixFilePermissions(file))
        }
    }
}
