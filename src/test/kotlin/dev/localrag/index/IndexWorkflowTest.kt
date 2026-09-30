package dev.localrag.index

import dev.localrag.app.RagService
import dev.localrag.domain.ChatPort
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.EmbeddingPort
import dev.localrag.domain.IndexProgressUpdate
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceSegment
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.source.ImportFileMetadata
import dev.localrag.source.SourceCatalog
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.io.TempDir

class IndexWorkflowTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `indexing adds all sources keeps per-file failures citations and accurate progress across restart`() {
        val database = temporaryDirectory.resolve("collection.sqlite")
        val sourcesDirectory = temporaryDirectory.resolve("sources")
        val repository = SqliteIndexRepository(database)
        var repositoryOpen = true
        try {
            val catalog = SourceCatalog(sourcesDirectory, repository)
            val bytes = mapOf(
                "guide.md" to "ORBIT_EVIDENCE: markers follow a stable path.\n".toByteArray(),
                "broken.txt" to "broken fixture".toByteArray(),
                "scan.pdf" to "%PDF-1.7\n".toByteArray(),
                "wiring.kt" to "COPPER_EVIDENCE: the winding uses copper wire.\n".toByteArray(),
            )
            val batch = catalog.beginImport(bytes.map { (name, content) -> ImportFileMetadata(name, content.size.toLong()) })
            batch.files.forEach { slot ->
                catalog.writeImportedFile(batch.importId, slot.fileId, ByteArrayInputStream(bytes.getValue(slot.name)))
            }
            catalog.finishImport(batch.importId)

            val updates = mutableListOf<IndexProgressUpdate>()
            val workflow = IndexWorkflow(
                extractor = FixtureSourceExtractor(),
                chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
                embeddings = FixtureEmbeddings(),
                index = repository,
                sourcesDirectory = sourcesDirectory,
            )
            val outcome = workflow.index(repository.pendingSources()) { updates += it }
            val records = catalog.sources().associateBy(SourceRecord::name)
            val orbitId = records.getValue("guide.md").sourceId
            val copperId = records.getValue("wiring.kt").sourceId
            val completions = updates.filter { it.phase == "source-complete" }

            assertEquals(2, outcome.indexedSourceIds.size)
            assertEquals(2, outcome.failedSources.size)
            assertEquals(SourceStatus.FAILED, records.getValue("broken.txt").status)
            assertEquals(SourceStatus.FAILED, records.getValue("scan.pdf").status)
            assertEquals(listOf(1, 2), records.getValue("scan.pdf").unsearchablePages)
            assertEquals(listOf(1, 2, 3, 4), completions.map(IndexProgressUpdate::filesDone))
            assertEquals(2, completions.last().succeeded)
            assertEquals(2, completions.last().failed)
            assertTrue(completions.all { it.filesDone == it.succeeded + it.failed })
            assertTrue(repository.hasSourceLocation(orbitId, SourceLocation(lineStart = 1, lineEnd = 1), "Orbit overview"))
            assertTrue(repository.hasSourceLocation(copperId, SourceLocation(lineStart = 1, lineEnd = 1), "Copper components"))
            assertTrue(repository.chunkCount(orbitId, ChunkStrategy.FIXED_SIZE) > 0)
            assertTrue(repository.chunkCount(copperId, ChunkStrategy.STRUCTURAL) > 0)

            val rag = RagService(repository, FixtureEmbeddings(), EvidenceChat())
            val response = rag.answer(
                "Where are orbit markers?",
                ChunkStrategy.FIXED_SIZE,
                topK = 1,
                selection = ModelSelection("fixture", "fixture-chat"),
            )
            assertEquals("Baseline has no collection evidence.", response.baseline.answer)
            assertEquals("The indexed orbit fragment supports the route.", response.rag.answer)
            assertEquals(orbitId, response.rag.sources.single().sourceId)
            assertEquals("guide.md", response.rag.sources.single().source)
            assertEquals(SourceLocation(lineStart = 1, lineEnd = 1), response.rag.sources.single().location)

            repository.close()
            repositoryOpen = false
            val restarted = SqliteIndexRepository(database)
            try {
                val persisted = assertNotNull(restarted.source(orbitId))
                assertEquals(SourceStatus.READY, persisted.status)
                assertEquals(orbitId, restarted.search(ChunkStrategy.FIXED_SIZE, listOf(1f, 0f, 0f), "fixture-embedding", 1).single().chunk.draft.sourceId)
            } finally {
                restarted.close()
            }
        } finally {
            if (repositoryOpen) repository.close()
        }
    }

    private class FixtureSourceExtractor : SourceExtractor {
        override fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary {
            when (source.name) {
                "broken.txt" -> throw SourceExtractionException("Fixture text decoding failure.")
                "scan.pdf" -> throw SourceExtractionException("No text layer; OCR is not performed.", unsearchablePages = listOf(1, 2))
            }
            val content = Files.readString(storedFile)
            val section = if (source.type == SourceType.MARKDOWN) "Orbit overview" else "Copper components"
            emit(SourceSegment(content, section, SourceLocation(lineStart = 1, lineEnd = 1)))
            return ExtractionSummary(1)
        }
    }

    private class FixtureEmbeddings : EmbeddingPort {
        override val modelName = "fixture-embedding"

        override fun embed(texts: List<String>): List<List<Float>> = texts.map { text ->
            when {
                "orbit" in text.lowercase() -> listOf(1f, 0f, 0f)
                "copper" in text.lowercase() -> listOf(0f, 1f, 0f)
                else -> listOf(0f, 0f, 1f)
            }
        }
    }

    private class EvidenceChat : ChatPort {
        override fun answer(selection: ModelSelection, question: String, context: List<ScoredChunk>): String = when {
            context.isEmpty() -> "Baseline has no collection evidence."
            context.any { "ORBIT_EVIDENCE" in it.chunk.draft.text } -> "The indexed orbit fragment supports the route."
            else -> "The retrieved collection has no orbit evidence."
        }
    }
}
