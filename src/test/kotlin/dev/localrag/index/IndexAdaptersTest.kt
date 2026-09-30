package dev.localrag.index

import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.SourceSegment
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.jupiter.api.io.TempDir

class IndexAdaptersTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `PDF extraction preserves text page numbers and marks pages without text without OCR`() {
        val source = source("manual.pdf", SourceType.PDF)
        val pdf = temporaryDirectory.resolve("fixture.pdf")
        Files.writeString(pdf, "%PDF-1.7", StandardCharsets.US_ASCII)
        val extractor = PopplerPdfSourceExtractor(
            processRunner = FixturePdfProcessRunner("First page text\u000c\u000cThird page text\u000c"),
            temporaryDirectory = temporaryDirectory,
            timeout = Duration.ofSeconds(2),
        )
        val segments = mutableListOf<SourceSegment>()

        val summary = extractor.extract(source, pdf, segments::add)

        assertEquals(3, summary.pagesTotal)
        assertEquals(listOf(2), summary.unsearchablePages)
        assertEquals(listOf(1, 3), segments.map { it.location.pageStart })
        assertTrue(segments.all { it.location.pageStart == it.location.pageEnd })
    }

    @Test
    fun `PDF with no text layer is rejected instead of receiving synthetic OCR text`() {
        val source = source("scan.pdf", SourceType.PDF)
        val pdf = temporaryDirectory.resolve("scan.pdf")
        Files.writeString(pdf, "%PDF-1.7", StandardCharsets.US_ASCII)
        val extractor = PopplerPdfSourceExtractor(
            processRunner = FixturePdfProcessRunner("\u000c\u000c", pageCount = 2),
            temporaryDirectory = temporaryDirectory,
            timeout = Duration.ofSeconds(2),
        )

        val error = assertFailsWith<SourceExtractionException> {
            extractor.extract(source, pdf) { error("A blank scanned page must not be indexed.") }
        }

        assertTrue(error.message.orEmpty().contains("OCR не выполняется"))
        assertEquals(listOf(1, 2), error.unsearchablePages)
    }

    @Test
    fun `HTML extraction keeps heading provenance and discards script and style text`() {
        val source = source("page.html", SourceType.HTML)
        val html = temporaryDirectory.resolve("page.html")
        Files.writeString(
            html,
            "<html><h1>Overview</h1><p>Orbit markers describe the route.</p><script>SECRET_EXECUTION_MARKER</script><style>.SECRET_STYLE_MARKER { color: red; }</style></html>",
        )
        val segments = mutableListOf<SourceSegment>()

        StaticHtmlSourceExtractor().extract(source, html, segments::add)

        assertTrue(segments.any { it.section == "Overview" && it.text.contains("Orbit markers") })
        assertFalse(segments.joinToString(" ") { it.text }.contains("SECRET_EXECUTION_MARKER"), segments.toString())
        assertFalse(segments.joinToString(" ") { it.text }.contains("SECRET_STYLE_MARKER"), segments.toString())
        assertTrue(segments.all { !it.location.isDefined })
    }

    @Test
    fun `plain text keeps original line locations while streaming an oversized line in bounded segments`() {
        val source = source("source.kt", SourceType.CODE)
        val text = temporaryDirectory.resolve("source.kt")
        Files.writeString(text, "class Navigator\n" + "x".repeat(150_000) + " end\n")
        val segments = mutableListOf<SourceSegment>()

        PlainTextSourceExtractor().extract(source, text, segments::add)

        assertEquals("class Navigator", segments.first().section)
        val oversizedLine = segments.drop(1)
        assertTrue(oversizedLine.size > 1)
        assertTrue(oversizedLine.all { it.location.lineStart == 2 && it.location.lineEnd == 2 })
        assertTrue(oversizedLine.all { it.text.length <= 64 * 1024 })
    }

    @Test
    fun `fixed chunker uses 500 whitespace units and retains exactly fifty units between windows`() {
        val source = source("guide.md", SourceType.MARKDOWN)
        val text = (1..501).joinToString(" ") { "word$it" }
        val chunks = mutableListOf<ChunkDraft>()
        val accumulator = FixedSizeChunker().accumulator(source, chunks::add)
        accumulator.accept(SourceSegment(text, "Overview", SourceLocation(lineStart = 1, lineEnd = 1)))
        accumulator.finish()

        assertEquals(listOf(500, 51), chunks.map(ChunkDraft::tokenUnits))
        assertEquals("word1", chunks.first().text.substringBefore(' '))
        assertEquals("word451", chunks.last().text.substringBefore(' '))
        assertEquals("word501", chunks.last().text.substringAfterLast(' '))
    }

    @Test
    fun `structural chunks flush at section boundaries and keep unique IDs`() {
        val source = source("guide.md", SourceType.MARKDOWN)
        val chunks = mutableListOf<ChunkDraft>()
        val accumulator = StructuralChunker(maxTokenUnits = 3, overlapTokenUnits = 1).accumulator(source, chunks::add)
        accumulator.accept(SourceSegment("one two three", "First section", SourceLocation(lineStart = 1, lineEnd = 1)))
        accumulator.accept(SourceSegment("four five", "Second section", SourceLocation(lineStart = 8, lineEnd = 8)))
        accumulator.finish()

        assertEquals(listOf("one two three", "four five"), chunks.map(ChunkDraft::text))
        assertEquals(listOf("First section", "Second section"), chunks.map(ChunkDraft::section))
        assertEquals(2, chunks.map(ChunkDraft::chunkId).toSet().size)
        assertEquals(listOf(1, 8), chunks.map { it.location.lineStart })
        assertTrue(chunks.all { it.strategy == ChunkStrategy.STRUCTURAL })
    }

    private fun source(name: String, type: SourceType) = SourceRecord(
        sourceId = UUID.randomUUID().toString(),
        name = name,
        type = type,
        sizeBytes = 1,
        status = SourceStatus.PENDING,
        createdAt = "2026-09-29T00:00:00Z",
    )

    private class FixturePdfProcessRunner(
        private val extractedText: String,
        private val pageCount: Int = 3,
    ) : PdfProcessRunner {
        override fun run(command: List<String>, timeout: Duration, stdout: java.io.OutputStream): ProcessResult {
            val output = if (command.first() == "pdfinfo") "Pages: $pageCount\n" else extractedText
            stdout.write(output.toByteArray(StandardCharsets.UTF_8))
            return ProcessResult(0)
        }
    }
}
