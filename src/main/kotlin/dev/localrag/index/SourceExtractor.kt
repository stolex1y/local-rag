package dev.localrag.index

import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceSegment
import dev.localrag.domain.SourceType
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.swing.text.MutableAttributeSet
import javax.swing.text.html.HTML
import javax.swing.text.html.HTMLEditorKit
import javax.swing.text.html.parser.ParserDelegator
import kotlin.concurrent.thread

data class ExtractionSummary(
    val segments: Long,
    val pagesTotal: Int? = null,
    val unsearchablePages: List<Int> = emptyList(),
)

interface SourceExtractor {
    fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary
}

class SourceExtractorRegistry(
    private val pdf: SourceExtractor = PopplerPdfSourceExtractor(),
    private val text: SourceExtractor = PlainTextSourceExtractor(),
    private val html: SourceExtractor = StaticHtmlSourceExtractor(),
) : SourceExtractor {
    override fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary =
        when (source.type) {
            SourceType.PDF -> pdf.extract(source, storedFile, emit)
            SourceType.HTML -> html.extract(source, storedFile, emit)
            SourceType.TEXT, SourceType.MARKDOWN, SourceType.CODE -> text.extract(source, storedFile, emit)
        }
}

class SourceExtractionException(
    message: String,
    cause: Throwable? = null,
    val unsearchablePages: List<Int> = emptyList(),
) : IllegalStateException(message, cause)

internal data class ProcessResult(val exitCode: Int)

internal fun interface PdfProcessRunner {
    fun run(command: List<String>, timeout: Duration, stdout: OutputStream): ProcessResult
}

class PopplerPdfSourceExtractor internal constructor(
    private val processRunner: PdfProcessRunner,
    private val temporaryDirectory: Path,
    private val timeout: Duration = DEFAULT_TIMEOUT,
) : SourceExtractor {
    constructor() : this(SystemPdfProcessRunner(), Path.of(System.getProperty("java.io.tmpdir")), DEFAULT_TIMEOUT)

    init {
        require(!timeout.isNegative && !timeout.isZero) { "PDF subprocess timeout must be positive." }
    }

    override fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary {
        val info = BoundedByteArrayOutputStream(MAX_PDFINFO_BYTES)
        val infoResult = runCommand(listOf("pdfinfo", storedFile.toString()), info)
        if (infoResult.exitCode != 0) throw SourceExtractionException("Не удалось прочитать метаданные PDF (код ${infoResult.exitCode}).")
        val pageCount = PAGE_COUNT.find(info.toString(StandardCharsets.UTF_8))?.groupValues?.get(1)?.toIntOrNull()
            ?: throw SourceExtractionException("PDF не сообщает корректное число страниц.")
        if (pageCount < 1) throw SourceExtractionException("В PDF нет страниц для извлечения текста.")

        val extracted = Files.createTempFile(temporaryDirectory, "local-rag-pdf-", ".txt")
        try {
            Files.newOutputStream(extracted).use { output ->
                val result = runCommand(
                    listOf("pdftotext", "-layout", "-enc", "UTF-8", storedFile.toString(), "-"),
                    output,
                )
                if (result.exitCode != 0) throw SourceExtractionException("Не удалось извлечь текст PDF (код ${result.exitCode}).")
            }
            return readPdfPages(extracted, pageCount, emit)
        } finally {
            Files.deleteIfExists(extracted)
        }
    }

    private fun readPdfPages(path: Path, expectedPages: Int, emit: (SourceSegment) -> Unit): ExtractionSummary {
        var pageNumber = 1
        var pageHasText = false
        var endedAtFormFeed = false
        var section = PDF_BODY
        var segments = 0L
        val unsearchablePages = mutableListOf<Int>()
        val line = StringBuilder()
        fun flushLine() {
            val value = line.toString().trim()
            line.setLength(0)
            if (value.isNotEmpty()) {
                val heading = heading(value)
                if (heading != null) section = heading
                emit(SourceSegment(value, section, SourceLocation(pageStart = pageNumber, pageEnd = pageNumber)))
                pageHasText = true
                segments++
            }
        }
        fun finishPage() {
            if (!pageHasText) {
                unsearchablePages += pageNumber
            }
            pageHasText = false
        }
        Files.newBufferedReader(path, StandardCharsets.UTF_8).use { reader ->
            var skipLineFeed = false
            while (true) {
                val code = reader.read()
                if (code < 0) break
                val char = code.toChar()
                if (skipLineFeed) {
                    skipLineFeed = false
                    if (char == '\n') continue
                }
                when (char) {
                    '\u000c' -> {
                        flushLine()
                        finishPage()
                        pageNumber++
                        endedAtFormFeed = true
                    }
                    '\n' -> flushLine()
                    '\r' -> {
                        flushLine()
                        skipLineFeed = true
                    }
                    else -> {
                        line.append(char)
                        if (!char.isWhitespace()) endedAtFormFeed = false
                        if (line.length > MAX_LINE_CHARS) throw SourceExtractionException("Строка PDF превышает безопасный размер.")
                    }
                }
            }
        }
        if (!endedAtFormFeed) {
            flushLine()
            finishPage()
        }
        val pagesRead = pageNumber - if (endedAtFormFeed) 1 else 0
        if (pagesRead != expectedPages) {
            throw SourceExtractionException(
                "Извлечено страниц: $pagesRead; метаданные PDF указывают $expectedPages.",
                unsearchablePages = unsearchablePages.toList(),
            )
        }
        if (segments == 0L) {
            throw SourceExtractionException(
                "В PDF не найден текстовый слой; OCR не выполняется.",
                unsearchablePages = unsearchablePages.toList(),
            )
        }
        return ExtractionSummary(segments, expectedPages, unsearchablePages)
    }

    private fun runCommand(command: List<String>, stdout: OutputStream): ProcessResult = try {
        processRunner.run(command, timeout, stdout)
    } catch (error: IOException) {
        throw SourceExtractionException("Не удалось запустить Poppler для PDF.", error)
    } catch (error: InterruptedException) {
        Thread.currentThread().interrupt()
        throw SourceExtractionException("Извлечение PDF было прервано.", error)
    }

    companion object {
        private const val MAX_LINE_CHARS = 2 * 1024 * 1024
        private const val MAX_PDFINFO_BYTES = 64 * 1024
        private val DEFAULT_TIMEOUT = Duration.ofMinutes(2)
        private const val PDF_BODY = "PDF document body"
        private val PAGE_COUNT = Regex("(?m)^Pages:\\s*(\\d+)\\s*$")
        private val NUMBERED_HEADING = Regex("^\\s*(?:(?:\\d+(?:\\.\\d+)*(?:[.)])?|[A-Z]\\.\\d+(?:\\.\\d+)*(?:[.)])?)\\s+)\\S.{1,180}$")
        private val UPPERCASE_HEADING = Regex("^[A-Z0-9][A-Z0-9 /_():.-]{4,120}$")

        private fun heading(line: String): String? {
            if (line.length !in 5..180) return null
            if (NUMBERED_HEADING.matches(line)) return line
            val letters = line.filter(Char::isLetter)
            return line.takeIf { letters.length >= 5 && letters == letters.uppercase() && UPPERCASE_HEADING.matches(it) }
        }
    }
}

class PlainTextSourceExtractor : SourceExtractor {
    override fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary {
        var section = BODY_SECTION
        var lineNumber = 1
        var firstSegmentOnLine = true
        var segments = 0L
        val buffer = StringBuilder(MAX_SEGMENT_CHARS)
        var lastWhitespace = -1
        fun emitBuffer(length: Int) {
            val value = buffer.substring(0, length).trim()
            val remainder = buffer.substring(length)
            buffer.setLength(0)
            buffer.append(remainder)
            lastWhitespace = -1
            for (index in 0 until buffer.length) if (buffer[index].isWhitespace()) lastWhitespace = index
            if (value.isNotEmpty()) {
                if (firstSegmentOnLine) {
                    val heading = when (source.type) {
                        SourceType.MARKDOWN -> MARKDOWN_HEADING.find(value)?.groupValues?.get(1)?.trim()
                        SourceType.CODE -> CODE_HEADING.find(value)?.value?.trim()
                        else -> TEXT_HEADING.find(value)?.value?.trim()
                    }
                    if (!heading.isNullOrBlank()) section = heading.take(MAX_SECTION_CHARS)
                    firstSegmentOnLine = false
                }
                emit(SourceSegment(value, section, SourceLocation(lineStart = lineNumber, lineEnd = lineNumber)))
                segments++
            }
        }
        fun finishLine() {
            if (buffer.isNotEmpty()) emitBuffer(buffer.length)
            firstSegmentOnLine = true
            lineNumber++
        }
        try {
            Files.newBufferedReader(storedFile, StandardCharsets.UTF_8).use { reader ->
                var skipLineFeed = false
                while (true) {
                    val code = reader.read()
                    if (code < 0) break
                    val char = code.toChar()
                    if (skipLineFeed) {
                        skipLineFeed = false
                        if (char == '\n') continue
                    }
                    when (char) {
                        '\r' -> {
                            finishLine()
                            skipLineFeed = true
                        }
                        '\n' -> finishLine()
                        else -> {
                            buffer.append(char)
                            if (char.isWhitespace()) lastWhitespace = buffer.lastIndex
                            if (buffer.length >= MAX_SEGMENT_CHARS) {
                                val boundary = if (lastWhitespace > 0) lastWhitespace + 1 else buffer.length
                                emitBuffer(boundary)
                            }
                        }
                    }
                }
            }
        } catch (error: IOException) {
            throw SourceExtractionException("Не удалось прочитать UTF-8 текстовый источник.", error)
        }
        if (buffer.isNotEmpty()) emitBuffer(buffer.length)
        if (segments == 0L) throw SourceExtractionException("В файле не найден непустой текст.")
        return ExtractionSummary(segments)
    }

    companion object {
        private const val BODY_SECTION = "Document body"
        private const val MAX_SECTION_CHARS = 160
        private const val MAX_SEGMENT_CHARS = 64 * 1024
        private val MARKDOWN_HEADING = Regex("^#{1,6}\\s+(.+?)\\s*#*\\s*$")
        private val CODE_HEADING = Regex("^(?:(?:export\\s+)?(?:async\\s+)?(?:function|class|interface|type|enum|object|struct|trait|def|fun)\\s+[A-Za-z_$][\\w$]*|(?:public|private|protected|internal)?\\s*(?:static\\s+)?(?:class|interface|enum|record)\\s+[A-Za-z_$][\\w$]*).*?$")
        private val TEXT_HEADING = Regex("^(?:\\d+(?:\\.\\d+)*[.)]?\\s+)?[A-Z][^.!?]{2,119}$")
    }
}


class StaticHtmlSourceExtractor : SourceExtractor {
    override fun extract(source: SourceRecord, storedFile: Path, emit: (SourceSegment) -> Unit): ExtractionSummary {
        var ignoredTag: HTML.Tag? = null
        var headingDepth = 0
        var headingText = StringBuilder()
        var section = HTML_BODY
        var segments = 0L
        val callback = object : HTMLEditorKit.ParserCallback() {
            override fun handleStartTag(tag: HTML.Tag, attributes: MutableAttributeSet, position: Int) {
                if (tag == HTML.Tag.SCRIPT || tag == HTML.Tag.STYLE) {
                    ignoredTag = tag
                    return
                }
                if (isHeading(tag)) {
                    headingDepth++
                    headingText = StringBuilder()
                }
            }

            override fun handleEndTag(tag: HTML.Tag, position: Int) {
                if (ignoredTag == tag) {
                    ignoredTag = null
                    return
                }
                if (isHeading(tag) && headingDepth > 0) {
                    headingDepth--
                    headingText.toString().trim().takeIf(String::isNotBlank)?.let { section = it.take(MAX_SECTION_CHARS) }
                }
            }
            override fun handleSimpleTag(tag: HTML.Tag, attributes: MutableAttributeSet, position: Int) {
                if (tag == HTML.Tag.SCRIPT || tag == HTML.Tag.STYLE) ignoredTag = tag
            }

            override fun handleText(data: CharArray, position: Int) {
                if (ignoredTag != null) return
                val value = data.concatToString().replace(WHITESPACE, " ").trim()
                if (value.isEmpty()) return
                if (headingDepth > 0) {
                    if (headingText.isNotEmpty()) headingText.append(' ')
                    headingText.append(value)
                } else {
                    emit(SourceSegment(value, section))
                    segments++
                }
            }

            private fun isHeading(tag: HTML.Tag): Boolean = tag in HEADING_TAGS
        }
        try {
            Files.newBufferedReader(storedFile, StandardCharsets.UTF_8).use { reader ->
                ActiveHtmlReader(reader).use { safeReader ->
                    ParserDelegator().parse(safeReader, callback, true)
                }
            }
        } catch (error: IOException) {
            throw SourceExtractionException("Не удалось разобрать статический HTML-файл.", error)
        }
        if (segments == 0L) throw SourceExtractionException("В HTML не найден текст вне script/style.")
        return ExtractionSummary(segments)
    }

    companion object {
        private const val HTML_BODY = "HTML body"
        private const val MAX_SECTION_CHARS = 160
        private val WHITESPACE = Regex("\\s+")
        private val HEADING_TAGS = setOf(HTML.Tag.H1, HTML.Tag.H2, HTML.Tag.H3, HTML.Tag.H4, HTML.Tag.H5, HTML.Tag.H6)
    }
}

private class ActiveHtmlReader(private val input: java.io.Reader) : java.io.Reader() {
    private val output = java.util.ArrayDeque<Char>()
    private var ignoredTag: String? = null

    override fun read(): Int {
        while (output.isEmpty()) {
            if (ignoredTag != null) {
                var code = input.read()
                while (code >= 0 && code != '<'.code) code = input.read()
                if (code < 0) return -1
                val tag = readTag()
                if (tag != null && closesIgnoredTag(tag, ignoredTag!!)) ignoredTag = null
            } else {
                val code = input.read()
                if (code < 0) return -1
                if (code != '<'.code) return code
                val tag = readTag()
                if (tag == null) {
                    output.addLast('<')
                } else {
                    val activeTag = ACTIVE_TAG_START.find(tag)?.groupValues?.get(1)?.lowercase()
                    if (activeTag == null) {
                        tag.forEach(output::addLast)
                    } else {
                        ignoredTag = activeTag
                    }
                }
            }
        }
        return output.removeFirst().code
    }

    override fun read(buffer: CharArray, offset: Int, length: Int): Int {
        require(offset >= 0 && length >= 0 && length <= buffer.size - offset)
        if (length == 0) return 0
        var count = 0
        while (count < length) {
            val code = read()
            if (code < 0) return if (count == 0) -1 else count
            buffer[offset + count++] = code.toChar()
        }
        return count
    }

    override fun close() = input.close()

    private fun readTag(): String? {
        val tag = StringBuilder().append('<')
        var quote: Char? = null
        while (true) {
            val code = input.read()
            if (code < 0) return null
            val char = code.toChar()
            tag.append(char)
            when {
                quote == char -> quote = null
                quote == null && char in setOf('\'', '"') -> quote = char
                quote == null && char == '>' -> return tag.toString()
            }
            if (tag.length > MAX_HTML_TAG_CHARS) return null
        }
    }

    private fun closesIgnoredTag(tag: String, ignored: String): Boolean =
        CLOSE_TAG.find(tag)?.groupValues?.get(1)?.equals(ignored, ignoreCase = true) == true

    companion object {
        private const val MAX_HTML_TAG_CHARS = 16 * 1024
        private val ACTIVE_TAG_START = Regex("(?is)^<\\s*(script|style)\\b[^>]*>$")
        private val CLOSE_TAG = Regex("(?is)^<\\s*/\\s*(script|style)\\s*>$")
    }
}
private class SystemPdfProcessRunner : PdfProcessRunner {
    override fun run(command: List<String>, timeout: Duration, stdout: OutputStream): ProcessResult {
        val process = ProcessBuilder(command).start()
        val stdoutFailure = AtomicReference<IOException?>()
        val stdoutThread = thread(name = "local-rag-pdf-stdout", isDaemon = true) {
            try {
                process.inputStream.use { input ->
                    val buffer = ByteArray(PROCESS_BUFFER_BYTES)
                    var bytes = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        bytes += count
                        if (bytes > MAX_PDF_TEXT_BYTES) throw IOException("PDF text output exceeded the configured limit.")
                        stdout.write(buffer, 0, count)
                    }
                }
            } catch (error: IOException) {
                stdoutFailure.set(error)
                process.destroyForcibly()
            }
        }
        val stderrThread = thread(name = "local-rag-pdf-stderr", isDaemon = true) {
            process.errorStream.use { input ->
                val buffer = ByteArray(PROCESS_BUFFER_BYTES)
                while (input.read(buffer) >= 0) Unit
            }
        }
        val completed = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS)
        if (!completed) {
            process.destroyForcibly()
            process.waitFor()
        }
        stdoutThread.join(PROCESS_THREAD_JOIN_MILLIS)
        stderrThread.join(PROCESS_THREAD_JOIN_MILLIS)
        if (stdoutThread.isAlive || stderrThread.isAlive) {
            process.destroyForcibly()
            throw IOException("PDF subprocess did not close its output streams.")
        }
        if (!completed) throw IOException("PDF subprocess timed out.")
        stdoutFailure.get()?.let { throw it }
        return ProcessResult(process.exitValue())
    }

    companion object {
        private const val PROCESS_BUFFER_BYTES = 32 * 1024
        private const val PROCESS_THREAD_JOIN_MILLIS = 2_000L
        private const val MAX_PDF_TEXT_BYTES = 512L * 1024 * 1024
    }
}

private class BoundedByteArrayOutputStream(private val maximum: Int) : ByteArrayOutputStream() {
    override fun write(value: Int) {
        if (count >= maximum) throw IOException("Process output exceeded the configured limit.")
        super.write(value)
    }

    override fun write(buffer: ByteArray, offset: Int, length: Int) {
        if (length > maximum - count) throw IOException("Process output exceeded the configured limit.")
        super.write(buffer, offset, length)
    }
}
