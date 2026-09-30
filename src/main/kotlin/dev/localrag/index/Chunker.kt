package dev.localrag.index

import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceSegment
import java.util.ArrayDeque

interface Chunker {
    val strategy: ChunkStrategy
    fun accumulator(source: SourceRecord, emit: (ChunkDraft) -> Unit): ChunkAccumulator
}

interface ChunkAccumulator {
    fun accept(segment: SourceSegment)
    fun finish()
}

class FixedSizeChunker(
    private val targetTokenUnits: Int = DEFAULT_TARGET,
    private val overlapTokenUnits: Int = DEFAULT_OVERLAP,
) : Chunker {
    init {
        require(targetTokenUnits > 0 && overlapTokenUnits in 0 until targetTokenUnits)
    }

    override val strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE

    override fun accumulator(source: SourceRecord, emit: (ChunkDraft) -> Unit): ChunkAccumulator =
        WindowAccumulator(source, strategy, targetTokenUnits, overlapTokenUnits, emit)

    companion object {
        const val DEFAULT_TARGET = 500
        const val DEFAULT_OVERLAP = 50
    }
}

class StructuralChunker(
    private val maxTokenUnits: Int = FixedSizeChunker.DEFAULT_TARGET,
    private val overlapTokenUnits: Int = FixedSizeChunker.DEFAULT_OVERLAP,
) : Chunker {
    init {
        require(maxTokenUnits > 0 && overlapTokenUnits in 0 until maxTokenUnits)
    }

    override val strategy: ChunkStrategy = ChunkStrategy.STRUCTURAL

    override fun accumulator(source: SourceRecord, emit: (ChunkDraft) -> Unit): ChunkAccumulator =
        object : ChunkAccumulator {
            private var currentSection: String? = null
            private val window = WindowAccumulator(source, strategy, maxTokenUnits, overlapTokenUnits, emit)

            override fun accept(segment: SourceSegment) {
                val section = segment.section.trim().ifEmpty { BODY_SECTION }
                if (section != currentSection) {
                    window.finishSection()
                    currentSection = section
                }
                window.accept(segment.copy(section = section))
            }

            override fun finish() {
                window.finish()
                currentSection = null
            }
        }

    companion object {
        const val BODY_SECTION = "Document body"
    }
}

private class WindowAccumulator(
    private val source: SourceRecord,
    private val strategy: ChunkStrategy,
    private val target: Int,
    private val overlap: Int,
    private val emit: (ChunkDraft) -> Unit,
) : ChunkAccumulator {
    private val window = ArrayDeque<PageToken>(target)
    private var chunkNumber = 0
    private var chunksInSection = 0
    private var finished = false

    override fun accept(segment: SourceSegment) {
        check(!finished) { "Chunk accumulator is already finished." }
        val section = segment.section.trim().ifEmpty { StructuralChunker.BODY_SECTION }
        TOKEN.findAll(segment.text).forEach { match ->
            window.addLast(PageToken(match.value, section, segment.location))
            if (window.size == target) emitWindow()
        }
    }

    override fun finish() {
        if (finished) return
        flushSection()
        finished = true
    }

    fun finishSection() {
        check(!finished) { "Chunk accumulator is already finished." }
        flushSection()
    }

    private fun flushSection() {
        if (window.isNotEmpty() && (chunksInSection == 0 || window.size > overlap)) emitWindow()
        window.clear()
        chunksInSection = 0
    }

    private fun emitWindow() {
        val tokens = ArrayList<PageToken>(minOf(window.size, target))
        repeat(minOf(window.size, target)) { tokens.add(window.removeFirst()) }
        val first = tokens.first()
        val last = tokens.last()
        val sections = LinkedHashSet<String>()
        tokens.forEach { token -> if (sections.size < MAX_SECTIONS_PER_CHUNK) sections += token.section }
        val chunkId = "${source.sourceId}-${strategy.name.lowercase()}-${(++chunkNumber).toString().padStart(CHUNK_ID_WIDTH, '0')}"
        emit(
            ChunkDraft(
                strategy = strategy,
                chunkId = chunkId,
                sourceId = source.sourceId,
                sourceName = source.name,
                section = sections.joinToString(SECTION_SEPARATOR),
                location = mergeLocation(first.location, last.location),
                tokenUnits = tokens.size,
                text = tokens.joinToString(" ", transform = PageToken::text),
            ),
        )
        chunksInSection++
        repeat(minOf(overlap, tokens.size)) { index -> window.addFirst(tokens[tokens.lastIndex - index]) }
    }

    private fun mergeLocation(first: SourceLocation, last: SourceLocation): SourceLocation = when {
        first.pageStart != null && last.pageEnd != null -> SourceLocation(first.pageStart, last.pageEnd)
        first.lineStart != null && last.lineEnd != null -> SourceLocation(lineStart = first.lineStart, lineEnd = last.lineEnd)
        else -> SourceLocation()
    }

    private data class PageToken(val text: String, val section: String, val location: SourceLocation)

    companion object {
        private const val CHUNK_ID_WIDTH = 6
        private const val MAX_SECTIONS_PER_CHUNK = 8
        private const val SECTION_SEPARATOR = " · "
        private val TOKEN = Regex("\\S+")
    }
}
