package dev.localrag.domain

import kotlinx.serialization.Serializable

@Serializable
enum class SourceType {
    PDF,
    TEXT,
    MARKDOWN,
    HTML,
    CODE,
}

@Serializable
enum class SourceStatus {
    PENDING,
    INDEXING,
    READY,
    FAILED,
    UNSUPPORTED,
    DELETING,
}

@Serializable
data class SourceRecord(
    val sourceId: String,
    val name: String,
    val type: SourceType,
    val sizeBytes: Long,
    val status: SourceStatus,
    val error: String? = null,
    val createdAt: String,
    val unsearchablePages: List<Int> = emptyList(),
)

@Serializable
data class SourceLocation(
    val pageStart: Int? = null,
    val pageEnd: Int? = null,
    val lineStart: Int? = null,
    val lineEnd: Int? = null,
) {
    init {
        val hasPage = pageStart != null || pageEnd != null
        val hasLine = lineStart != null || lineEnd != null
        require(!(hasPage && hasLine)) { "A source location cannot mix page and line ranges." }
        require(!hasPage || (pageStart != null && pageEnd != null && pageStart >= 1 && pageEnd >= pageStart)) {
            "A page location must be a valid one-based range."
        }
        require(!hasLine || (lineStart != null && lineEnd != null && lineStart >= 1 && lineEnd >= lineStart)) {
            "A line location must be a valid one-based range."
        }
    }

    val isDefined: Boolean get() = pageStart != null || lineStart != null
}

@Serializable
data class SourceSegment(
    val text: String,
    val section: String,
    val location: SourceLocation = SourceLocation(),
)

@Serializable
enum class ChunkStrategy {
    FIXED_SIZE,
    STRUCTURAL,
}

@Serializable
data class ChunkDraft(
    val strategy: ChunkStrategy,
    val chunkId: String,
    val sourceId: String,
    val sourceName: String,
    val section: String,
    val location: SourceLocation,
    val tokenUnits: Int,
    val text: String,
)

@Serializable
data class StoredChunk(
    val draft: ChunkDraft,
    val embedding: List<Float>,
    val embeddingModel: String,
)

@Serializable
data class ScoredChunk(
    val chunk: StoredChunk,
    val cosineScore: Double,
)

@Serializable
data class SourceCitation(
    val sourceId: String,
    val source: String,
    val section: String,
    val chunkId: String,
    val location: SourceLocation,
    val score: Double,
    val quote: String = "",
)

interface EmbeddingPort {
    val modelName: String
    fun embed(texts: List<String>): List<List<Float>>
}

@Serializable
data class ModelSelection(
    val providerId: String,
    val modelId: String,
)

interface ChatPort {
    fun answer(selection: ModelSelection, question: String, context: List<ScoredChunk> = emptyList()): String
}

interface RerankPort {
    fun rerank(selection: ModelSelection, question: String, candidateTexts: List<String>): List<Int>
}

data class StoredSource(
    val record: SourceRecord,
    val storageKey: String?,
    val contentSha256: String? = null,
)

data class IndexProgressUpdate(
    val phase: String,
    val filesDone: Int,
    val filesTotal: Int,
    val bytesDone: Long,
    val bytesTotal: Long,
    val currentSource: String? = null,
    val succeeded: Int = 0,
    val failed: Int = 0,
    val currentPosition: String? = null,
)

data class IndexOutcome(
    val indexedSourceIds: List<String>,
    val failedSources: List<SourceRecord>,
)
