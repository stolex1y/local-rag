package dev.localrag.domain

interface IndexRepository : AutoCloseable {
    fun registerSource(source: StoredSource)
    fun source(sourceId: String): SourceRecord?
    fun storedSource(sourceId: String): StoredSource?
    fun storedSources(): List<StoredSource>
    fun sourceByContentSha256(hash: String): StoredSource?
    fun updateSourceContentSha256(sourceId: String, hash: String)
    fun unhashedSourcesOfSize(sizeBytes: Long): List<StoredSource>
    fun sources(): List<SourceRecord>
    fun sources(status: SourceStatus): List<StoredSource>
    fun pendingSources(): List<StoredSource>
    fun hasSourceLocation(sourceId: String, location: SourceLocation, section: String?): Boolean
    fun markSourceStatus(
        sourceId: String,
        status: SourceStatus,
        error: String? = null,
        unsearchablePages: List<Int> = emptyList(),
    )
    fun clearChunks(sourceId: String)
    fun saveChunks(sourceId: String, chunks: List<StoredChunk>)
    fun search(
        strategy: ChunkStrategy,
        queryEmbedding: List<Float>,
        embeddingModel: String,
        limit: Int,
    ): List<ScoredChunk>
    fun chunkCount(sourceId: String, strategy: ChunkStrategy): Int
    fun deleteSource(sourceId: String)
    fun recoverInterruptedIndexes()
}
