package dev.localrag.index

import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.EmbeddingPort
import dev.localrag.domain.IndexOutcome
import dev.localrag.domain.IndexProgressUpdate
import dev.localrag.domain.IndexRepository
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredChunk
import dev.localrag.domain.StoredSource
import dev.localrag.ollama.LocalModelException
import java.nio.file.Files
import java.nio.file.Path

class IndexWorkflow(
    private val extractor: SourceExtractor,
    chunkers: List<Chunker>,
    private val embeddings: EmbeddingPort,
    private val index: IndexRepository,
    private val sourcesDirectory: Path,
) {
    private val chunkersByStrategy = chunkers.associateBy(Chunker::strategy)

    init {
        require(chunkersByStrategy.keys == ChunkStrategy.entries.toSet()) {
            "Обе стратегии — fixed-size и structural — обязательны."
        }
    }

    fun index(sources: List<StoredSource>, progress: (IndexProgressUpdate) -> Unit): IndexOutcome {
        require(sources.isNotEmpty()) { "Нет источников для индексации." }
        require(sources.all { it.record.status == SourceStatus.PENDING && it.storageKey != null }) {
            "Индексировать можно только сохранённые источники в состоянии PENDING."
        }
        val totalBytes = sources.fold(0L) { total, source ->
            require(source.record.sizeBytes <= Long.MAX_VALUE - total) { "Общий размер коллекции слишком велик." }
            total + source.record.sizeBytes
        }
        val indexed = mutableListOf<String>()
        val failed = mutableListOf<SourceRecord>()
        var completedFiles = 0
        var completedBytes = 0L
        sources.forEach { stored ->
            val record = stored.record
            val initialPosition = if (record.type == SourceType.PDF) "подготовка PDF" else null
            progress(IndexProgressUpdate("extracting", completedFiles, sources.size, completedBytes, totalBytes, record.name, indexed.size, failed.size, initialPosition))
            val storedPath = stored.storageKey?.let(sourcesDirectory::resolve)?.normalize()
            try {
                require(storedPath != null && storedPath.parent == sourcesDirectory.normalize()) { "Storage key is invalid." }
                require(Files.isRegularFile(storedPath)) { "Локальная копия источника не найдена." }
                indexOne(stored, storedPath, progress, completedFiles, sources.size, completedBytes, totalBytes, indexed.size, failed.size)
                indexed += record.sourceId
            } catch (error: Exception) {
                when (index.source(record.sourceId)?.status) {
                    SourceStatus.PENDING, SourceStatus.INDEXING -> {
                        index.markSourceStatus(
                            record.sourceId,
                            SourceStatus.FAILED,
                            safeSourceError(error),
                            (error as? SourceExtractionException)?.unsearchablePages.orEmpty(),
                        )
                        index.source(record.sourceId)?.let(failed::add)
                    }
                    else -> Unit
                }
            } finally {
                completedBytes += record.sizeBytes
                completedFiles++
                progress(IndexProgressUpdate("source-complete", completedFiles, sources.size, completedBytes, totalBytes, null, indexed.size, failed.size))
            }
        }
        return IndexOutcome(indexed, failed)
    }

    private fun indexOne(
        source: StoredSource,
        path: Path,
        progress: (IndexProgressUpdate) -> Unit,
        filesDone: Int,
        filesTotal: Int,
        bytesDone: Long,
        bytesTotal: Long,
        succeeded: Int,
        failed: Int,
    ) {
        val record = source.record
        index.markSourceStatus(record.sourceId, SourceStatus.INDEXING)
        index.clearChunks(record.sourceId)
        val pending = ChunkStrategy.entries.associateWith { mutableListOf<ChunkDraft>() }.toMutableMap()
        val written = ChunkStrategy.entries.associateWith { 0 }.toMutableMap()
        var currentPosition: String? = if (record.type == SourceType.PDF) "подготовка PDF" else null
        val accumulators = chunkersByStrategy.mapValues { (strategy, chunker) ->
            chunker.accumulator(record) { draft ->
                val batch = pending.getValue(strategy)
                batch += draft
                if (batch.size >= EMBEDDING_BATCH_SIZE) {
                    persistBatch(record.sourceId, batch)
                    written[strategy] = written.getValue(strategy) + batch.size
                    batch.clear()
                    progress(IndexProgressUpdate("embedding", filesDone, filesTotal, bytesDone, bytesTotal, record.name, succeeded, failed, currentPosition))
                }
            }
        }
        val summary = extractor.extract(record, path) { segment ->
            val page = segment.location.pageEnd?.let { "страница $it" }
            if (page != null && page != currentPosition) {
                currentPosition = page
                progress(
                    IndexProgressUpdate(
                        "extracting",
                        filesDone,
                        filesTotal,
                        bytesDone,
                        bytesTotal,
                        record.name,
                        succeeded,
                        failed,
                        currentPosition,
                    ),
                )
            }
            accumulators.values.forEach { accumulator -> accumulator.accept(segment) }
        }
        accumulators.values.forEach(ChunkAccumulator::finish)
        ChunkStrategy.entries.forEach { strategy ->
            val batch = pending.getValue(strategy)
            if (batch.isNotEmpty()) {
                persistBatch(record.sourceId, batch)
                written[strategy] = written.getValue(strategy) + batch.size
                batch.clear()
            }
            check(written.getValue(strategy) > 0) { "Стратегия $strategy не создала индексируемых фрагментов." }
        }
        index.markSourceStatus(record.sourceId, SourceStatus.READY, unsearchablePages = summary.unsearchablePages)
    }

    private fun persistBatch(sourceId: String, drafts: List<ChunkDraft>) {
        val inputs = drafts.map { draft -> "title: ${draft.sourceName}\nsection: ${draft.section}\ntext: ${draft.text}" }
        val vectors = embeddings.embed(inputs)
        require(vectors.size == drafts.size) { "Embedding-сервис вернул неверное число векторов." }
        require(vectors.all { vector -> vector.isNotEmpty() && vector.all(Float::isFinite) }) {
            "Embedding-сервис вернул пустой или некорректный вектор."
        }
        index.saveChunks(
            sourceId,
            drafts.mapIndexed { position, draft -> StoredChunk(draft, vectors[position], embeddings.modelName) },
        )
    }

    private fun safeSourceError(error: Exception): String = when (error) {
        is LocalModelException -> error.message ?: "Локальная embedding-модель недоступна."
        else -> error.message?.takeIf(String::isNotBlank)?.take(MAX_ERROR_LENGTH)
            ?: "Не удалось обработать источник; проверьте его формат и локальную модель."
    }

    companion object {
        private const val EMBEDDING_BATCH_SIZE = 16
        private const val MAX_ERROR_LENGTH = 240
    }
}
