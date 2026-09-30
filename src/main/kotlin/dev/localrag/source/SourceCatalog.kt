package dev.localrag.source

import dev.localrag.domain.IndexRepository
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredSource
import kotlinx.serialization.Serializable
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.time.Instant
import java.util.UUID

@Serializable
data class ImportFileMetadata(
    val name: String,
    val sizeBytes: Long,
)

@Serializable
data class ImportFileSlot(
    val fileId: String,
    val sourceId: String,
    val name: String,
    val sizeBytes: Long,
)

@Serializable
data class ImportBatch(
    val importId: String,
    val totalBytes: Long,
    val files: List<ImportFileSlot>,
)

@Serializable
data class ImportFileResult(
    val fileId: String,
    val source: SourceRecord,
)

@Serializable
data class ImportCompletion(
    val files: List<ImportFileResult>,
    val sources: List<SourceRecord>,
)

class SourceCatalog(
    private val sourcesDirectory: Path,
    private val repository: IndexRepository,
    private val removeBenchmarkReferences: (String) -> Unit = {},
) {
    private data class MutableSlot(
        val slot: ImportFileSlot,
        var result: ImportFileResult? = null,
        var writing: Boolean = false,
    )

    private data class MutableBatch(
        val batch: ImportBatch,
        val startedAtMillis: Long,
        val slots: MutableMap<String, MutableSlot>,
    )

    private val lock = Any()
    private val batches = mutableMapOf<String, MutableBatch>()

    init {
        PrivateLocalStorage.prepareDirectory(sourcesDirectory)
        recoverDeletions()
        cleanOrphanedFiles()
        repository.recoverInterruptedIndexes()
    }

    fun beginImport(files: List<ImportFileMetadata>): ImportBatch {
        require(files.isNotEmpty()) { "Выберите хотя бы один файл." }
        val normalized = files.map { file ->
            val name = normalizeName(file.name)
            require(file.sizeBytes in 1..MAX_FILE_BYTES) {
                "Файл «$name» должен занимать от 1 байта до ${MAX_FILE_MIB} MiB."
            }
            ImportFileMetadata(name, file.sizeBytes)
        }
        val totalBytes = normalized.fold(0L) { total, file ->
            require(file.sizeBytes <= MAX_IMPORT_BYTES - total) {
                "Общий размер одного добавления не должен превышать ${MAX_IMPORT_MIB} MiB."
            }
            total + file.sizeBytes
        }
        val importId = UUID.randomUUID().toString()
        val slots = normalized.map { file ->
            ImportFileSlot(UUID.randomUUID().toString(), UUID.randomUUID().toString(), file.name, file.sizeBytes)
        }
        val batch = ImportBatch(importId, totalBytes, slots)
        synchronized(lock) {
            expireOldBatches()
            ensurePendingIndexRunCapacity(totalBytes)
            batches[importId] = MutableBatch(batch, System.currentTimeMillis(), slots.associate { it.fileId to MutableSlot(it) }.toMutableMap())
        }
        return batch
    }

    private fun ensurePendingIndexRunCapacity(additionalBytes: Long) {
        val pendingSources = repository.pendingSources()
        val pendingIds = HashSet<String>(pendingSources.size)
        var usedBytes = 0L

        fun reserve(bytes: Long) {
            require(bytes <= MAX_IMPORT_BYTES - usedBytes) {
                "Суммарный размер ожидающих индексации источников и добавления не должен превышать ${MAX_IMPORT_MIB} MiB за один запуск."
            }
            usedBytes += bytes
        }

        pendingSources.forEach { source ->
            pendingIds.add(source.record.sourceId)
            reserve(source.record.sizeBytes)
        }
        batches.values.forEach { batch ->
            batch.slots.values.forEach { current ->
                if (current.result == null && current.slot.sourceId !in pendingIds) {
                    reserve(current.slot.sizeBytes)
                }
            }
        }
        reserve(additionalBytes)
    }

    fun writeImportedFile(importId: String, fileId: String, input: InputStream): ImportFileResult {
        val state = synchronized(lock) {
            val batch = batches[importId] ?: throw ImportNotFoundException()
            val slot = batch.slots[fileId] ?: throw ImportNotFoundException()
            slot.result?.let { return it }
            if (slot.writing) throw ImportInProgressException()
            slot.writing = true
            slot
        }
        val slot = state.slot
        val temporary = pathFor(slot.sourceId, TEMP_SUFFIX)
        val stored = pathFor(slot.sourceId, SOURCE_SUFFIX)
        return try {
            var copied = 0L
            val digest = MessageDigest.getInstance(SHA256_ALGORITHM)
            input.use { source ->
                PrivateLocalStorage.createPrivateFile(temporary)
                Files.newOutputStream(temporary, StandardOpenOption.WRITE).use { output ->
                    val buffer = ByteArray(COPY_BUFFER_BYTES)
                    while (true) {
                        val count = source.read(buffer)
                        if (count < 0) break
                        if (count == 0) continue
                        copied += count
                        if (copied > slot.sizeBytes || copied > MAX_FILE_BYTES) {
                            throw SourceFileException("Размер фактических данных превысил объявленный лимит.", SourceStatus.FAILED)
                        }
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                }
            }
            if (copied != slot.sizeBytes) {
                throw SourceFileException("Файл передан не полностью: ожидалось ${slot.sizeBytes} байт, получено $copied.", SourceStatus.FAILED)
            }
            val contentHash = HEX_FORMAT.formatHex(digest.digest())
            val type = sourceType(slot.name, temporary)
            synchronized(lock) {
                val existing = sourceWithContentHash(contentHash, copied)

                if (existing != null) {
                    if (existing.record.status == SourceStatus.FAILED) {
                        val storedCopy = pathForKey(requireNotNull(existing.storageKey))
                        if (Files.isRegularFile(storedCopy)) {
                            Files.deleteIfExists(temporary)
                        } else {
                            moveAtomically(temporary, storedCopy)
                        }
                        repository.markSourceStatus(existing.record.sourceId, SourceStatus.PENDING)
                        val requeued = repository.source(existing.record.sourceId)
                            ?: throw IllegalStateException("The matched source no longer exists.")
                        complete(importId, slot.fileId, ImportFileResult(slot.fileId, requeued))
                    } else {
                        Files.deleteIfExists(temporary)
                        complete(importId, slot.fileId, ImportFileResult(slot.fileId, existing.record))
                    }
                } else {
                    moveAtomically(temporary, stored)
                    val record = SourceRecord(
                        sourceId = slot.sourceId,
                        name = slot.name,
                        type = type,
                        sizeBytes = copied,
                        status = SourceStatus.PENDING,
                        createdAt = Instant.now().toString(),
                    )
                    try {
                        repository.registerSource(StoredSource(record, slot.sourceId + SOURCE_SUFFIX, contentHash))
                    } catch (error: Exception) {
                        Files.deleteIfExists(stored)
                        throw error
                    }
                    complete(importId, slot.fileId, ImportFileResult(slot.fileId, record))
                }
            }
        } catch (error: SourceFileException) {
            Files.deleteIfExists(temporary)
            failedResult(importId, slot, error.status, error.message ?: "Не удалось принять файл.")
        } catch (error: IOException) {
            Files.deleteIfExists(temporary)
            failedResult(importId, slot, SourceStatus.FAILED, "Не удалось прочитать или сохранить файл: ${safeMessage(error)}")
        } catch (error: Exception) {
            Files.deleteIfExists(temporary)
            synchronized(lock) { batches[importId]?.slots?.get(slot.fileId)?.writing = false }
            throw error
        }

    }
    private fun sourceWithContentHash(hash: String, sizeBytes: Long): StoredSource? {
        repository.sourceByContentSha256(hash)?.let { return it }
        for (source in repository.unhashedSourcesOfSize(sizeBytes)) {
            val path = pathForKey(source.storageKey!!)
            if (!Files.isRegularFile(path)) continue
            val existingHash = sha256(path)
            repository.updateSourceContentSha256(source.record.sourceId, existingHash)
            if (existingHash == hash) return source.copy(contentSha256 = existingHash)
        }
        return repository.sourceByContentSha256(hash)
    }

    private fun sha256(path: Path): String = Files.newInputStream(path).use { input ->
        val digest = MessageDigest.getInstance(SHA256_ALGORITHM)
        val buffer = ByteArray(COPY_BUFFER_BYTES)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            if (count > 0) digest.update(buffer, 0, count)
        }
        HEX_FORMAT.formatHex(digest.digest())
    }

    fun failImportedFile(importId: String, fileId: String, message: String): ImportFileResult {
        val slot = synchronized(lock) {
            val current = batches[importId]?.slots?.get(fileId) ?: throw ImportNotFoundException()
            current.result?.let { return it }
            if (current.writing) throw ImportInProgressException()
            current.writing = true
            current.slot
        }
        return failedResult(importId, slot, SourceStatus.FAILED, message)
    }

    fun finishImport(importId: String): ImportCompletion {
        val batch = synchronized(lock) {
            val current = batches[importId] ?: throw ImportNotFoundException()
            if (current.slots.values.any(MutableSlot::writing)) throw ImportInProgressException()
            current.slots.values.filter { it.result == null }.forEach { it.writing = true }
            current
        }
        val results = batch.slots.values.map { current ->
            current.result ?: failedResult(importId, current.slot, SourceStatus.FAILED, "Файл не был загружен.")
        }
        synchronized(lock) { batches.remove(importId) }
        return ImportCompletion(results, repository.sources())
    }

    fun sources(): List<SourceRecord> = repository.sources()

    fun pendingSources(): List<SourceRecord> = repository.pendingSources().map(StoredSource::record)

    fun sourceEntries(status: SourceStatus): List<StoredSource> = repository.sources(status)

    fun deleteSource(sourceId: String, confirmed: Boolean) {
        synchronized(lock) {
            require(confirmed) { "Удаление источника требует отдельного подтверждения." }
            require(isGeneratedId(sourceId)) { "Идентификатор источника некорректен." }
            val source = repository.storedSource(sourceId) ?: throw SourceNotFoundException()
            if (source.record.status == SourceStatus.INDEXING) throw SourceBusyException()
            repository.markSourceStatus(sourceId, SourceStatus.DELETING)
            source.storageKey?.let { key -> Files.deleteIfExists(pathForKey(key)) }
            removeBenchmarkReferences(sourceId)
            repository.deleteSource(sourceId)
        }
    }
    private fun validateUtf8(path: Path) {
        val decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
        try {
            Files.newInputStream(path).use { input ->
                val reader = java.io.InputStreamReader(input, decoder)
                val chars = CharArray(TEXT_VALIDATION_BUFFER_CHARS)
                while (true) {
                    val count = reader.read(chars)
                    if (count < 0) break
                    for (index in 0 until count) {
                        val char = chars[index]
                        if (char == '\u0000' || (char.code < 0x20 && char !in ALLOWED_CONTROLS)) {
                            throw SourceFileException("Файл содержит бинарные или управляющие данные, а не UTF-8 текст.", SourceStatus.UNSUPPORTED)
                        }
                    }
                }
            }
        } catch (_: java.nio.charset.CharacterCodingException) {
            throw SourceFileException("Файл не является корректным UTF-8 текстом.", SourceStatus.UNSUPPORTED)
        }
    }

    private fun failedResult(
        importId: String,
        slot: ImportFileSlot,
        status: SourceStatus,
        message: String,
    ): ImportFileResult {
        try {
            val record = SourceRecord(
                sourceId = slot.sourceId,
                name = slot.name,
                type = sourceTypeFromName(slot.name),
                sizeBytes = slot.sizeBytes,
                status = status,
                error = message.take(MAX_ERROR_LENGTH),
                createdAt = Instant.now().toString(),
            )
            repository.registerSource(StoredSource(record, null))
            return complete(importId, slot.fileId, ImportFileResult(slot.fileId, record))
        } catch (error: Exception) {
            synchronized(lock) { batches[importId]?.slots?.get(slot.fileId)?.writing = false }
            throw error
        }
    }

    private fun complete(importId: String, fileId: String, result: ImportFileResult): ImportFileResult = synchronized(lock) {
        val slot = batches[importId]?.slots?.get(fileId) ?: throw ImportNotFoundException()
        slot.result = result
        slot.writing = false
        result
    }

    private fun sourceType(name: String, file: Path): SourceType {
        val extension = name.substringAfterLast('.', "").lowercase()
        return when (extension) {
            "pdf" -> {
                val signature = Files.newInputStream(file).use { it.readNBytes(PDF_SIGNATURE_BYTES) }
                if (!signature.contentEquals(PDF_SIGNATURE)) {
                    throw SourceFileException("Файл с расширением PDF не содержит заголовок PDF.", SourceStatus.UNSUPPORTED)
                }
                SourceType.PDF
            }
            "txt" -> {
                validateUtf8(file)
                SourceType.TEXT
            }
            "md", "markdown" -> {
                validateUtf8(file)
                SourceType.MARKDOWN
            }
            "htm", "html" -> {
                validateUtf8(file)
                SourceType.HTML
            }
            else -> {
                validateUtf8(file)
                SourceType.CODE
            }
        }
    }


    private fun sourceTypeFromName(name: String): SourceType = when (name.substringAfterLast('.', "").lowercase()) {
        "pdf" -> SourceType.PDF
        "txt" -> SourceType.TEXT
        "md", "markdown" -> SourceType.MARKDOWN
        "htm", "html" -> SourceType.HTML
        else -> SourceType.CODE
    }

    private fun normalizeName(rawName: String): String {
        val basename = rawName.replace('\\', '/').substringAfterLast('/').trim()
        require(basename.isNotEmpty() && basename != "." && basename != "..") { "Имя файла пустое или некорректное." }
        require(basename.length <= MAX_NAME_LENGTH) { "Имя файла не должно превышать $MAX_NAME_LENGTH символов." }
        require(basename.none { it.isISOControl() }) { "Имя файла содержит недопустимые управляющие символы." }
        return basename
    }

    private fun recoverDeletions() {
        repository.sources(SourceStatus.DELETING).forEach { source ->
            runCatching {
                source.storageKey?.let { Files.deleteIfExists(pathForKey(it)) }
                removeBenchmarkReferences(source.record.sourceId)
                repository.deleteSource(source.record.sourceId)
            }
        }
    }

    private fun cleanOrphanedFiles() {
        val known = repository.storedSources().mapNotNull(StoredSource::storageKey).toSet()
        Files.newDirectoryStream(sourcesDirectory).use { files ->
            files.forEach { path ->
                val filename = path.fileName.toString()
                if (filename.endsWith(TEMP_SUFFIX) || (filename.endsWith(SOURCE_SUFFIX) && filename !in known)) {
                    runCatching { Files.deleteIfExists(path) }
                }
            }
        }
    }

    private fun expireOldBatches() {
        val expiration = System.currentTimeMillis() - SESSION_TTL_MILLIS
        val expired = batches.filterValues { batch ->
            batch.startedAtMillis < expiration && batch.slots.values.none(MutableSlot::writing)
        }.keys
        expired.forEach { importId ->
            val batch = batches.remove(importId) ?: return@forEach
            batch.slots.values.filter { it.result == null }.forEach { slot ->
                runCatching { Files.deleteIfExists(pathFor(slot.slot.sourceId, TEMP_SUFFIX)) }
            }
        }
    }

    private fun pathFor(sourceId: String, suffix: String): Path = pathForKey(sourceId + suffix)

    private fun pathForKey(key: String): Path {
        require(key.matches(STORAGE_KEY_PATTERN)) { "Stored source key is invalid." }
        val resolved = sourcesDirectory.resolve(key).normalize()
        require(resolved.parent == sourcesDirectory.normalize()) { "Stored source key escapes the source directory." }
        return resolved
    }

    private fun moveAtomically(from: Path, to: Path) {
        try {
            Files.move(from, to, StandardCopyOption.ATOMIC_MOVE)
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(from, to)
        }
    }

    private fun safeMessage(error: IOException): String = "не удалось сохранить файл из-за ошибки ввода-вывода"

    private fun isGeneratedId(value: String): Boolean = runCatching { UUID.fromString(value).toString() == value }.getOrDefault(false)

    private class SourceFileException(message: String, val status: SourceStatus) : RuntimeException(message)

    companion object {
        const val MAX_FILE_BYTES = 256L * 1024 * 1024
        const val MAX_IMPORT_BYTES = 512L * 1024 * 1024
        const val MAX_FILE_MIB = 256
        const val MAX_IMPORT_MIB = 512
        private const val COPY_BUFFER_BYTES = 64 * 1024
        private const val TEXT_VALIDATION_BUFFER_CHARS = 32 * 1024
        private const val PDF_SIGNATURE_BYTES = 5
        private const val MAX_NAME_LENGTH = 255
        private const val MAX_ERROR_LENGTH = 240
        private const val SESSION_TTL_MILLIS = 30 * 60 * 1000L
        private const val SOURCE_SUFFIX = ".source"
        private const val TEMP_SUFFIX = ".partial"
        private const val SHA256_ALGORITHM = "SHA-256"
        private val HEX_FORMAT = java.util.HexFormat.of()
        private val PDF_SIGNATURE = "%PDF-".toByteArray(StandardCharsets.US_ASCII)
        private val ALLOWED_CONTROLS = setOf('\n', '\r', '\t', '\u000c')
        private val STORAGE_KEY_PATTERN = Regex("[0-9a-fA-F-]{36}\\.(?:source|partial)")
    }
}

class ImportNotFoundException : RuntimeException("Сеанс импорта или файл не найден.")
class ImportInProgressException : RuntimeException("Этот файл уже загружается.")
class SourceNotFoundException : RuntimeException("Источник не найден.")
class SourceBusyException : RuntimeException("Источник нельзя удалить, пока его индексируют.")
