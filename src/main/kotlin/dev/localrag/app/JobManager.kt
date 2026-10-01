package dev.localrag.app

import dev.localrag.domain.IndexProgressUpdate
import dev.localrag.ollama.LocalModelException
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

class JobBusyException : RuntimeException("Другое локальное задание ещё выполняется.")
class JobNotFoundException : RuntimeException("Задание не найдено или его состояние уже удалено.")

@kotlinx.serialization.Serializable
data class JobSnapshot(
    val jobId: String,
    val kind: String,
    val state: String,
    val phase: String,
    val filesDone: Int,
    val filesTotal: Int,
    val bytesDone: Long,
    val bytesTotal: Long,
    val currentSource: String? = null,
    val currentPosition: String? = null,
    val succeeded: Int = 0,
    val failed: Int = 0,
    val elapsedMs: Long,
    val error: String? = null,
)

class JobProgress internal constructor(
    private val updateBlock: (IndexProgressUpdate) -> Unit,
) {
    fun update(value: IndexProgressUpdate) = updateBlock(value)
}

class JobManager(
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { task ->
        Thread(task, "local-rag-worker").apply { isDaemon = true }
    },
) : AutoCloseable {
    private class MutableJob(
        val jobId: String,
        val kind: String,
        val filesTotal: Int,
        val bytesTotal: Long,
        val startedAt: Long,
    ) {
        var state: String = "queued"
        var phase: String = "queued"
        var filesDone: Int = 0
        var bytesDone: Long = 0
        var currentSource: String? = null
        var currentPosition: String? = null
        var succeeded: Int = 0
        var failed: Int = 0
        var error: String? = null

        fun snapshot(now: Long) = JobSnapshot(
            jobId = jobId,
            kind = kind,
            state = state,
            phase = phase,
            filesDone = filesDone,
            filesTotal = filesTotal,
            bytesDone = bytesDone.coerceAtMost(bytesTotal),
            bytesTotal = bytesTotal,
            currentSource = currentSource,
            currentPosition = currentPosition,
            succeeded = succeeded,
            failed = failed,
            elapsedMs = (now - startedAt).coerceAtLeast(0L),
            error = error,
        )
    }

    private val lock = Any()
    private val jobs = linkedMapOf<String, MutableJob>()
    @Volatile private var activeJob: MutableJob? = null

    fun submit(
        kind: String,
        filesTotal: Int,
        bytesTotal: Long,
        task: (JobProgress) -> Unit,
    ): JobSnapshot {
        require(kind in setOf("INDEX", "BENCHMARK"))
        require(filesTotal > 0 && bytesTotal >= 0)
        synchronized(lock) {
            if (activeJob != null) throw JobBusyException()
            pruneHistory()
            val job = MutableJob(UUID.randomUUID().toString(), kind, filesTotal, bytesTotal, System.currentTimeMillis())
            jobs[job.jobId] = job
            activeJob = job
            try {
                executor.execute {
                    synchronized(lock) { job.state = "running" }
                    try {
                        task(JobProgress { update ->
                            synchronized(lock) {
                                job.phase = update.phase
                                job.filesDone = update.filesDone.coerceIn(0, job.filesTotal)
                                job.bytesDone = update.bytesDone.coerceIn(0, job.bytesTotal)
                                job.currentSource = update.currentSource
                                job.currentPosition = update.currentPosition
                                job.succeeded = update.succeeded.coerceIn(0, job.filesTotal)
                                job.failed = update.failed.coerceIn(0, job.filesTotal)
                            }
                        })
                        synchronized(lock) {
                            job.state = "completed"
                            job.phase = "completed"
                            job.filesDone = job.filesTotal
                            job.bytesDone = job.bytesTotal
                            job.currentSource = null
                            job.currentPosition = null
                        }
                    } catch (error: Exception) {
                        synchronized(lock) {
                            job.state = "failed"
                            job.phase = "failed"
                            job.currentSource = null
                            job.currentPosition = null
                            job.error = safeError(error)
                        }
                    } finally {
                        synchronized(lock) { if (activeJob === job) activeJob = null }
                    }
                }
            } catch (error: RejectedExecutionException) {
                activeJob = null
                jobs.remove(job.jobId)
                throw JobBusyException()
            }
            return job.snapshot(System.currentTimeMillis())
        }
    }

    fun get(jobId: String): JobSnapshot? = synchronized(lock) {
        jobs[jobId]?.snapshot(System.currentTimeMillis())
    }

    fun active(): JobSnapshot? = activeJob?.let { job -> synchronized(lock) { job.snapshot(System.currentTimeMillis()) } }

    private fun pruneHistory() {
        val terminal = jobs.values.filter { it.state in TERMINAL_STATES }
        terminal.take((jobs.size - MAX_RETAINED_JOBS + 1).coerceAtLeast(0)).forEach { jobs.remove(it.jobId) }
    }

    private fun safeError(error: Exception): String = when (error) {
        is LocalModelException -> error.message ?: "Локальная модель недоступна."
        else -> error.message?.takeIf(String::isNotBlank)?.take(MAX_ERROR_LENGTH) ?: "Задание не выполнено."
    }

    override fun close() {
        executor.shutdownNow()
    }

    companion object {
        private const val MAX_RETAINED_JOBS = 40
        private const val MAX_ERROR_LENGTH = 240
        private val TERMINAL_STATES = setOf("completed", "failed")
    }
}
