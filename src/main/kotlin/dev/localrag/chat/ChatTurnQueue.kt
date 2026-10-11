package dev.localrag.chat

import dev.localrag.domain.ModelSelection
import dev.localrag.domain.ChunkStrategy
import java.util.concurrent.Callable
import java.util.concurrent.FutureTask
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadFactory
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Runs every provider through one durable, process-wide inference queue. */
class ChatTurnQueue(
    private val store: ChatStore,
    private val chatService: ChatService,
) : AutoCloseable {
    private val executor = ThreadPoolExecutor(
        1,
        1,
        0L,
        TimeUnit.MILLISECONDS,
        LinkedBlockingQueue(),
        ThreadFactory { task -> Thread(task, "local-rag-chat-turn").apply { isDaemon = true } },
    )
    private val transitionLocks = Array(64) { Any() }

    private fun transitionLock(turnId: String): Any =
        transitionLocks[turnId.hashCode() and (transitionLocks.size - 1)]
    private val scheduled = java.util.concurrent.ConcurrentHashMap<String, FutureTask<Unit>>()

    fun enqueue(
        workspaceId: String,
        sessionId: String,
        turnId: String,
        question: String,
        strategy: ChunkStrategy,
        topK: Int?,
        selection: ModelSelection,
    ): ChatTurnJob = synchronized(transitionLock(turnId)) {
        val job = store.enqueueTurn(workspaceId, sessionId, turnId, question, strategy, topK, selection)
        if (job.state == "queued") schedule(job, workspaceId)
        job
    }

    fun get(workspaceId: String, sessionId: String, turnId: String): ChatTurnJob? =
        store.turnJob(workspaceId, sessionId, turnId)

    fun cancel(workspaceId: String, sessionId: String, turnId: String): ChatTurnJob? =
        synchronized(transitionLock(turnId)) {
            val job = store.cancelTurn(workspaceId, sessionId, turnId) ?: return null
            if (job.state == "cancelled") {
                val task = scheduled.remove(turnId)
                if (task != null) {
                    task.cancel(true)
                    executor.remove(task)
                }
            }
            job
        }

    private fun schedule(job: ChatTurnJob, workspaceId: String) {
        val reference = AtomicReference<FutureTask<Unit>>()
        val task = FutureTask(Callable {
            try {
                if (store.markTurnRunning(workspaceId, job.sessionId, job.turnId)) {
                    chatService.turn(
                        sessionId = job.sessionId,
                        turnId = job.turnId,
                        question = job.question,
                        strategy = job.strategy,
                        selection = job.selection,
                        topK = job.topK,
                        workspaceId = workspaceId,
                    )
                    store.markTurnCompleted(workspaceId, job.sessionId, job.turnId)
                }
            } catch (error: Exception) {
                if (!Thread.currentThread().isInterrupted && error !is ChatTurnCancelledException) {
                    val message = error.message?.take(ChatStore.MAX_TURN_ERROR_LENGTH)
                        ?: "Не удалось завершить запрос. Проверьте журнал приложения."
                    store.markTurnFailed(workspaceId, job.sessionId, job.turnId, message)
                }
            } finally {
                reference.get()?.let { scheduled.remove(job.turnId, it) }
            }
            Unit
        })
        reference.set(task)
        while (true) {
            val current = scheduled[job.turnId]
            if (current != null && !current.isDone) return
            val installed = if (current == null) {
                scheduled.putIfAbsent(job.turnId, task) == null
            } else {
                scheduled.replace(job.turnId, current, task)
            }
            if (installed) {
                executor.execute(task)
                return
            }
        }
    }

    override fun close() {
        executor.shutdownNow()
    }
}
