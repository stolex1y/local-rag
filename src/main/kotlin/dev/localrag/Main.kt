package dev.localrag

import dev.localrag.app.ApplicationService
import dev.localrag.app.JobManager
import dev.localrag.app.RagService
import dev.localrag.benchmark.BenchmarkRunner
import dev.localrag.benchmark.BenchmarkStore
import dev.localrag.index.FixedSizeChunker
import dev.localrag.index.IndexWorkflow
import dev.localrag.index.SourceExtractorRegistry
import dev.localrag.index.SqliteIndexRepository
import dev.localrag.index.StructuralChunker
import dev.localrag.ollama.OllamaApi
import dev.localrag.ollama.OllamaChatPort
import dev.localrag.ollama.OllamaEmbeddingPort
import dev.localrag.source.LocalRagPaths
import dev.localrag.source.SourceCatalog
import dev.localrag.web.LocalHttpServer
import java.net.URI
import java.util.concurrent.CountDownLatch

fun main() {
    val paths = LocalRagPaths.current()
    val index = SqliteIndexRepository(paths.database)
    try {
        val benchmarkStore = BenchmarkStore(paths.database)
        try {
            val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
            val jobs = JobManager()
            try {
                val ollamaAddress = System.getenv("LOCAL_RAG_OLLAMA_URL")
                val ollama = if (ollamaAddress.isNullOrBlank()) OllamaApi() else OllamaApi(URI.create(ollamaAddress))
                val embeddings = OllamaEmbeddingPort(ollama)
                val chat = OllamaChatPort(ollama)
                val rag = RagService(index, embeddings, chat)
                val indexing = IndexWorkflow(
                    extractor = SourceExtractorRegistry(),
                    chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
                    embeddings = embeddings,
                    index = index,
                    sourcesDirectory = paths.sources,
                )
                val benchmark = BenchmarkRunner(rag, benchmarkStore, index)
                val application = ApplicationService(index, benchmarkStore, catalog, jobs, indexing, rag, benchmark, ollama)
                val port = System.getenv("LOCAL_RAG_PORT")?.toIntOrNull() ?: 8765
                LocalHttpServer(application, port).use { server ->
                    server.start()
                    println("Local RAG доступен: http://127.0.0.1:${server.port}/")
                    CountDownLatch(1).await()
                }
            } finally {
                jobs.close()
            }
        } finally {
            benchmarkStore.close()
        }
    } finally {
        index.close()
    }
}
