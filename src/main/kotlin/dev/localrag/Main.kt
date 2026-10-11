package dev.localrag

import dev.localrag.app.ApplicationService
import dev.localrag.app.JobManager
import dev.localrag.app.RagService
import dev.localrag.benchmark.BenchmarkRunner
import dev.localrag.benchmark.BenchmarkStore
import dev.localrag.chat.ChatService
import dev.localrag.chat.ChatStore
import dev.localrag.chat.WorkspaceStore
import dev.localrag.generation.ChatCompletionsApi
import dev.localrag.generation.EnvironmentProviderCredentials
import dev.localrag.generation.ModelConfiguration
import dev.localrag.generation.ModelSelectionStore
import dev.localrag.generation.ProviderCatalog
import dev.localrag.index.FixedSizeChunker
import dev.localrag.index.IndexWorkflow
import dev.localrag.index.SourceExtractorRegistry
import dev.localrag.index.SqliteIndexRepository
import dev.localrag.index.StructuralChunker
import dev.localrag.ollama.OllamaApi
import dev.localrag.ollama.OllamaEmbeddingPort
import dev.localrag.ollama.OllamaReranker
import dev.localrag.source.LocalRagPaths
import dev.localrag.source.SourceCatalog
import dev.localrag.web.LocalHttpServer
import dev.localrag.web.LocalHttpServerConfig
import java.net.URI
import java.nio.file.Path
import java.util.concurrent.CountDownLatch

fun main() {
    val serverConfig = LocalHttpServerConfig.fromEnvironment()
    val paths = LocalRagPaths.current()
    SqliteIndexRepository(paths.database).use { index ->
        BenchmarkStore(paths.database).use { benchmarkStore ->
            ChatStore(paths.database).use { chatStore ->
                WorkspaceStore(paths.database).use { workspaceStore ->
                    val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
                    val jobs = JobManager()
                    try {
                        val ollamaAddress = System.getenv("LOCAL_RAG_OLLAMA_URL")
                        val ollamaBaseUrl = if (ollamaAddress.isNullOrBlank()) URI("http://127.0.0.1:11434") else URI.create(ollamaAddress)
                        val ollama = OllamaApi(ollamaBaseUrl)
                        val modelCatalog = ProviderCatalog.load(
                            Path.of("config/providers.json"),
                            Path.of("config/agent.json"),
                            ollamaBaseUrl,
                            includeCloud = serverConfig.bindAddress.isLoopbackAddress,
                        )
                        val modelConfiguration = ModelConfiguration(modelCatalog, ModelSelectionStore(paths.modelSelection, modelCatalog), EnvironmentProviderCredentials)
                        val embeddings = OllamaEmbeddingPort(ollama)
                        val chat = ChatCompletionsApi(modelConfiguration)
                        val rag = RagService(index, embeddings, chat, OllamaReranker(ollama))
                        val chatService = ChatService(chatStore, rag, chat, embeddings)
                        val indexing = IndexWorkflow(
                            extractor = SourceExtractorRegistry(),
                            chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
                            embeddings = embeddings,
                            index = index,
                            sourcesDirectory = paths.sources,
                        )
                        val benchmark = BenchmarkRunner(rag, benchmarkStore, index)
                        val application = ApplicationService(
                            index,
                            benchmarkStore,
                            catalog,
                            jobs,
                            indexing,
                            rag,
                            benchmark,
                            ollama,
                            modelConfiguration,
                            chatStore,
                            chatService,
                        )
                        try {
                            LocalHttpServer(application, workspaceStore, serverConfig).use { server ->
                                server.start()
                                if (!serverConfig.bindAddress.isLoopbackAddress) {
                                    System.err.println("ПРЕДУПРЕЖДЕНИЕ: LAN HTTP без TLS и входа; любой доступный клиент может использовать общую коллекцию.")
                                }
                                println("Local RAG доступен: ${server.origin}/")
                                CountDownLatch(1).await()
                            }
                        } finally {
                            application.close()
                        }
                    } finally {
                        jobs.close()
                    }
                }
            }
        }
    }
}
