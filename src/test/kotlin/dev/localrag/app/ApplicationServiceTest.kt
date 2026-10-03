package dev.localrag.app

import dev.localrag.chat.ChatStore
import dev.localrag.chat.ChatService

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import dev.localrag.benchmark.BenchmarkExpectedSource
import dev.localrag.benchmark.BenchmarkQuestion
import dev.localrag.benchmark.BenchmarkRunner
import dev.localrag.benchmark.BenchmarkStore
import dev.localrag.domain.ChunkDraft
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.SourceLocation
import dev.localrag.domain.SourceRecord
import dev.localrag.domain.SourceStatus
import dev.localrag.domain.SourceType
import dev.localrag.domain.StoredChunk
import dev.localrag.domain.ChatPort
import dev.localrag.domain.ModelSelection
import dev.localrag.domain.RerankPort
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.StoredSource
import dev.localrag.index.FixedSizeChunker
import dev.localrag.index.IndexWorkflow
import dev.localrag.index.SourceExtractorRegistry
import dev.localrag.index.SqliteIndexRepository
import dev.localrag.index.StructuralChunker
import dev.localrag.ollama.OllamaApi
import dev.localrag.ollama.OllamaEmbeddingPort
import dev.localrag.source.SourceCatalog
import dev.localrag.web.LocalHttpServer
import dev.localrag.generation.ModelConfiguration
import dev.localrag.generation.ModelSelectionStore
import dev.localrag.generation.ProviderCatalog
import dev.localrag.generation.ProviderCredentialSource
import dev.localrag.generation.ProviderDefinition
import dev.localrag.generation.ProviderModelDefinition
import java.net.InetSocketAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.jupiter.api.io.TempDir

class ApplicationServiceTest {
    private data class Gate(val entered: CountDownLatch = CountDownLatch(1), val release: CountDownLatch = CountDownLatch(1))
    @TempDir
    lateinit var temporaryDirectory: Path


    @Test
    fun `chat session routes create list restore and require delete confirmation`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val created = api.postJson("/api/chat/sessions", "{}")
                assertEquals(201, created.statusCode())
                val sessionId = json(created.body()).jsonObject.getValue("session").jsonObject
                    .getValue("id").jsonPrimitive.content
                val detail = api.get("/api/chat/sessions/$sessionId")
                assertEquals(200, detail.statusCode())
                assertEquals(0, json(detail.body()).jsonObject.getValue("session").jsonObject
                    .getValue("messageCount").jsonPrimitive.content.toInt())
                assertEquals(1, json(api.get("/api/chat/sessions").body()).jsonArray.size)
                assertEquals(400, api.deleteJson("/api/chat/sessions/$sessionId", """{"confirm":false}""").statusCode())
                assertEquals(200, api.deleteJson("/api/chat/sessions/$sessionId", """{"confirm":true}""").statusCode())
                assertEquals(404, api.get("/api/chat/sessions/$sessionId").statusCode())
                assertEquals(404, api.deleteJson("/api/chat/sessions/$sessionId", """{"confirm":true}""").statusCode())
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `chat turn persists user input abstains without evidence and retries idempotently`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val sessionId = json(api.postJson("/api/chat/sessions", "{}").body())
                    .jsonObject.getValue("session").jsonObject.getValue("id").jsonPrimitive.content
                val turnId = java.util.UUID.randomUUID().toString()
                val body = """{"turnId":"$turnId","question":"Какая дата оплаты?","strategy":"STRUCTURAL"}"""
                val first = api.postJson("/api/chat/sessions/$sessionId/turns", body)
                assertEquals(200, first.statusCode(), first.body())
                val firstMessage = json(first.body()).jsonObject.getValue("message").jsonObject
                assertEquals(dev.localrag.chat.ChatService.ABSTENTION, firstMessage.getValue("content").jsonPrimitive.content)
                assertEquals(0, firstMessage.getValue("documentCitations").jsonArray.size)
                assertEquals(0, firstMessage.getValue("memoryReferences").jsonArray.size)

                val retried = api.postJson("/api/chat/sessions/$sessionId/turns", body)
                assertEquals(200, retried.statusCode())
                val retriedMessage = json(retried.body()).jsonObject.getValue("message").jsonObject
                assertEquals(firstMessage.getValue("id"), retriedMessage.getValue("id"))
                val detail = json(api.get("/api/chat/sessions/$sessionId").body()).jsonObject
                assertEquals(2, detail.getValue("messages").jsonArray.size)
            } finally {
                fixture.close()
            }
        }
    }
    @Test
    fun `HTTP flow imports additively indexes queries benchmarks and deletes referenced source data`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val status = api.get("/api/status")
                assertEquals(200, status.statusCode())
                val initialStatus = json(status.body()).jsonObject
                assertTrue(initialStatus.getValue("models").jsonObject.getValue("available").jsonPrimitive.content.toBoolean())
                assertEquals(0, initialStatus.getValue("sources").jsonObject.getValue("total").jsonPrimitive.content.toInt())
                assertFalse(initialStatus.containsKey("pilot"))
                assertEquals(404, api.get("/api/pilot").statusCode())

                val guide = "Orbit markers follow a stable route.\n".toByteArray()
                val begin = api.postJson(
                    "/api/imports",
                    """{"files":[{"name":"guide.txt","sizeBytes":${guide.size}},{"name":"damaged.bin","sizeBytes":2}]}""",
                    requestOrigin = "http://attacker.example",
                )
                assertEquals(403, begin.statusCode())
                val tooLarge = api.postJson(
                    "/api/imports",
                    """{"files":[{"name":"oversized.txt","sizeBytes":${dev.localrag.source.SourceCatalog.MAX_FILE_BYTES + 1}}]}""",
                )
                assertEquals(400, tooLarge.statusCode())

                val imported = importBatch(
                    api,
                    listOf("guide.txt" to guide, "damaged.bin" to byteArrayOf(0, 0xff.toByte())),
                )
                val guideRecord = imported.getValue("guide.txt")
                val unsupportedRecord = imported.getValue("damaged.bin")
                assertEquals("PENDING", guideRecord.getValue("status").jsonPrimitive.content)
                assertEquals("UNSUPPORTED", unsupportedRecord.getValue("status").jsonPrimitive.content)
                val guideId = guideRecord.getValue("sourceId").jsonPrimitive.content

                val pending = json(api.get("/api/index/pending").body()).jsonObject
                assertEquals(listOf(guideId), pending.getValue("files").jsonArray.map { it.jsonObject.getValue("sourceId").jsonPrimitive.content })
                assertEquals(guide.size, pending.getValue("totalBytes").jsonPrimitive.content.toInt())
                val unconfirmed = api.postJson("/api/index", indexPayload(false, listOf(guideId), guide.size.toLong()))
                assertEquals(400, unconfirmed.statusCode())
                val changedSnapshot = api.postJson("/api/index", indexPayload(true, emptyList(), 0))
                assertEquals(409, changedSnapshot.statusCode())

                val firstIndex = api.postJson("/api/index", indexPayload(true, listOf(guideId), guide.size.toLong()))
                assertEquals(202, firstIndex.statusCode())
                val firstJobId = json(firstIndex.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                val firstJob = api.awaitCompleted(firstJobId)
                assertEquals(1, firstJob.getValue("filesDone").jsonPrimitive.content.toInt())
                assertEquals(1, firstJob.getValue("succeeded").jsonPrimitive.content.toInt())
                assertFalse(firstJob.containsKey("etaMs"))
                assertFalse(firstJob.containsKey("fullEstimateMs"))
                assertEquals(1, firstJob.getValue("filesTotal").jsonPrimitive.content.toInt())

                val notes = "Copper wire carries the current.\n".toByteArray()
                val notesRecord = importBatch(api, listOf("notes.md" to notes)).getValue("notes.md")
                val notesId = notesRecord.getValue("sourceId").jsonPrimitive.content
                val secondPending = json(api.get("/api/index/pending").body()).jsonObject
                assertEquals(listOf(notesId), secondPending.getValue("files").jsonArray.map { it.jsonObject.getValue("sourceId").jsonPrimitive.content })
                val secondIndex = api.postJson("/api/index", indexPayload(true, listOf(notesId), notes.size.toLong()))
                assertEquals(202, secondIndex.statusCode())
                val secondJobId = json(secondIndex.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(1, api.awaitCompleted(secondJobId).getValue("succeeded").jsonPrimitive.content.toInt())

                val statusAfterIndexing = json(api.get("/api/status").body()).jsonObject
                assertEquals(2, statusAfterIndexing.getValue("sources").jsonObject.getValue("ready").jsonPrimitive.content.toInt())
                assertEquals(1, statusAfterIndexing.getValue("sources").jsonObject.getValue("failed").jsonPrimitive.content.toInt())
                val query = api.postJson(
                    "/api/query",
                    """{"question":"Where are orbit markers?","strategy":"FIXED_SIZE","topK":1}""",
                )
                assertEquals(200, query.statusCode())
                val answer = json(query.body()).jsonObject
                assertEquals("Baseline answer without indexed sources.", answer.getValue("baseline").jsonObject.getValue("answer").jsonPrimitive.content)
                val rag = answer.getValue("rag").jsonObject
                assertEquals("RAG answer grounded in imported context.", rag.getValue("answer").jsonPrimitive.content)
                val citation = rag.getValue("sources").jsonArray.single().jsonObject
                assertEquals(guideId, citation.getValue("sourceId").jsonPrimitive.content)
                assertEquals("guide.txt", citation.getValue("source").jsonPrimitive.content)
                assertEquals(1, citation.getValue("location").jsonObject.getValue("lineStart").jsonPrimitive.content.toInt())
                assertEquals("Orbit markers follow a stable route.", citation.getValue("quote").jsonPrimitive.content)

                val questions = (1..10).map { number ->
                    BenchmarkQuestion(
                        id = "q${number.toString().padStart(2, '0')}",
                        question = "Where are orbit markers? Case $number",
                        expectedFacts = listOf("The route is described."),
                        expectedSources = listOf(
                            BenchmarkExpectedSource(
                                sourceId = guideId,
                                location = dev.localrag.domain.SourceLocation(lineStart = 1, lineEnd = 1),
                                section = "Document body",
                            ),
                        ),
                    )
                }
                val questionsJson = Json.encodeToString(questions)
                val savedQuestions = api.putJson("/api/benchmark/questions", """{"questions":$questionsJson}""")
                assertEquals(200, savedQuestions.statusCode(), savedQuestions.body())
                assertTrue(json(savedQuestions.body()).jsonObject.getValue("runnable").jsonPrimitive.content.toBoolean())
                val rerankerCallsBeforeBenchmark = fixture.rerankCalls.get()
                val benchmarkStart = api.postJson("/api/benchmark/run", "{}")
                assertEquals(202, benchmarkStart.statusCode())
                val benchmarkJobId = json(benchmarkStart.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(20, api.awaitCompleted(benchmarkJobId).getValue("filesDone").jsonPrimitive.content.toInt())
                val resultsResponse = json(api.get("/api/benchmark/results").body()).jsonObject
                assertEquals(20, resultsResponse.getValue("results").jsonArray.size)
                val benchmarkRows = resultsResponse.getValue("results").jsonArray.map { it.jsonObject }
                val fixedQuestionIds = benchmarkRows.filter {
                    it.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE"
                }.map { it.getValue("questionId").jsonPrimitive.content }.toSet()
                val structuralQuestionIds = benchmarkRows.filter {
                    it.getValue("strategy").jsonPrimitive.content == "STRUCTURAL"
                }.map { it.getValue("questionId").jsonPrimitive.content }.toSet()
                assertEquals((1..10).map { "q${it.toString().padStart(2, '0')}" }.toSet(), fixedQuestionIds)
                assertEquals(fixedQuestionIds, structuralQuestionIds)
                val retrievalSummary = resultsResponse.getValue("summary").jsonObject.getValue("retrieval").jsonArray
                assertEquals(2, retrievalSummary.size)
                assertTrue(retrievalSummary.all { row ->
                    val metrics = row.jsonObject
                    metrics.getValue("completedQuestions").jsonPrimitive.content.toInt() == 10 &&
                        metrics.getValue("rawHitAtK").jsonPrimitive.content.toDouble() == 1.0 &&
                        metrics.getValue("rawMrr").jsonPrimitive.content.toDouble() == 1.0 &&
                        metrics.getValue("enhancedHitAtK").jsonPrimitive.content.toDouble() == 1.0 &&
                        metrics.getValue("enhancedMrr").jsonPrimitive.content.toDouble() == 1.0
                })
                val firstBenchmarkRow = benchmarkRows.first { it.getValue("questionId").jsonPrimitive.content == "q01" }
                assertEquals(1, firstBenchmarkRow.getValue("rawExpectedSourceRank").jsonPrimitive.content.toInt())
                assertEquals(1, firstBenchmarkRow.getValue("expectedSourceRank").jsonPrimitive.content.toInt())
                assertTrue(
                    firstBenchmarkRow.getValue("rawRetrievedSources").jsonArray.first().jsonObject
                        .getValue("quote").jsonPrimitive.content.startsWith("Orbit markers"),
                )
                assertEquals(20, fixture.rerankCalls.get() - rerankerCallsBeforeBenchmark)

                val review = api.postJson(
                    "/api/benchmark/review",
                    """{"questionId":"q01","strategy":"FIXED_SIZE","baselineRating":"FAIL","ragRating":"PASS","note":"Retrieved local evidence."}""",
                )
                assertEquals(200, review.statusCode())
                val reviewedRows = json(api.get("/api/benchmark/results").body()).jsonObject.getValue("results").jsonArray
                val reviewed = reviewedRows.map { it.jsonObject }.single { it.getValue("questionId").jsonPrimitive.content == "q01" && it.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE" }
                assertEquals("FAIL", reviewed.getValue("baselineRating").jsonPrimitive.content)
                assertEquals("PASS", reviewed.getValue("ragRating").jsonPrimitive.content)

                val stored = assertNotNull(fixture.index.storedSource(guideId))
                val storedCopy = fixture.sourcesDirectory.resolve(assertNotNull(stored.storageKey))
                val unconfirmedDelete = api.deleteJson("/api/sources/$guideId", """{"confirm":false}""")
                assertEquals(400, unconfirmedDelete.statusCode())
                assertTrue(Files.exists(storedCopy))
                val deleted = api.deleteJson("/api/sources/$guideId", """{"confirm":true}""")
                assertEquals(200, deleted.statusCode())
                assertFalse(Files.exists(storedCopy))
                assertEquals(null, fixture.index.source(guideId))
                assertTrue(fixture.index.search(ChunkStrategy.FIXED_SIZE, listOf(1f, 0f, 0f), dev.localrag.ollama.EMBEDDING_MODEL, 20).none { it.chunk.draft.sourceId == guideId })
                assertTrue(json(api.get("/api/benchmark/results").body()).jsonObject.getValue("results").jsonArray.isEmpty())
                val invalidatedQuestions = json(api.get("/api/benchmark/questions").body()).jsonObject
                assertFalse(invalidatedQuestions.getValue("runnable").jsonPrimitive.content.toBoolean())
                assertTrue(invalidatedQuestions.getValue("errors").jsonArray.any { "отсутствует" in it.jsonPrimitive.content })
            } finally {
                fixture.close()
            }
        }
    }


    @Test
    fun `retrieval summary uses ten question denominator for partial results`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val expectedSourceId = "00000000-0000-4000-8000-000000000001"
                val distractorSourceId = "00000000-0000-4000-8000-000000000002"
                val questions = (1..10).map { number ->
                    BenchmarkQuestion(
                        id = "q${number.toString().padStart(2, '0')}",
                        question = "Synthetic question $number",
                        expectedFacts = listOf("Expected fact $number"),
                        expectedSources = listOf(
                            BenchmarkExpectedSource(
                                sourceId = expectedSourceId,
                                location = SourceLocation(lineStart = 1, lineEnd = 1),
                                section = "Document body",
                            ),
                        ),
                    )
                }
                fixture.store.saveQuestions(questions)
                fun scoredChunk(chunkId: String, sourceId: String, text: String) = ScoredChunk(
                    chunk = StoredChunk(
                        draft = ChunkDraft(
                            strategy = ChunkStrategy.FIXED_SIZE,
                            chunkId = chunkId,
                            sourceId = sourceId,
                            sourceName = "$chunkId.txt",
                            section = "Document body",
                            location = SourceLocation(lineStart = 1, lineEnd = 1),
                            tokenUnits = 1,
                            text = text,
                        ),
                        embedding = listOf(1f, 0f, 0f, 0f),
                        embeddingModel = "fixture-embedding",
                    ),
                    cosineScore = 1.0,
                )

                val hit = scoredChunk("hit", expectedSourceId, "Expected excerpt")
                fixture.store.saveResult(
                    question = questions[0],
                    strategy = ChunkStrategy.FIXED_SIZE,
                    rawSources = listOf(hit),
                    sources = listOf(hit),
                    baselineAnswer = "Baseline",
                    ragAnswer = "Grounded",
                    rawExpectedSourceRank = 1,
                    expectedSourceRank = 1,
                    selection = ModelSelection("fixture", "fixture-chat"),
                )
                val miss = scoredChunk("miss", distractorSourceId, "Unrelated excerpt")
                fixture.store.saveResult(
                    question = questions[1],
                    strategy = ChunkStrategy.FIXED_SIZE,
                    rawSources = listOf(miss),
                    sources = emptyList(),
                    baselineAnswer = "Baseline",
                    ragAnswer = "Abstain",
                    rawExpectedSourceRank = null,
                    expectedSourceRank = null,
                    selection = ModelSelection("fixture", "fixture-chat"),
                )

                val summary = json(api.get("/api/benchmark/results").body()).jsonObject
                    .getValue("summary").jsonObject.getValue("retrieval").jsonArray
                    .map { it.jsonObject }.single { it.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE" }
                assertEquals(2, summary.getValue("completedQuestions").jsonPrimitive.content.toInt())
                assertEquals(0.1, summary.getValue("rawHitAtK").jsonPrimitive.content.toDouble())
                assertEquals(0.1, summary.getValue("rawMrr").jsonPrimitive.content.toDouble())
                assertEquals(0.1, summary.getValue("enhancedHitAtK").jsonPrimitive.content.toDouble())
                assertEquals(0.1, summary.getValue("enhancedMrr").jsonPrimitive.content.toDouble())
            } finally {
                fixture.close()
            }
        }
    }
    @Test
    fun `HTTP import enforces the pending collection index run limit`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.index.registerSource(
                    StoredSource(
                        SourceRecord(
                            sourceId = java.util.UUID.randomUUID().toString(),
                            name = "existing.txt",
                            type = SourceType.TEXT,
                            sizeBytes = dev.localrag.source.SourceCatalog.MAX_IMPORT_BYTES,
                            status = SourceStatus.PENDING,
                            createdAt = "2026-09-30T00:00:00Z",
                        ),
                        null,
                    ),
                )
                fixture.server.start()
                val api = ApiClient(fixture.server.port)

                val rejected = api.postJson(
                    "/api/imports",
                    """{"files":[{"name":"extra.txt","sizeBytes":1}]}""",
                )
                assertEquals(400, rejected.statusCode())
                assertTrue(rejected.body().contains("512 MiB"), rejected.body())
                val pending = json(api.get("/api/index/pending").body()).jsonObject
                assertEquals(
                    dev.localrag.source.SourceCatalog.MAX_IMPORT_BYTES,
                    pending.getValue("totalBytes").jsonPrimitive.content.toLong(),
                )
                assertEquals(1, json(api.get("/api/sources").body()).jsonArray.size)
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `index snapshot reservation prevents concurrent source deletion before Ollama readiness returns`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val source = importBatch(api, listOf("race.txt" to "orbit route".toByteArray())).getValue("race.txt")
                val sourceId = source.getValue("sourceId").jsonPrimitive.content
                val pending = json(api.get("/api/index/pending").body()).jsonObject
                val totalBytes = pending.getValue("totalBytes").jsonPrimitive.content.toLong()
                val gateForModels = ollama.holdNextTags()
                val gateForIndexing = ollama.holdNextEmbedding()
                val indexRequest = api.postJsonAsync("/api/index", indexPayload(true, listOf(sourceId), totalBytes))

                assertTrue(gateForModels.entered.await(5, TimeUnit.SECONDS))
                val deleteRequest = api.deleteJsonAsync("/api/sources/$sourceId", """{"confirm":true}""")
                gateForModels.release.countDown()

                val accepted = indexRequest.get(5, TimeUnit.SECONDS)
                assertEquals(202, accepted.statusCode())
                val jobId = json(accepted.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertTrue(gateForIndexing.entered.await(5, TimeUnit.SECONDS))
                assertEquals(409, deleteRequest.get(5, TimeUnit.SECONDS).statusCode())
                assertNotNull(fixture.index.source(sourceId))

                gateForIndexing.release.countDown()
                assertEquals("completed", api.awaitCompleted(jobId).getValue("state").jsonPrimitive.content)
                assertEquals("READY", fixture.index.source(sourceId)?.status?.name)
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `benchmark rejects a PDF page marked without searchable text before saving`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val sourceId = fixture.seedSparsePdf()
                val questions = (1..BenchmarkStore.EXPECTED_QUESTION_COUNT).map { number ->
                    BenchmarkQuestion(
                        id = "page-$number",
                        question = "What appears on page two?",
                        expectedFacts = listOf("Page two contains searchable text."),
                        expectedSources = listOf(
                            BenchmarkExpectedSource(sourceId, SourceLocation(pageStart = 2, pageEnd = 2)),
                        ),
                    )
                }
                val saved = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(questions)}}""",
                )
                assertEquals(400, saved.statusCode(), saved.body())
                assertTrue(saved.body().contains("invalid_benchmark"))
                assertTrue(saved.body().contains("ожидаемая страница"))
                val checked = json(api.get("/api/benchmark/questions").body()).jsonObject
                assertTrue(checked.getValue("questions").jsonArray.isEmpty())
                assertFalse(checked.getValue("runnable").jsonPrimitive.content.toBoolean())
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `HTTP benchmark rejects malformed datasets without replacing accepted questions or blocking search`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val bytes = "Orbit markers follow a stable route.\n".toByteArray()
                val source = importBatch(api, listOf("guide.txt" to bytes)).getValue("guide.txt")
                val sourceId = source.getValue("sourceId").jsonPrimitive.content
                val pending = json(api.get("/api/index/pending").body()).jsonObject
                val totalBytes = pending.getValue("totalBytes").jsonPrimitive.content.toLong()
                val indexStart = api.postJson("/api/index", indexPayload(true, listOf(sourceId), totalBytes))
                assertEquals(202, indexStart.statusCode())
                val indexJobId = json(indexStart.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                api.awaitCompleted(indexJobId)

                fun question(id: String) = BenchmarkQuestion(
                    id = id,
                    question = "Where are orbit markers?",
                    expectedFacts = listOf("The route is described."),
                    expectedSources = listOf(
                        BenchmarkExpectedSource(
                            sourceId = sourceId,
                            location = SourceLocation(lineStart = 1, lineEnd = 1),
                            section = "Document body",
                        ),
                    ),
                )

                val accepted = (1..10).map { question("q${it.toString().padStart(2, '0')}") }
                val saved = api.putJson("/api/benchmark/questions", """{"questions":${Json.encodeToString(accepted)}}""")
                assertEquals(200, saved.statusCode(), saved.body())
                assertTrue(json(saved.body()).jsonObject.getValue("runnable").jsonPrimitive.content.toBoolean())

                val benchmarkStart = api.postJson("/api/benchmark/run", "{}")
                assertEquals(202, benchmarkStart.statusCode(), benchmarkStart.body())
                val benchmarkJobId = json(benchmarkStart.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(20, api.awaitCompleted(benchmarkJobId).getValue("filesDone").jsonPrimitive.content.toInt())
                val review = api.postJson(
                    "/api/benchmark/review",
                    """{"questionId":"q01","strategy":"FIXED_SIZE","baselineRating":"FAIL","ragRating":"PASS","note":"Keep this review."}""",
                )
                assertEquals(200, review.statusCode(), review.body())

                val nineQuestions = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(accepted.take(9))}}""",
                )
                assertEquals(400, nineQuestions.statusCode())
                assertTrue(nineQuestions.body().contains("invalid_benchmark"))
                assertTrue(nineQuestions.body().contains("ровно 10 вопросов"))

                val duplicateIds = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(accepted.map { it.copy(id = "q01") })}}""",
                )
                assertEquals(400, duplicateIds.statusCode())
                assertTrue(duplicateIds.body().contains("ID вопросов должны быть уникальными"))

                val malformedSyntax = api.putJson("/api/benchmark/questions", "{")
                assertEquals(400, malformedSyntax.statusCode())
                assertTrue(malformedSyntax.body().contains("invalid_json"))

                val incomplete = api.putJson("/api/benchmark/questions", """{"questions":[{"id":"q01"}]}""")
                assertEquals(400, incomplete.statusCode())
                assertTrue(incomplete.body().contains("invalid_json"))

                val missingSourceId = java.util.UUID.randomUUID().toString()
                val unresolved = accepted.map { question ->
                    question.copy(expectedSources = question.expectedSources.map { it.copy(sourceId = missingSourceId) })
                }
                val unresolvedSources = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(unresolved)}}""",
                )
                assertEquals(400, unresolvedSources.statusCode())
                assertTrue(unresolvedSources.body().contains("invalid_benchmark"))
                assertTrue(unresolvedSources.body().contains("отсутствует"))

                val pendingSourceId = importBatch(
                    api,
                    listOf("pending.txt" to "Unindexed source content.\n".toByteArray()),
                ).getValue("pending.txt").getValue("sourceId").jsonPrimitive.content
                val unindexed = accepted.map { question ->
                    question.copy(expectedSources = question.expectedSources.map { it.copy(sourceId = pendingSourceId) })
                }
                val unindexedResponse = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(unindexed)}}""",
                )
                assertEquals(400, unindexedResponse.statusCode())
                assertTrue(unindexedResponse.body().contains("ещё не проиндексирован"))

                val invalidLocation = accepted.map { question ->
                    question.copy(
                        expectedSources = question.expectedSources.map {
                            it.copy(location = SourceLocation(lineStart = 999, lineEnd = 999))
                        },
                    )
                }
                val invalidLocationResponse = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(invalidLocation)}}""",
                )
                assertEquals(400, invalidLocationResponse.statusCode())
                assertTrue(invalidLocationResponse.body().contains("ожидаемая страница"))
                val incompatibleType = accepted.map { question ->
                    question.copy(
                        expectedSources = question.expectedSources.map {
                            it.copy(location = SourceLocation(pageStart = 1, pageEnd = 1), section = null)
                        },
                    )
                }
                val incompatibleTypeResponse = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(incompatibleType)}}""",
                )
                assertEquals(400, incompatibleTypeResponse.statusCode())
                assertTrue(incompatibleTypeResponse.body().contains("не соответствует типу источника"))

                val persisted = json(api.get("/api/benchmark/questions").body()).jsonObject
                assertTrue(persisted.getValue("runnable").jsonPrimitive.content.toBoolean())
                assertEquals(
                    accepted.map(BenchmarkQuestion::id),
                    persisted.getValue("questions").jsonArray.map { it.jsonObject.getValue("id").jsonPrimitive.content },
                )

                val preservedRows = json(api.get("/api/benchmark/results").body()).jsonObject
                    .getValue("results").jsonArray.map { it.jsonObject }
                assertEquals(20, preservedRows.size)
                val preservedReview = preservedRows.single {
                    it.getValue("questionId").jsonPrimitive.content == "q01" &&
                        it.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE"
                }
                assertEquals("FAIL", preservedReview.getValue("baselineRating").jsonPrimitive.content)
                assertEquals("PASS", preservedReview.getValue("ragRating").jsonPrimitive.content)
                assertEquals("Keep this review.", preservedReview.getValue("note").jsonPrimitive.content)
                val query = api.postJson(
                    "/api/query",
                    """{"question":"Where are orbit markers?","strategy":"FIXED_SIZE","topK":1}""",
                )
                assertEquals(200, query.statusCode(), query.body())
                assertEquals(sourceId, json(query.body()).jsonObject.getValue("rag").jsonObject
                    .getValue("sources").jsonArray.single().jsonObject.getValue("sourceId").jsonPrimitive.content)
                val noMatch = api.postJson(
                    "/api/query",
                    """{"question":"What are penguin breeding habits?","strategy":"FIXED_SIZE","topK":1}""",
                )
                assertEquals(200, noMatch.statusCode(), noMatch.body())
                val ragNoMatch = json(noMatch.body()).jsonObject.getValue("rag").jsonObject
                assertTrue(ragNoMatch.getValue("sources").jsonArray.isEmpty())
                assertEquals(
                    "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник.",
                    ragNoMatch.getValue("answer").jsonPrimitive.content,
                )
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `HTTP retrieval includes threshold boundary and excludes lower scores for both strategies`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val boundary = "Boundary calibration document.\n".toByteArray()
                val below = "Subthreshold calibration document.\n".toByteArray()
                val imported = importBatch(api, listOf("boundary.txt" to boundary, "below.txt" to below))
                val sourceIds = listOf(
                    imported.getValue("boundary.txt").getValue("sourceId").jsonPrimitive.content,
                    imported.getValue("below.txt").getValue("sourceId").jsonPrimitive.content,
                )
                val indexed = api.postJson(
                    "/api/index",
                    indexPayload(true, sourceIds, (boundary.size + below.size).toLong()),
                )
                assertEquals(202, indexed.statusCode(), indexed.body())
                val jobId = json(indexed.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(2, api.awaitCompleted(jobId).getValue("succeeded").jsonPrimitive.content.toInt())

                for (strategy in listOf("FIXED_SIZE", "STRUCTURAL")) {
                    val response = api.postJson(
                        "/api/query",
                        """{"question":"Calibration axis direction?","strategy":"$strategy","topK":20}""",
                    )
                    assertEquals(200, response.statusCode(), response.body())
                    val sources = json(response.body()).jsonObject.getValue("rag").jsonObject
                        .getValue("sources").jsonArray
                    assertEquals(
                        listOf("boundary.txt"),
                        sources.map { it.jsonObject.getValue("source").jsonPrimitive.content },
                    )
                    assertEquals(0.2, sources.single().jsonObject.getValue("score").jsonPrimitive.content.toDouble())
                }
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `HTTP benchmark retains baseline and abstains when enhanced retrieval has no evidence`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                val guide = "Orbit markers follow a stable route.\n".toByteArray()
                val guideRecord = importBatch(api, listOf("guide.txt" to guide)).getValue("guide.txt")
                val guideId = guideRecord.getValue("sourceId").jsonPrimitive.content
                val indexed = api.postJson(
                    "/api/index",
                    indexPayload(true, listOf(guideId), guide.size.toLong()),
                )
                assertEquals(202, indexed.statusCode(), indexed.body())
                val jobId = json(indexed.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(1, api.awaitCompleted(jobId).getValue("succeeded").jsonPrimitive.content.toInt())

                val questions = (1..10).map { number ->
                    BenchmarkQuestion(
                        id = "q${number.toString().padStart(2, '0')}",
                        question = "What are penguin breeding habits?",
                        expectedFacts = listOf("No penguin facts are in the indexed guide."),
                        expectedSources = listOf(
                            BenchmarkExpectedSource(
                                sourceId = guideId,
                                location = SourceLocation(lineStart = 1, lineEnd = 1),
                                section = "Document body",
                            ),
                        ),
                    )
                }
                val saved = api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(questions)}}""",
                )
                assertEquals(200, saved.statusCode(), saved.body())
                assertTrue(json(saved.body()).jsonObject.getValue("runnable").jsonPrimitive.content.toBoolean())
                val generationGate = fixture.holdNextGeneration()
                val started = api.postJson("/api/benchmark/run", "{}")
                assertEquals(202, started.statusCode(), started.body())
                val benchmarkJobId = json(started.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertTrue(generationGate.entered.await(5, TimeUnit.SECONDS))
                val changedSelection = api.putJson(
                    "/api/models/selection",
                    """{"providerId":"fixture","modelId":"fixture-chat-alt"}""",
                )
                assertEquals(200, changedSelection.statusCode(), changedSelection.body())
                generationGate.release.countDown()
                assertEquals(20, api.awaitCompleted(benchmarkJobId).getValue("filesDone").jsonPrimitive.content.toInt())

                val rows = json(api.get("/api/benchmark/results").body()).jsonObject
                    .getValue("results").jsonArray.map { it.jsonObject }
                assertEquals(20, rows.size)
                assertEquals(10, fixture.generationCalls.get())
                assertEquals(0, fixture.rerankCalls.get())
                assertTrue(rows.all {
                    it.getValue("providerId").jsonPrimitive.content == "fixture" &&
                        it.getValue("modelId").jsonPrimitive.content == "fixture-chat"
                })
                assertTrue(fixture.generationSelections.all { it == ModelSelection("fixture", "fixture-chat") })
                assertTrue(rows.all {
                    it.getValue("rawRetrievedSources").jsonArray.single().jsonObject.getValue("source").jsonPrimitive.content == "guide.txt" &&
                        it.getValue("retrievedSources").jsonArray.isEmpty() &&
                        it.getValue("ragAnswer").jsonPrimitive.content ==
                        "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник."
                })
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `HTTP rejects missing cloud credential without calling a generation fallback`() {
        FakeOllama().use { ollama ->
            val fixture = LocalRagFixture(temporaryDirectory, ollama)
            try {
                fixture.server.start()
                val api = ApiClient(fixture.server.port)
                fixture.credentialConfigured.set(false)

                val status = json(api.get("/api/status").body()).jsonObject
                assertTrue(status.getValue("models").jsonObject.getValue("available").jsonPrimitive.content.toBoolean())
                val response = api.postJson(
                    "/api/query",
                    """{"question":"Answer without a cloud credential.","strategy":"FIXED_SIZE"}""",
                )
                assertEquals(502, response.statusCode())
                assertTrue(response.body().contains("cloud_model_unavailable"))
                assertEquals(0, ollama.embeddingCalls.get())
                assertEquals(0, fixture.generationCalls.get())
            } finally {
                fixture.close()
            }
        }
    }

    @Test
    fun `HTTP application restart preserves source benchmark results and ratings`() {
        FakeOllama().use { ollama ->
            val sourceBytes = "Orbit markers follow a stable route.\n".toByteArray()
            val first = LocalRagFixture(temporaryDirectory, ollama)
            val sourceId: String
            try {
                first.server.start()
                val api = ApiClient(first.server.port)
                val source = importBatch(api, listOf("guide.txt" to sourceBytes)).getValue("guide.txt")
                sourceId = source.getValue("sourceId").jsonPrimitive.content
                val pending = json(api.get("/api/index/pending").body()).jsonObject
                val indexStart = api.postJson(
                    "/api/index",
                    indexPayload(true, listOf(sourceId), pending.getValue("totalBytes").jsonPrimitive.content.toLong()),
                )
                val indexJobId = json(indexStart.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                api.awaitCompleted(indexJobId)

                val questions = (1..10).map { number ->
                    BenchmarkQuestion(
                        id = "q${number.toString().padStart(2, '0')}",
                        question = "Where are orbit markers? Case $number",
                        expectedFacts = listOf("The route is described."),
                        expectedSources = listOf(
                            BenchmarkExpectedSource(
                                sourceId = sourceId,
                                location = SourceLocation(lineStart = 1, lineEnd = 1),
                                section = "Document body",
                            ),
                        ),
                    )
                }
                assertEquals(200, api.putJson(
                    "/api/benchmark/questions",
                    """{"questions":${Json.encodeToString(questions)}}""",
                ).statusCode())
                val benchmarkStart = api.postJson("/api/benchmark/run", "{}")
                val benchmarkJobId = json(benchmarkStart.body()).jsonObject.getValue("jobId").jsonPrimitive.content
                assertEquals(20, api.awaitCompleted(benchmarkJobId).getValue("filesDone").jsonPrimitive.content.toInt())
                assertEquals(200, api.postJson(
                    "/api/benchmark/review",
                    """{"questionId":"q01","strategy":"FIXED_SIZE","baselineRating":"FAIL","ragRating":"PASS","note":"Persisted review."}""",
                ).statusCode())
            } finally {
                first.close()
            }

            val reopened = LocalRagFixture(temporaryDirectory, ollama)
            try {
                reopened.server.start()
                val api = ApiClient(reopened.server.port)
                val status = json(api.get("/api/status").body()).jsonObject
                assertEquals(1, status.getValue("sources").jsonObject.getValue("ready").jsonPrimitive.content.toInt())
                assertEquals(sourceId, json(api.get("/api/sources").body()).jsonArray.single()
                    .jsonObject.getValue("sourceId").jsonPrimitive.content)
                val questions = json(api.get("/api/benchmark/questions").body()).jsonObject
                assertTrue(questions.getValue("runnable").jsonPrimitive.content.toBoolean())
                val results = json(api.get("/api/benchmark/results").body()).jsonObject
                assertEquals(20, results.getValue("results").jsonArray.size)
                val review = results.getValue("results").jsonArray.map { it.jsonObject }.single {
                    it.getValue("questionId").jsonPrimitive.content == "q01" &&
                        it.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE"
                }
                assertEquals("FAIL", review.getValue("baselineRating").jsonPrimitive.content)
                assertEquals("PASS", review.getValue("ragRating").jsonPrimitive.content)
                assertEquals("Persisted review.", review.getValue("note").jsonPrimitive.content)
                val query = api.postJson(
                    "/api/query",
                    """{"question":"Where are orbit markers?","strategy":"FIXED_SIZE","topK":1}""",
                )
                assertEquals(200, query.statusCode(), query.body())
                assertEquals(sourceId, json(query.body()).jsonObject.getValue("rag").jsonObject
                    .getValue("sources").jsonArray.single().jsonObject.getValue("sourceId").jsonPrimitive.content)
            } finally {
                reopened.close()
            }
        }
    }

    private fun importBatch(api: ApiClient, files: List<Pair<String, ByteArray>>): Map<String, kotlinx.serialization.json.JsonObject> {
        val metadata = files.joinToString(",") { (name, bytes) -> """{"name":"$name","sizeBytes":${bytes.size}}""" }
        val begin = api.postJson("/api/imports", """{"files":[$metadata]}""")
        assertEquals(201, begin.statusCode())
        val batch = json(begin.body()).jsonObject
        val importId = batch.getValue("importId").jsonPrimitive.content
        val slots = batch.getValue("files").jsonArray.associateBy { it.jsonObject.getValue("name").jsonPrimitive.content }
        files.forEach { (name, bytes) ->
            val slot = slots.getValue(name).jsonObject
            val fileId = slot.getValue("fileId").jsonPrimitive.content
            assertEquals(200, api.putBytes("/api/imports/$importId/files/$fileId", bytes).statusCode())
        }
        val completed = api.postJson("/api/imports/$importId/complete", "{}")
        assertEquals(200, completed.statusCode())
        return json(completed.body()).jsonObject.getValue("files").jsonArray.associate { item ->
            val source = item.jsonObject.getValue("source").jsonObject
            source.getValue("name").jsonPrimitive.content to source
        }
    }

    private fun indexPayload(confirmed: Boolean, sourceIds: List<String>, totalBytes: Long): String =
        Json.encodeToString(buildJsonObject {
            put("confirm", JsonPrimitive(confirmed))
            put("sourceIds", JsonArray(sourceIds.map(::JsonPrimitive)))
            put("totalBytes", JsonPrimitive(totalBytes))
        })

    private fun json(raw: String) = Json.parseToJsonElement(raw)

    private class LocalRagFixture(directory: Path, ollama: FakeOllama) : AutoCloseable {
        val database = directory.resolve("v11.sqlite")
        val sourcesDirectory = directory.resolve("sources")
        val credentialConfigured = AtomicBoolean(true)
        val generationCalls = AtomicInteger()
        val rerankCalls = AtomicInteger()
        val index = SqliteIndexRepository(database)
        val store = BenchmarkStore(database)
        private val chatStore = ChatStore(database)
        private val jobs = JobManager()
        private val api = ollama.client()
        private val embeddings = OllamaEmbeddingPort(api)
        private val selection = ModelSelection("fixture", "fixture-chat")
        private val providers = ProviderCatalog.create(
            listOf(
                ProviderDefinition(
                    "fixture",
                    "Fixture provider",
                    "https://api.deepseek.com",
                    "/chat/completions",
                    "FIXTURE_API_KEY",
                    listOf(
                        ProviderModelDefinition("fixture-chat", "Fixture chat"),
                        ProviderModelDefinition("fixture-chat-alt", "Fixture chat alternate"),
                    ),
                ),
            ),
            selection,
        )
        private val modelConfiguration = ModelConfiguration(
            providers,
            ModelSelectionStore(directory.resolve("model-selection.json"), providers),
            ProviderCredentialSource { if (credentialConfigured.get()) "fixture-secret" else null },
        )
        private val chat = FixtureChat(generationCalls)
        val generationSelections: List<ModelSelection> get() = chat.selections.toList()
        fun holdNextGeneration() = chat.holdNext()
        private val rag = RagService(index, embeddings, chat, FixtureReranker(rerankCalls))
        private val catalog = SourceCatalog(sourcesDirectory, index, store::deleteSourceReferences)
        private val indexing = IndexWorkflow(
            extractor = SourceExtractorRegistry(),
            chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
            embeddings = embeddings,
            index = index,
            sourcesDirectory = sourcesDirectory,
        )
        private val benchmark = BenchmarkRunner(rag, store, index)
        private val chatService = ChatService(chatStore, rag, chat, embeddings)
        private val application = ApplicationService(index, store, catalog, jobs, indexing, rag, benchmark, api, modelConfiguration, chatStore, chatService)
        val server = LocalHttpServer(application, port = 0)
        fun seedSparsePdf(): String {
            val record = SourceRecord(
                sourceId = java.util.UUID.randomUUID().toString(),
                name = "sparse.pdf",
                type = SourceType.PDF,
                sizeBytes = 8,
                status = SourceStatus.PENDING,
                createdAt = "2026-09-29T00:00:00Z",
            )
            index.registerSource(StoredSource(record, null))
            index.markSourceStatus(record.sourceId, SourceStatus.INDEXING)
            val draft = ChunkDraft(
                strategy = ChunkStrategy.FIXED_SIZE,
                chunkId = "${record.sourceId}-fixed-size-000001",
                sourceId = record.sourceId,
                sourceName = record.name,
                section = "Page three",
                location = SourceLocation(pageStart = 1, pageEnd = 3),
                tokenUnits = 6,
                text = "text from pages one and three",
            )
            index.saveChunks(
                record.sourceId,
                listOf(StoredChunk(draft, listOf(1f, 0f, 0f), dev.localrag.ollama.EMBEDDING_MODEL)),
            )
            index.markSourceStatus(record.sourceId, SourceStatus.READY, unsearchablePages = listOf(2))
            return record.sourceId
        }


        override fun close() {
            server.close()
            jobs.close()
            chatStore.close()
            store.close()
            index.close()
        }
    }

    private class ApiClient(port: Int) {
        private val origin = "http://127.0.0.1:$port"
        private val client = HttpClient.newHttpClient()

        fun get(path: String): HttpResponse<String> = send("GET", path, null, null, null)

        fun postJson(path: String, body: String, requestOrigin: String = origin): HttpResponse<String> =
            send("POST", path, body.toByteArray(), "application/json", requestOrigin)

        fun postJsonAsync(path: String, body: String): CompletableFuture<HttpResponse<String>> =
            sendAsync("POST", path, body.toByteArray(), "application/json", origin)

        fun putJson(path: String, body: String): HttpResponse<String> =
            send("PUT", path, body.toByteArray(), "application/json", origin)

        fun putBytes(path: String, body: ByteArray): HttpResponse<String> =
            send("PUT", path, body, "application/octet-stream", origin)

        fun deleteJson(path: String, body: String): HttpResponse<String> =
            send("DELETE", path, body.toByteArray(), "application/json", origin)

        fun deleteJsonAsync(path: String, body: String): CompletableFuture<HttpResponse<String>> =
            sendAsync("DELETE", path, body.toByteArray(), "application/json", origin)

        fun awaitCompleted(jobId: String): kotlinx.serialization.json.JsonObject {
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
            while (System.nanoTime() < deadline) {
                val response = get("/api/jobs/$jobId")
                if (response.statusCode() == 200) {
                    val job = Json.parseToJsonElement(response.body()).jsonObject
                    when (job.getValue("state").jsonPrimitive.content) {
                        "completed" -> {
                            val status = Json.parseToJsonElement(get("/api/status").body()).jsonObject
                            if (status.getValue("activeJob") == kotlinx.serialization.json.JsonNull) return job
                        }
                        "failed" -> error("Local job failed: ${job["error"]}")
                    }
                }
                Thread.sleep(20)
            }
            error("Local job $jobId did not complete within the test deadline.")
        }

        private fun send(
            method: String,
            path: String,
            body: ByteArray?,
            contentType: String?,
            requestOrigin: String?,
        ): HttpResponse<String> = client.send(request(method, path, body, contentType, requestOrigin), HttpResponse.BodyHandlers.ofString())

        private fun sendAsync(
            method: String,
            path: String,
            body: ByteArray?,
            contentType: String?,
            requestOrigin: String?,
        ): CompletableFuture<HttpResponse<String>> =
            client.sendAsync(request(method, path, body, contentType, requestOrigin), HttpResponse.BodyHandlers.ofString())

        private fun request(
            method: String,
            path: String,
            body: ByteArray?,
            contentType: String?,
            requestOrigin: String?,
        ): HttpRequest {
            val builder = HttpRequest.newBuilder(URI("$origin$path"))
            if (requestOrigin != null) builder.header("Origin", requestOrigin)
            if (contentType != null) builder.header("Content-Type", contentType)
            val publisher = body?.let(HttpRequest.BodyPublishers::ofByteArray) ?: HttpRequest.BodyPublishers.noBody()
            builder.method(method, publisher)
            return builder.build()
        }
    }

    private class FixtureChat(private val calls: AtomicInteger) : ChatPort, dev.localrag.chat.ChatTurnGenerator {
        val selections = CopyOnWriteArrayList<ModelSelection>()
        private val nextGate = AtomicReference<Gate?>()

        fun holdNext() = Gate().also(nextGate::set)

        override fun answer(selection: ModelSelection, question: String, context: List<ScoredChunk>): String {
            calls.incrementAndGet()
            selections += selection
            val gate = nextGate.getAndSet(null)
            if (gate != null) {
                gate.entered.countDown()
                check(gate.release.await(10, TimeUnit.SECONDS)) { "The test did not release generation." }
            }
            return if (context.isEmpty()) "Baseline answer without indexed sources."
            else "RAG answer grounded in imported context."
        }
        override fun completeTurn(
            selection: ModelSelection,
            request: dev.localrag.chat.ChatTurnRequest,
        ): dev.localrag.chat.ChatTurnCompletion =
            dev.localrag.chat.ChatTurnCompletion("Synthetic fixture answer.", dev.localrag.chat.ChatTaskStateDelta())
    }

    private class FixtureReranker(private val calls: AtomicInteger) : RerankPort {
        override fun rerank(selection: ModelSelection, question: String, candidateTexts: List<String>): List<Int> {
            calls.incrementAndGet()
            return candidateTexts.indices.toList()
        }
    }
    private class FakeOllama : AutoCloseable {
        private val executor: ExecutorService = Executors.newCachedThreadPool()

        val embeddingCalls = AtomicInteger()
        private val tagsGate = AtomicReference<Gate?>()
        private val embeddingGate = AtomicReference<Gate?>()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeOllama.executor
            createContext("/api/tags") { exchange ->
                waitForGate(tagsGate)
                val names = listOf("embeddinggemma:300m")
                respond(exchange, buildJsonObject {
                    put("models", JsonArray(names.map { name -> buildJsonObject { put("name", JsonPrimitive(name)) } }))
                }.toString())
            }
            createContext("/api/embed") { exchange ->
                waitForGate(embeddingGate)
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader().use { it.readText() }).jsonObject
                val input = request.getValue("input").jsonArray.map { it.jsonPrimitive.content }
                embeddingCalls.addAndGet(input.size)
                val vectors = input.map { text ->
                    val normalized = text.lowercase()
                    val values = when {
                        "calibration axis direction" in normalized -> listOf(1.0) + List(24) { 0.0 }
                        "boundary calibration document" in normalized -> List(25) { 0.2 }
                        "subthreshold calibration document" in normalized -> listOf(0.19) + List(24) { 0.2 }
                        "orbit" in normalized -> listOf(1.0, 0.0, 0.0)
                        "copper" in normalized -> listOf(0.0, 1.0, 0.0)
                        else -> listOf(0.0, 0.0, 1.0)
                    }
                    buildJsonArray { values.forEach { add(JsonPrimitive(it)) } }
                }
                respond(exchange, buildJsonObject { put("embeddings", JsonArray(vectors)) }.toString())
            }
        }

        init {
            server.start()
        }

        fun client() = OllamaApi(URI("http://127.0.0.1:${server.address.port}"))

        fun holdNextTags() = Gate().also(tagsGate::set)

        fun holdNextEmbedding() = Gate().also(embeddingGate::set)

        override fun close() {
            tagsGate.getAndSet(null)?.release?.countDown()
            embeddingGate.getAndSet(null)?.release?.countDown()
            server.stop(0)
            executor.shutdownNow()
        }

        private fun waitForGate(reference: AtomicReference<Gate?>) {
            val gate = reference.getAndSet(null) ?: return
            gate.entered.countDown()
            check(gate.release.await(10, TimeUnit.SECONDS)) { "The test did not release a fake Ollama request." }
        }

        private fun respond(exchange: HttpExchange, body: String) {
            val bytes = body.toByteArray()
            exchange.responseHeaders.set("Content-Type", "application/json")
            exchange.sendResponseHeaders(200, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
    }
}
