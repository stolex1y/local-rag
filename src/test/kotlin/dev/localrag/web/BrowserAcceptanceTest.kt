package dev.localrag.web

import com.microsoft.playwright.BrowserType
import com.microsoft.playwright.Playwright
import com.microsoft.playwright.Page
import com.microsoft.playwright.assertions.PlaywrightAssertions.assertThat
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
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.ByteArrayOutputStream
import java.net.InetSocketAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.charset.StandardCharsets.US_ASCII
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer

@Tag("browser")
class BrowserAcceptanceTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun fullSourceAndBenchmarkJourneyPreservesStateAndDeduplicatesContent() {
        FakeOllama().use { fake ->
            RequestProbe().use { probe ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("app-data"))
            val index = SqliteIndexRepository(paths.database)
            try {
                val benchmarkStore = BenchmarkStore(paths.database)
                try {
                    val jobs = JobManager()
                    try {
                        val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
                        val ollama = OllamaApi(fake.uri)
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
                        LocalHttpServer(application, 0).use { server ->
                            server.start()
                            exerciseUserJourney(fake, probe, server.port, paths.sources)
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
        }
    }

    @Test
    fun applicationRestartRestoresSourcesAndBenchmarkReviewsInBrowser() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("restart-app-data"))
            val sourceFile = temporaryDirectory.resolve("restart-note.txt")
            Files.writeString(sourceFile, "Orbit markers follow a stable route.\n", UTF_8)
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    var fixture: BrowserAppFixture? = BrowserAppFixture(paths, fake)
                    try {
                        val first = fixture!!
                        page.navigate("http://127.0.0.1:${first.server.port}/")
                        assertThat(page.locator("#model-note")).containsText("Ollama доступен")
                        page.onDialog { it.accept() }

                        page.locator("#source-files").setInputFiles(sourceFile)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        assertThat(page.locator("#source-list")).containsText("restart-note.txt")
                        val firstSource = Json.parseToJsonElement(get("http://127.0.0.1:${first.server.port}/api/sources"))
                            .jsonArray.single().jsonObject
                        val sourceId = firstSource.getValue("sourceId").jsonPrimitive.content

                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")
                        val questionsFile = temporaryDirectory.resolve("restart-questions.json")
                        Files.writeString(
                            questionsFile,
                            benchmarkQuestions(sourceId, 1, "Orbit markers follow a stable route."),
                            UTF_8,
                        )
                        page.locator("#benchmark-file").setInputFiles(questionsFile)
                        page.locator("#benchmark-upload").click()
                        assertThat(page.locator("#benchmark-error")).containsText("Набор сохранён")
                        page.locator("#benchmark-run").click()
                        assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                        assertEquals(20, page.locator("#benchmark-results .strategy-card").count())

                        val rating = page.locator("#benchmark-results .rating-form").first()
                        rating.locator("select[name=\"baselineRating\"]").selectOption("FAIL")
                        rating.locator("select[name=\"ragRating\"]").selectOption("PASS")
                        rating.locator("input[name=\"note\"]").fill("Restart survives.")
                        page.waitForResponse("**/api/benchmark/review") {
                            rating.locator("button[type=\"submit\"]").click()
                        }
                        assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                        assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")

                        first.close()
                        fixture = null
                        fixture = BrowserAppFixture(paths, fake)
                        val reopened = fixture!!
                        page.navigate("http://127.0.0.1:${reopened.server.port}/")
                        assertThat(page.locator("#source-list")).containsText("restart-note.txt")
                        assertEquals(1, page.locator("#source-list .source-row").count())
                        val restoredSource = Json.parseToJsonElement(get("http://127.0.0.1:${reopened.server.port}/api/sources"))
                            .jsonArray.single().jsonObject
                        assertEquals(sourceId, restoredSource.getValue("sourceId").jsonPrimitive.content)
                        Files.list(paths.sources).use { assertEquals(1L, it.count()) }
                        assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                        assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")
                        assertEquals(
                            "Restart survives.",
                            page.locator("#benchmark-results .rating-form").first()
                                .locator("input[name=\"note\"]").inputValue(),
                        )

                        page.locator("#question").fill("Where are orbit markers?")
                        page.locator("#query-button").click()
                        assertThat(page.locator("#query-results")).containsText("restart-note.txt")
                        assertThat(page.locator("#query-results")).containsText("строки 1")
                    } finally {
                        fixture?.close()
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }

    private fun exerciseUserJourney(fake: FakeOllama, probe: RequestProbe, port: Int, sourcesDirectory: Path) {
        val firstDuplicate = temporaryDirectory.resolve("first/version-note.txt")
        val secondDuplicate = temporaryDirectory.resolve("second/version-note.txt")
        Files.createDirectories(firstDuplicate.parent)
        Files.createDirectories(secondDuplicate.parent)
        Files.writeString(firstDuplicate, "Calibration V-01: beacon threshold is 7 units.\n", UTF_8)
        Files.writeString(secondDuplicate, "Calibration V-02: beacon threshold is 9 units.\n", UTF_8)
        val identicalCopy = temporaryDirectory.resolve("same-bytes-copy.txt")
        Files.writeString(identicalCopy, "Calibration V-01: beacon threshold is 7 units.\n", UTF_8)
        val html = temporaryDirectory.resolve("greenhouse.html")
        Files.writeString(
            html,
            """
                <html><head><title>Greenhouse telemetry</title>
                <style>.hidden { display: none; } STYLE_ONLY_SECRET</style></head>
                <body><h1>Greenhouse telemetry</h1><p>Humidity target is 64 percent.</p>
                <script>const hidden = "SCRIPT_ONLY_SECRET"; fetch("${probe.url}/script");</script>
                <img src="${probe.url}/remote-image.png"></body></html>
            """.trimIndent(),
            UTF_8,
        )
        val code = temporaryDirectory.resolve("copper-calibration.kt")
        Files.writeString(code, "object CopperCalibration { const val LIMIT = 19 }\n", UTF_8)
        val invalidUtf8 = temporaryDirectory.resolve("invalid-utf8.md")
        Files.write(invalidUtf8, byteArrayOf(0xc3.toByte(), 0x28))
        val binary = temporaryDirectory.resolve("capture.bin")
        Files.write(binary, byteArrayOf(0, 1, 2))
        val pdf = temporaryDirectory.resolve("synthetic-field-guide.pdf")
        writeSyntheticPdf(pdf)
        val reference = Path.of("examples/synthetic-observatory.md").toAbsolutePath()
        val referenceLines = Files.readAllLines(reference, UTF_8)
        val referenceLine = referenceLines.indexOfFirst { "Calibration record ORBIT-017" in it } + 1
        assertTrue(referenceLine > 0)
        assertTrue(Files.readString(reference, UTF_8).trim().split(Regex("\\s+")).size >= 6_500)

        val origin = "http://127.0.0.1:$port"
        val playwright = Playwright.create()
        try {
            val browser = playwright.chromium().launch(
                BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
            )
            try {
                val page = browser.newPage()
                page.setDefaultTimeout(30_000.0)
                page.navigate(origin)
                assertThat(page.locator("#model-note")).containsText("Ollama доступен")
                assertFalse(page.locator("body").innerText().contains("LTE", ignoreCase = true))

                val acceptNextDialog = AtomicBoolean(false)
                val dialogMessages = CopyOnWriteArrayList<String>()
                page.onDialog { dialog ->
                    dialogMessages.add(dialog.message())
                    if (acceptNextDialog.compareAndSet(true, false)) dialog.accept() else dialog.dismiss()
                }
                page.locator("#source-files").setInputFiles(arrayOf(firstDuplicate, identicalCopy, secondDuplicate, html, code, invalidUtf8, binary, pdf, reference))
                page.locator("#upload-button").click()
                assertThat(page.locator("#upload-success")).containsText("6 файл(ов)")
                assertThat(page.locator("#upload-success")).containsText("invalid-utf8.md")
                assertThat(page.locator("#upload-success")).containsText("capture.bin")

                val duplicateRows = page.locator("#source-list .source-row").all()
                    .filter { it.locator(".source-name").textContent() == "version-note.txt" }
                assertEquals(2, duplicateRows.size)
                val duplicateRowIds = duplicateRows.mapNotNull { row ->
                    Regex("ID ([0-9a-f-]{12})").find(row.locator(".source-meta").textContent().orEmpty())?.groupValues?.get(1)
                }
                assertEquals(2, duplicateRowIds.distinct().size)

                val sourceRecords = Json.parseToJsonElement(get("$origin/api/sources")).jsonArray.map { it.jsonObject }
                val duplicateIds = sourceRecords.filter { it.getValue("name").jsonPrimitive.content == "version-note.txt" }
                    .map { it.getValue("sourceId").jsonPrimitive.content }
                assertEquals(2, duplicateIds.size)
                assertNotEquals(duplicateIds[0], duplicateIds[1])
                assertEquals(8, sourceRecords.size)
                assertTrue(sourceRecords.none { it.getValue("name").jsonPrimitive.content == "same-bytes-copy.txt" })
                val pendingFiles = Json.parseToJsonElement(get("$origin/api/index/pending")).jsonObject.getValue("files").jsonArray
                assertEquals(6, pendingFiles.size, "Unexpected pending files: $pendingFiles")
                assertTrue(page.locator("#index-button").isEnabled(), "Index button disabled: sources=$sourceRecords")
                page.waitForResponse("**/api/index/pending") {
                    page.locator("#index-button").click()
                }
                val cancelledIndex = awaitDialog(page, dialogMessages)
                assertTrue(cancelledIndex.contains("6 файлов"))
                assertTrue(cancelledIndex.contains("greenhouse.html"))
                assertEquals(0, fake.embeddingCalls.get())
                assertThat(page.locator("#source-list")).containsText("ожидает индексации")

                acceptNextDialog.set(true)
                page.locator("#index-button").click()
                assertThat(page.locator("#index-success")).containsText("Индексация завершена")
                assertTrue(awaitDialog(page, dialogMessages).startsWith("Начать индексацию"))
                assertThat(page.locator("#source-list")).containsText("страницы без текста: 2")

                val chatStart = fake.chatMessages.size
                page.locator("#question").fill("What is the greenhouse humidity target?")
                page.locator("#query-button").click()
                val queryResults = page.locator("#query-results")
                assertThat(queryResults).containsText("Baseline · без коллекции")
                assertThat(queryResults).containsText("RAG · с найденными фрагментами")
                assertThat(queryResults).containsText("greenhouse.html")
                assertThat(queryResults).containsText("Greenhouse telemetry")
                val queryMessages = fake.chatMessages.drop(chatStart)
                assertEquals(2, queryMessages.size)
                assertTrue(queryMessages.first().none { "Humidity target is 64 percent" in it })
                assertTrue(queryMessages.last().any { "Humidity target is 64 percent" in it })
                assertTrue(queryMessages.flatten().none { "SCRIPT_ONLY_SECRET" in it || "STYLE_ONLY_SECRET" in it })
                val pdfChatStart = fake.chatMessages.size
                page.locator("#question").fill("What does the PDF field note say about conductivity?")
                page.locator("#query-button").click()
                assertThat(queryResults).containsText("Baseline · без коллекции")
                assertThat(queryResults).containsText("RAG · с найденными фрагментами")
                assertThat(queryResults).containsText("synthetic-field-guide.pdf")
                assertThat(queryResults).containsText("стр. 1")
                val pdfMessages = fake.chatMessages.drop(pdfChatStart)
                assertEquals(2, pdfMessages.size)
                assertTrue(pdfMessages.first().none { "conductivity is 21 units" in it })
                assertTrue(pdfMessages.last().any { "conductivity is 21 units" in it })
                assertEquals(0, probe.requests.get())

                page.locator("#top-k").fill("10")
                for (strategy in listOf("FIXED_SIZE", "STRUCTURAL")) {
                    page.locator("#strategy").selectOption(strategy)
                    val codeChatStart = fake.chatMessages.size
                    page.locator("#question").fill("What copper calibration limit is recorded?")
                    page.locator("#query-button").click()
                    assertThat(queryResults).containsText("copper-calibration.kt")
                    assertThat(queryResults).containsText("строки 1")
                    val codeMessages = fake.chatMessages.drop(codeChatStart)
                    assertEquals(2, codeMessages.size)
                    assertTrue(codeMessages.last().any { "LIMIT = 19" in it })

                    val textChatStart = fake.chatMessages.size
                    page.locator("#question").fill("What beacon threshold is recorded in the note?")
                    page.locator("#query-button").click()
                    assertThat(queryResults).containsText("version-note.txt")
                    assertThat(queryResults).containsText("строки 1")
                    val textMessages = fake.chatMessages.drop(textChatStart)
                    assertEquals(2, textMessages.size)
                    assertTrue(textMessages.last().any { "beacon threshold" in it })
                }
                assertEquals(0, probe.requests.get())
                val emptyContextChatStart = fake.chatMessages.size
                page.locator("#question").fill("What are penguin breeding habits?")
                page.locator("#query-button").click()
                assertThat(queryResults).containsText("По этому ответу нет проверенных цитат.")
                assertThat(queryResults).containsText("В коллекции не найдено подходящих фрагментов.")
                val emptyContextMessages = fake.chatMessages.drop(emptyContextChatStart)
                assertEquals(2, emptyContextMessages.size)
                assertTrue(emptyContextMessages.last().first().contains("Локальные фрагменты не переданы для подтверждения ответа"))
                assertTrue(emptyContextMessages.last().first().contains("не выдумывай цитаты"))
                assertTrue(emptyContextMessages.last().last().contains("penguin"))

                page.locator("button[aria-label^=\"Удалить источник version-note.txt\"]").first().click()
                assertTrue(awaitDialog(page, dialogMessages).contains(duplicateIds.first()))
                assertEquals(2, page.locator("#source-list .source-row").all().count { it.locator(".source-name").textContent() == "version-note.txt" })

                acceptNextDialog.set(true)
                page.locator("button[aria-label^=\"Удалить источник version-note.txt\"]").first().click()
                assertTrue(awaitDialog(page, dialogMessages).contains(duplicateIds.first()))
                assertThat(page.locator("#source-list .source-row")).hasCount(7)
                assertEquals(1, page.locator("#source-list .source-row").all().count { it.locator(".source-name").textContent() == "version-note.txt" })
                assertTrue(Json.parseToJsonElement(get("$origin/api/sources")).jsonArray.none {
                    it.jsonObject.getValue("sourceId").jsonPrimitive.content == duplicateIds.first()
                })
                assertTrue(!Files.exists(sourcesDirectory.resolve("${duplicateIds.first()}.source")))

                val referenceId = sourceRecords.single { it.getValue("name").jsonPrimitive.content == "synthetic-observatory.md" }
                    .getValue("sourceId").jsonPrimitive.content
                val threshold = Regex("threshold to (\\d+ units)").find(referenceLines[referenceLine - 1])!!.groupValues[1]
                val benchmarkFile = temporaryDirectory.resolve("questions.json")
                Files.writeString(benchmarkFile, benchmarkQuestions(referenceId, referenceLine, "ORBIT-017 threshold is $threshold"), UTF_8)
                page.locator("#benchmark-file").setInputFiles(benchmarkFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).containsText("Набор сохранён")

                page.locator("#benchmark-run").click()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                val expectedSources = page.locator("#benchmark-results .gold p").allInnerTexts()
                assertTrue(expectedSources.any { it.contains("synthetic-observatory.md [") })
                assertEquals(20, page.locator("#benchmark-results .strategy-card").count())
                val rating = page.locator("#benchmark-results .rating-form").first()
                rating.locator("select[name=\"baselineRating\"]").selectOption("FAIL")
                rating.locator("select[name=\"ragRating\"]").selectOption("PASS")
                rating.locator("input[name=\"note\"]").fill("Synthetic review.")
                rating.locator("button[type=\"submit\"]").click()
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")
                val missingSourceId = "00000000-0000-4000-8000-000000000000"
                val unresolvedSourceFile = temporaryDirectory.resolve("questions-missing-source.json")
                Files.writeString(
                    unresolvedSourceFile,
                    benchmarkQuestions(
                        missingSourceId,
                        referenceLine,
                        "The referenced source is unavailable.",
                        questionText = "What is in the missing source?",
                    ),
                    UTF_8,
                )
                page.locator("#benchmark-file").setInputFiles(unresolvedSourceFile)
                page.waitForResponse("**/api/benchmark/questions") {
                    page.locator("#benchmark-upload").click()
                }
                assertThat(page.locator("#benchmark-error")).containsText("отсутствует")
                assertTrue(page.locator("#benchmark-run").isEnabled())
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")
                val unresolvedQuestionsState = Json.parseToJsonElement(get("$origin/api/benchmark/questions")).jsonObject
                assertTrue(unresolvedQuestionsState.getValue("runnable").jsonPrimitive.content.toBoolean())
                assertEquals(10, unresolvedQuestionsState.getValue("questions").jsonArray.size)
                assertEquals(
                    referenceId,
                    unresolvedQuestionsState.getValue("questions").jsonArray.first().jsonObject
                        .getValue("expectedSources").jsonArray.first().jsonObject
                        .getValue("sourceId").jsonPrimitive.content,
                )
                val preservedRows = Json.parseToJsonElement(get("$origin/api/benchmark/results")).jsonObject
                    .getValue("results").jsonArray
                assertEquals(20, preservedRows.size)
                val preservedReview = preservedRows.single {
                    it.jsonObject.getValue("questionId").jsonPrimitive.content == "q01" &&
                        it.jsonObject.getValue("strategy").jsonPrimitive.content == "FIXED_SIZE"
                }.jsonObject
                assertEquals("FAIL", preservedReview.getValue("baselineRating").jsonPrimitive.content)
                assertEquals("PASS", preservedReview.getValue("ragRating").jsonPrimitive.content)
                assertEquals("Synthetic review.", preservedReview.getValue("note").jsonPrimitive.content)
                val nineQuestionFile = temporaryDirectory.resolve("nine-questions.json")
                Files.writeString(
                    nineQuestionFile,
                    benchmarkQuestions(referenceId, referenceLine, "ORBIT-017 threshold is $threshold", questionCount = 9),
                    UTF_8,
                )
                page.locator("#benchmark-file").setInputFiles(nineQuestionFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).containsText("ровно 10 вопросов")
                assertTrue(page.locator("#benchmark-run").isEnabled())
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")

                val malformedFile = temporaryDirectory.resolve("malformed-questions.json")
                Files.writeString(malformedFile, """{"questions":[{"id":"q01"}]}""", UTF_8)
                page.locator("#benchmark-file").setInputFiles(malformedFile)
                page.waitForResponse("**/api/benchmark/questions") {
                    page.locator("#benchmark-upload").click()
                }
                assertThat(page.locator("#benchmark-error")).not().hasText("")
                assertTrue(page.locator("#benchmark-run").isEnabled())
                val malformedSyntaxFile = temporaryDirectory.resolve("malformed-syntax.json")
                Files.writeString(malformedSyntaxFile, "{\"questions\":[", UTF_8)
                page.locator("#benchmark-file").setInputFiles(malformedSyntaxFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).not().hasText("")
                assertTrue(page.locator("#benchmark-run").isEnabled())
                val acceptedQuestions = Json.parseToJsonElement(get("$origin/api/benchmark/questions")).jsonObject
                assertTrue(acceptedQuestions.getValue("runnable").jsonPrimitive.content.toBoolean())
                assertEquals(10, acceptedQuestions.getValue("questions").jsonArray.size)
                assertTrue(page.locator("#query-button").isEnabled())
                page.locator("#question").fill("What is the greenhouse humidity target?")
                page.locator("#query-button").click()
                assertThat(page.locator("#query-results")).containsText("greenhouse.html")


                fake.failChat.set(true)
                page.locator("#benchmark-run").click()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark не завершён")
                assertThat(page.locator("#benchmark-error")).containsText("результаты предыдущих запусков")
                assertTrue(page.locator("#benchmark-results").textContent().orEmpty().isNotBlank())

                page.reload()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark не завершён")
                assertThat(page.locator("#benchmark-error")).containsText("результаты предыдущих запусков")
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")

                fake.failChat.set(false)
                val noEvidenceFile = temporaryDirectory.resolve("questions-no-evidence.json")
                Files.writeString(
                    noEvidenceFile,
                    benchmarkQuestions(
                        referenceId,
                        referenceLine,
                        "The synthetic collection contains no penguin facts.",
                        questionText = "What are penguin breeding habits?",
                    ),
                    UTF_8,
                )
                page.locator("#benchmark-file").setInputFiles(noEvidenceFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).containsText("Набор сохранён")
                page.locator("#benchmark-run").click()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                assertThat(page.locator("#benchmark-results"))
                    .containsText("В коллекции не найдено подходящих фрагментов.")
                val noEvidenceRows = Json.parseToJsonElement(get("$origin/api/benchmark/results"))
                    .jsonObject.getValue("results").jsonArray
                assertEquals(20, noEvidenceRows.size)
                assertTrue(noEvidenceRows.all {
                    it.jsonObject.getValue("retrievedSources").jsonArray.isEmpty() &&
                        it.jsonObject.getValue("ragAnswer").jsonPrimitive.content
                            .startsWith("В коллекции не найдено подходящих фрагментов.")
                })
                assertThat(page.locator("#source-list")).containsText("greenhouse.html")
                assertTrue(!page.locator("#source-list").textContent().orEmpty().contains("same-bytes-copy.txt"))
                acceptNextDialog.set(true)
                val referenceRowAction = page.locator("button[aria-label^=\"Удалить источник synthetic-observatory.md\"]")
                page.waitForResponse("**/api/sources/$referenceId") {
                    referenceRowAction.click()
                }
                assertTrue(awaitDialog(page, dialogMessages).contains(referenceId))
                assertThat(page.locator("#source-list .source-row")).hasCount(6)
                assertTrue(!page.locator("#source-list").textContent().orEmpty().contains("synthetic-observatory.md"))
                assertThat(page.locator("#benchmark-state")).containsText("источник $referenceId отсутствует.")
                assertTrue(page.locator("#benchmark-run").isDisabled())

                val persistedReferenceIds = Json.parseToJsonElement(get("$origin/api/benchmark/questions"))
                    .jsonObject.getValue("questions").jsonArray.flatMap { question ->
                        question.jsonObject.getValue("expectedSources").jsonArray.map { expectedSource ->
                            expectedSource.jsonObject.getValue("sourceId").jsonPrimitive.content
                        }
                    }
                assertEquals(
                    List(10) { referenceId },
                    persistedReferenceIds,
                    "Deleting a referenced source must not rewrite benchmark question references.",
                )
                val completedChatCount = fake.chatMessages.size
                val completedEmbeddingCalls = fake.embeddingCalls.get()
                fake.missingChatModel.set(true)
                page.reload()
                assertThat(page.locator("#model-note")).containsText("Локальные модели недоступны")
                assertTrue(page.locator("#query-button").isDisabled())
                assertEquals(completedChatCount, fake.chatMessages.size)
                assertEquals(completedEmbeddingCalls, fake.embeddingCalls.get())
            } finally {
                browser.close()
            }
        } finally {
            playwright.close()
        }
    }

    private fun awaitDialog(page: Page, messages: List<String>): String {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (messages.isEmpty() && System.nanoTime() < deadline) page.waitForTimeout(10.0)
        assertTrue(messages.isNotEmpty(), "Expected a browser confirmation dialog.")
        return messages.last()
    }

    private fun benchmarkQuestions(
        sourceId: String,
        line: Int,
        expectedFact: String,
        questionText: String = "What threshold is recorded for ORBIT-017?",
        questionCount: Int = 10,
    ): String {
        val questions = (1..questionCount).joinToString(",") { number ->
            val id = "q${number.toString().padStart(2, '0')}"
            """{"id":"$id","question":"Question $number: $questionText","expectedFacts":["$expectedFact"],"expectedSources":[{"sourceId":"$sourceId","location":{"lineStart":$line,"lineEnd":$line}}]}"""
        }
        return "{\"questions\":[$questions]}"
    }

    private fun writeSyntheticPdf(path: Path) {
        fun stream(content: String): String {
            val length = content.toByteArray(US_ASCII).size
            return "<< /Length $length >>\nstream\n$content\nendstream"
        }

        val objects = listOf(
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            stream("BT /F1 18 Tf 72 720 Td (PDF field note: conductivity is 21 units.) Tj ET"),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 7 0 R >>",
            stream(""),
        )
        val output = ByteArrayOutputStream()
        fun write(text: String) {
            output.write(text.toByteArray(US_ASCII))
        }

        write("%PDF-1.4\n")
        val offsets = mutableListOf(0)
        objects.forEachIndexed { index, objectBody ->
            offsets.add(output.size())
            write("${index + 1} 0 obj\n$objectBody\nendobj\n")
        }
        val crossReferenceOffset = output.size()
        write("xref\n0 ${objects.size + 1}\n0000000000 65535 f \n")
        offsets.drop(1).forEach { offset ->
            write(String.format(Locale.ROOT, "%010d 00000 n \n", offset))
        }
        write("trailer\n<< /Size ${objects.size + 1} /Root 1 0 R >>\nstartxref\n$crossReferenceOffset\n%%EOF\n")
        Files.write(path, output.toByteArray())
    }

    private fun get(url: String): String {
        val response = HttpClient.newHttpClient().send(
            HttpRequest.newBuilder(URI(url)).GET().build(),
            HttpResponse.BodyHandlers.ofString(UTF_8),
        )
        assertEquals(200, response.statusCode())
        return response.body()
    }

    private class BrowserAppFixture(paths: LocalRagPaths, fake: FakeOllama) : AutoCloseable {
        private val index = SqliteIndexRepository(paths.database)
        private val benchmarkStore = BenchmarkStore(paths.database)
        private val jobs = JobManager()
        private val ollama = OllamaApi(fake.uri)
        private val embeddings = OllamaEmbeddingPort(ollama)
        private val rag = RagService(index, embeddings, OllamaChatPort(ollama))
        private val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
        private val indexing = IndexWorkflow(
            extractor = SourceExtractorRegistry(),
            chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
            embeddings = embeddings,
            index = index,
            sourcesDirectory = paths.sources,
        )
        private val benchmark = BenchmarkRunner(rag, benchmarkStore, index)
        private val application = ApplicationService(index, benchmarkStore, catalog, jobs, indexing, rag, benchmark, ollama)
        val server = LocalHttpServer(application, 0)

        init {
            server.start()
        }

        override fun close() {
            server.close()
            jobs.close()
            benchmarkStore.close()
            index.close()
        }
    }

    private class FakeOllama : AutoCloseable {
        val failChat = AtomicBoolean(false)
        val missingChatModel = AtomicBoolean(false)
        val embeddingCalls = AtomicInteger()
        val chatMessages = CopyOnWriteArrayList<List<String>>()
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeOllama.executor
            createContext("/api/tags") { exchange ->
                val models = if (missingChatModel.get()) {
                    """{"models":[{"name":"embeddinggemma:300m"}]}"""
                } else {
                    """{"models":[{"name":"embeddinggemma:300m"},{"name":"qwen2.5:0.5b-instruct"}]}"""
                }
                respond(exchange, 200, models)
            }
            createContext("/api/embed") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                val inputs = request.getValue("input").jsonArray.map { it.jsonPrimitive.content }
                embeddingCalls.addAndGet(inputs.size)
                val embeddings = buildJsonArray {
                    inputs.forEach { input ->
                        val vector = when {
                            "penguin" in input.lowercase() -> listOf(0.0, 0.0, 0.0, 1.0)
                            "orbit" in input.lowercase() -> listOf(1.0, 0.0, 0.0, 0.0)
                            "greenhouse" in input.lowercase() -> listOf(0.0, 1.0, 0.0, 0.0)
                            "pdf" in input.lowercase() -> listOf(0.0, 0.0, 1.0, 0.0)
                            else -> listOf(1.0, 1.0, 0.0, 0.0)
                        }
                        add(buildJsonArray { vector.forEach { add(JsonPrimitive(it)) } })
                    }
                }
                respond(exchange, 200, buildJsonObject { put("embeddings", embeddings) }.toString())
            }
            createContext("/api/chat") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                val messages = request.getValue("messages").jsonArray.map {
                    it.jsonObject.getValue("content").jsonPrimitive.content
                }
                chatMessages.add(messages)
                if (failChat.get()) {
                    respond(exchange, 503, """{"error":"Synthetic local model failure"}""")
                } else {
                    respond(exchange, 200, """{"message":{"content":"Synthetic answer."}}""")
                }
            }
        }

        val uri: URI get() = URI("http://127.0.0.1:${server.address.port}")

        init {
            server.start()
        }

        override fun close() {
            server.stop(0)
            executor.shutdownNow()
        }

        private fun respond(exchange: HttpExchange, status: Int, body: String) {
            val bytes = body.toByteArray(UTF_8)
            exchange.responseHeaders.set("Content-Type", "application/json; charset=utf-8")
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
    }

    private class RequestProbe : AutoCloseable {
        val requests = AtomicInteger()
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@RequestProbe.executor
            createContext("/") { exchange ->
                requests.incrementAndGet()
                exchange.sendResponseHeaders(204, -1)
                exchange.close()
            }
        }

        val url: String get() = "http://127.0.0.1:${server.address.port}"

        init {
            server.start()
        }

        override fun close() {
            server.stop(0)
            executor.shutdownNow()
        }
    }
}
