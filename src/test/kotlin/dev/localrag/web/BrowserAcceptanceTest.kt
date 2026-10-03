package dev.localrag.web

import dev.localrag.chat.ChatStore
import dev.localrag.chat.ChatService

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
import dev.localrag.ollama.OllamaEmbeddingPort
import dev.localrag.source.LocalRagPaths
import dev.localrag.source.SourceCatalog
import dev.localrag.generation.ChatCompletionsApi
import dev.localrag.generation.ModelConfiguration
import dev.localrag.generation.ModelSelectionStore
import dev.localrag.generation.ProviderCatalog
import dev.localrag.generation.ProviderCredentialSource
import dev.localrag.generation.ProviderDefinition
import dev.localrag.generation.ProviderModelDefinition
import dev.localrag.domain.ModelSelection
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
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
import java.sql.DriverManager
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer

@Tag("browser")
class BrowserAcceptanceTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun chatSessionDisplaysRetrievedCitationsAndRestoresDirectUrl() {
        withChatBrowser { page, app, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val sessionId = createChatSession(page)
            val turnResponse = submitChatQuestion(page, "greenhouse payment date?")
            val message = turnResponse.getValue("message").jsonObject
            val citationData = message.getValue("documentCitations").jsonArray.single().jsonObject
            val sources = Json.parseToJsonElement(
                page.request().get("http://127.0.0.1:${app.server.port}/api/sources").text(),
            ).jsonArray.map { it.jsonObject }
            val expectedSourceId = sources.single {
                it.getValue("name").jsonPrimitive.content == "greenhouse-chat.md"
            }.getValue("sourceId").jsonPrimitive.content
            assertEquals("greenhouse-chat.md", citationData.getValue("source").jsonPrimitive.content)
            assertEquals(expectedSourceId, citationData.getValue("sourceId").jsonPrimitive.content)
            assertEquals(1, citationData.getValue("location").jsonObject.getValue("lineStart").jsonPrimitive.content.toInt())
            assertEquals(2, citationData.getValue("location").jsonObject.getValue("lineEnd").jsonPrimitive.content.toInt())
            val quote = citationData.getValue("quote").jsonPrimitive.content
            assertTrue("2026-03-15" in quote)
            val providerPayload = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertTrue(providerPayload.getValue("document_chunks").jsonArray.any { quote in it.jsonPrimitive.content })
            assertFalse(providerPayload.toString().contains("greenhouse-chat.md"))
            val citation = page.locator(".chat-bubble.assistant .chat-evidence .citations")
                .first().locator("li").first()
            assertThat(citation).containsText("greenhouse-chat.md · Greenhouse · строки 1–2")
            assertThat(citation).containsText("2026-03-15")
            assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText("Источники документов")
            assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText("Память")

            page.reload()
            assertEquals(sessionId, page.url().substringAfter("session="))
            assertThat(page.locator("#chat-messages")).containsText("Synthetic chat answer.")
            assertThat(page.locator(".chat-bubble.assistant .citations").first()).containsText("greenhouse-chat.md")
        }
    }

    @Test
    fun chatSessionsKeepLongTaskStatePrivateAndExposeOnlyExplicitlySharedMemory() {
        withChatBrowser { page, _, fake ->
            val dialogMessages = CopyOnWriteArrayList<String>()
            page.onDialog {
                dialogMessages.add(it.message())
                it.accept()
            }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val taskSessionA = createChatSession(page)
            for (turn in 1..12) {
                submitChatQuestion(page, if (turn == 1) "session-A goal" else "session-A turn $turn")
            }
            assertThat(page.locator("#task-goal")).hasValue("Goal A")
            assertThat(page.locator("#task-clarifications")).hasValue("Clarification A")
            assertThat(page.locator("#task-constraints")).hasValue("Constraint A")
            assertThat(page.locator("#task-terms")).hasValue("A-term — Term A")
            val lastRequestA = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertEquals(8, lastRequestA.getValue("previous_messages").jsonArray.size)
            assertEquals("Goal A", lastRequestA.getValue("task_state").jsonObject.getValue("goal").jsonPrimitive.content)
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("Current goal: Goal A")
            repeat(12) { index ->
                assertThat(page.locator(".chat-bubble.assistant").nth(index).locator(".chat-evidence"))
                    .containsText("greenhouse-chat.md")
            }

            page.reload()
            assertThat(page.locator("#task-goal")).hasValue("Goal A")
            assertThat(page.locator("#task-clarifications")).hasValue("Clarification A")
            assertThat(page.locator("#task-constraints")).hasValue("Constraint A")
            assertThat(page.locator("#task-terms")).hasValue("A-term — Term A")
            val memoryEvidence = page.locator(".chat-bubble.assistant").last()
                .locator(".chat-evidence .citations").nth(1)
            assertThat(memoryEvidence).containsText("Факты памяти не передавались.")
            page.locator(".memory-item").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Цель задачи: Goal A"),
            ).locator("button").click()
            assertThat(page.locator("#chat-error")).containsText("добавлен в общую память")

            val taskSessionB = createChatSession(page)
            assertNotEquals(taskSessionA, taskSessionB)
            for (turn in 1..12) {
                submitChatQuestion(page, if (turn == 1) "session-B goal" else "session-B turn $turn")
                if (turn == 1) {
                    val firstRequestB = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
                    val firstContext = firstRequestB.toString()
                    assertTrue("Цель задачи: Goal A" in firstContext)
                    assertFalse("Clarification A" in firstContext)
                    assertFalse("Constraint A" in firstContext)
                    assertFalse("A-term" in firstContext)
                    assertTrue(firstRequestB.getValue("previous_messages").jsonArray.isEmpty())
                    val sharedEvidence = page.locator(".chat-bubble.assistant").last()
                        .locator(".chat-evidence .citations").nth(1)
                    assertThat(sharedEvidence).containsText("Общая память ·")
                    assertThat(sharedEvidence).containsText("Цель задачи: Goal A")
                }
            }
            val lastRequestB = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            val sessionBContext = lastRequestB.toString()
            assertTrue("Цель задачи: Goal A" in sessionBContext)
            assertFalse("Clarification A" in sessionBContext)
            assertFalse("Constraint A" in sessionBContext)
            assertFalse("A-term" in sessionBContext)
            assertEquals(8, lastRequestB.getValue("previous_messages").jsonArray.size)
            assertEquals("Goal B", page.locator("#task-goal").inputValue())
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("Current goal: Goal B")
            repeat(12) { index ->
                assertThat(page.locator(".chat-bubble.assistant").nth(index).locator(".chat-evidence"))
                    .containsText("greenhouse-chat.md")
            }
            assertTrue(dialogMessages.contains("Скопировать этот факт в общую память? Он станет доступен другим сессиям."))
            page.navigate("${page.url().substringBefore('?')}?tab=chat&session=$taskSessionA")
            assertThat(page.locator("#task-goal")).hasValue("Goal A")
            page.locator("#chat-delete-session").click()
            page.waitForFunction("() => !new URL(location.href).searchParams.has('session')")
            assertTrue(dialogMessages.contains("Удалить эту сессию, сообщения, факты и состояние задачи? Общая память останется."))
            assertThat(page.locator("#chat-sessions")).not().containsText("session-A goal")
            assertThat(page.locator("#chat-sessions")).containsText("session-B goal")
            assertThat(page.locator("#shared-memory-list")).containsText("Цель задачи: Goal A")
        }
    }

    @Test
    fun malformedChatResponseKeepsUserTurnAndCanRetryWithoutDuplicate() {
        withChatBrowser { page, _, fake ->
            page.locator("#chat-tab").click()
            createChatSession(page)
            page.locator("#task-goal").fill("Prior synthetic goal")
            page.locator("#task-state-save").click()
            assertThat(page.locator("#chat-error")).containsText("Состояние задачи сохранено.")
            fake.cloud.malformedStructuredResponse.set(true)
            page.locator("#chat-question").fill("Retry this synthetic turn")
            page.locator("#chat-send").click()
            page.locator("#chat-error").waitFor()
            assertEquals(1, page.locator(".chat-bubble.user").count())
            assertEquals(0, page.locator(".chat-bubble.assistant").count())
            assertThat(page.locator("#task-goal")).hasValue("Prior synthetic goal")
            assertThat(page.locator(".chat-bubble.user button")).containsText("Повторить этот ход")

            page.locator(".chat-bubble.user button").click()
            assertThat(page.locator(".chat-bubble.assistant")).containsText("Не знаю на основе текущих источников.")
            assertEquals(1, page.locator(".chat-bubble.user").count())
            assertEquals(1, page.locator(".chat-bubble.assistant").count())
            assertThat(page.locator("#task-goal")).hasValue("Prior synthetic goal")
            val evidence = page.locator(".chat-bubble.assistant .chat-evidence .citations")
            assertThat(evidence.nth(0)).containsText("Документные фрагменты не найдены.")
            assertThat(evidence.nth(1)).containsText("Факты памяти не передавались.")
        }
    }


    @Test
    fun chatLeavesCollectionAndBenchmarkFlowsUsable() {
        withChatBrowser { page, app, _ ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            createChatSession(page)
            submitChatQuestion(page, "greenhouse payment date?")

            page.locator("#collection-tab").click()
            assertThat(page.locator("#source-list")).containsText("greenhouse-chat.md")
            assertThat(page.locator("#model-select")).hasValue("deepseek-flash")
            page.locator("#question").fill("What is the greenhouse payment date?")
            val queryResponse = page.waitForResponse("**/api/query") {
                page.locator("#query-button").click()
            }
            assertEquals(200, queryResponse.status())
            assertThat(page.locator("#query-results")).containsText("greenhouse-chat.md")
            assertThat(page.locator("#query-results")).containsText("строки 1–2")
            assertThat(page.locator("#query-results")).containsText("2026-03-15")

            val source = Json.parseToJsonElement(
                page.request().get("http://127.0.0.1:${app.server.port}/api/sources").text(),
            ).jsonArray.single { it.jsonObject.getValue("name").jsonPrimitive.content == "greenhouse-chat.md" }
                .jsonObject
            val sourceId = source.getValue("sourceId").jsonPrimitive.content
            val questionsFile = temporaryDirectory.resolve("chat-followup-benchmark.json")
            Files.writeString(
                questionsFile,
                benchmarkQuestions(
                    sourceId,
                    2,
                    "The greenhouse payment date is 2026-03-15.",
                    questionText = "What is the greenhouse payment date?",
                    questionCount = 10,
                ),
                UTF_8,
            )
            page.locator("#benchmark-file").setInputFiles(questionsFile)
            page.locator("#benchmark-upload").click()
            assertThat(page.locator("#benchmark-error")).containsText("Набор сохранён")
            page.locator("#benchmark-run").click()
            assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
            assertThat(page.locator("#benchmark-results")).containsText("Baseline")
            assertThat(page.locator("#benchmark-results")).containsText("RAG")
        }
    }
    private fun withChatBrowser(block: (Page, BrowserAppFixture, FakeOllama) -> Unit) {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("chat-browser-app-data"))
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        block(page, app, fake)
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }

    private fun indexSyntheticGreenhouseSource(page: Page) {
        val source = temporaryDirectory.resolve("greenhouse-chat.md")
        Files.writeString(source, "# Greenhouse\nThe greenhouse payment date is 2026-03-15.\n")
        page.locator("#source-files").setInputFiles(source)
        page.locator("#upload-button").click()
        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
        page.locator("#index-button").click()
        assertThat(page.locator("#index-success")).containsText("Индексация завершена")
    }

    private fun createChatSession(page: Page): String {
        val previousId = page.url().substringAfter("session=", "")
        page.waitForResponse("**/api/chat/sessions") {
            page.locator("#chat-new-session").click()
        }
        page.waitForFunction(
            "() => { const id = new URL(location.href).searchParams.get('session'); return Boolean(id) && id !== '$previousId'; }",
        )
        return page.url().substringAfter("session=")
    }

    private fun submitChatQuestion(page: Page, question: String): JsonObject {
        page.locator("#chat-question").fill(question)
        val response = page.waitForResponse("**/api/chat/sessions/*/turns") {
            page.locator("#chat-send").click()
        }
        assertThat(page.locator(".chat-bubble.assistant").last()).containsText("Synthetic chat answer.")
        return Json.parseToJsonElement(response.text()).jsonObject
    }

    @Test
    fun fullSourceAndBenchmarkJourneyPreservesStateAndDeduplicatesContent() {
        FakeOllama().use { fake ->
            RequestProbe().use { probe ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("app-data"))
            val index = SqliteIndexRepository(paths.database)
            try {
                val benchmarkStore = BenchmarkStore(paths.database)
                try {
                    val chatStore = ChatStore(paths.database)
                    val jobs = JobManager()
                    try {
                        val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
                        val ollama = OllamaApi(fake.uri)
                        val embeddings = OllamaEmbeddingPort(ollama)
                        val modelConfiguration = modelConfiguration(paths, fake.cloud)
                        val chat = ChatCompletionsApi(modelConfiguration)
                        val rag = RagService(index, embeddings, chat, chat)
                        val chatService = ChatService(chatStore, rag, chat, embeddings)
                        val indexing = IndexWorkflow(
                            extractor = SourceExtractorRegistry(),
                            chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
                            embeddings = embeddings,
                            index = index,
                            sourcesDirectory = paths.sources,
                        )
                        val benchmark = BenchmarkRunner(rag, benchmarkStore, index)
                        val application = ApplicationService(index, benchmarkStore, catalog, jobs, indexing, rag, benchmark, ollama, modelConfiguration, chatStore, chatService)
                        LocalHttpServer(application, 0).use { server ->
                            server.start()
                            exerciseUserJourney(fake, probe, server.port, paths.sources)
                        }
                    } finally {
                        jobs.close()
                        chatStore.close()
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
    fun legacyBenchmarkResultsKeepHistoricalRetrievalLabel() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("legacy-browser-app-data"))
            Files.createDirectories(paths.dataDirectory)
            DriverManager.getConnection("jdbc:sqlite:${paths.database.toAbsolutePath()}").use { connection ->
                connection.createStatement().use { statement ->
                    statement.execute(
                        """CREATE TABLE benchmark_questions (
                            question_id TEXT PRIMARY KEY,
                            question TEXT NOT NULL,
                            expected_facts_json TEXT NOT NULL,
                            expected_sources_json TEXT NOT NULL,
                            position INTEGER NOT NULL UNIQUE
                        )""",
                    )
                    statement.execute(
                        """CREATE TABLE benchmark_results (
                            question_id TEXT NOT NULL,
                            strategy TEXT NOT NULL,
                            question TEXT NOT NULL,
                            expected_facts_json TEXT NOT NULL,
                            expected_sources_json TEXT NOT NULL,
                            retrieved_sources_json TEXT NOT NULL,
                            expected_source_hit INTEGER NOT NULL,
                            baseline_answer TEXT NOT NULL,
                            rag_answer TEXT NOT NULL,
                            baseline_rating TEXT,
                            rag_rating TEXT,
                            note TEXT,
                            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY(question_id,strategy)
                        )""",
                    )
                    statement.execute(
                        """INSERT INTO benchmark_questions VALUES
                            ('legacy-question','Legacy question','["legacy fact"]','[]',0)""",
                    )
                    statement.execute(
                        """INSERT INTO benchmark_results(
                            question_id,strategy,question,expected_facts_json,expected_sources_json,
                            retrieved_sources_json,expected_source_hit,baseline_answer,rag_answer,
                            baseline_rating,rag_rating,note
                        ) VALUES (
                            'legacy-question','FIXED_SIZE','Legacy question','["legacy fact"]','[]',
                            '[]',0,'Legacy baseline','Legacy RAG','PASS','PARTIAL','Legacy note'
                        )""",
                    )
                }
            }

            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        assertThat(page.locator("#benchmark-results")).containsText("Legacy RAG")
                        assertThat(page.locator("#benchmark-results")).containsText("Прежний retrieval · до D23")
                        assertThat(page.locator("#benchmark-results")).not().containsText("Enhanced retrieval · нормализация + reranker")
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }


    @Test
    fun queryShowsRawAndEnhancedRetrievalWithValidatedCitations() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("retrieval-comparison-app-data"))
            val source = temporaryDirectory.resolve("greenhouse-comparison.md")
            Files.writeString(
                source,
                "# Greenhouse alpha\nController reading is ALPHA.\n\n# Greenhouse beta\nController reading is BETA.\n\n# Greenhouse gamma\nController reading is GAMMA.\n\n# Greenhouse delta\nController reading is DELTA.\n",
            )
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        page.locator("#strategy").selectOption("STRUCTURAL")
                        page.locator("#top-k").fill("10")
                        val question = "  WHAT—   greenhouse “controller” reading?  "
                        val normalizedQuestion = "what- greenhouse \"controller\" reading?"
                        val queryEmbeddingStart = fake.embeddingInputs.size
                        page.locator("#question").fill(question)
                        val response = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        val result = Json.parseToJsonElement(response.text()).jsonObject
                        assertEquals(
                            listOf(question, normalizedQuestion),
                            fake.embeddingInputs.drop(queryEmbeddingStart),
                        )
                        val raw = result.getValue("rawSources").jsonArray.map { it.jsonObject }
                        val enhanced = result.getValue("rag").jsonObject.getValue("sources").jsonArray.map { it.jsonObject }
                        assertEquals(4, raw.size)
                        assertEquals(4, raw.map { it.getValue("location") }.toSet().size)
                        assertEquals(raw.map { it.getValue("chunkId").jsonPrimitive.content }.reversed(),
                            enhanced.map { it.getValue("chunkId").jsonPrimitive.content })
                        assertTrue(raw.all { it.getValue("quote").jsonPrimitive.content.length <= 300 })
                        val rawCard = page.locator("#query-results .answer-card").nth(1).innerText()
                        val enhancedCard = page.locator("#query-results .answer-card").nth(2).innerText()
                        val firstRawQuote = raw[0].getValue("quote").jsonPrimitive.content
                        val secondRawQuote = raw[1].getValue("quote").jsonPrimitive.content
                        assertTrue(rawCard.indexOf(firstRawQuote) in 0 until rawCard.indexOf(secondRawQuote))
                        assertTrue(enhancedCard.indexOf(secondRawQuote) in 0 until enhancedCard.indexOf(firstRawQuote))
                        assertTrue(rawCard.contains("cosine"))
                        val firstLocation = raw[0].getValue("location").jsonObject
                        val expectedLocation = firstLocation["pageStart"]?.jsonPrimitive?.content?.toIntOrNull()
                            ?.let { "стр. $it" }
                            ?: firstLocation["lineStart"]?.jsonPrimitive?.content?.toIntOrNull()
                                ?.let { "строки $it" }
                            ?: "место не указано"
                        assertTrue(rawCard.contains(expectedLocation))
                        assertTrue(rawCard.contains("greenhouse-comparison.md") && rawCard.contains("фрагмент"))
                        assertTrue(listOf("ALPHA", "BETA", "GAMMA", "DELTA").all(enhancedCard::contains))

                        val rerankPayload = Json.parseToJsonElement(fake.cloud.rerankMessages.last().last()).jsonObject
                        assertEquals(setOf("question", "candidates"), rerankPayload.keys)
                        assertEquals(4, rerankPayload.getValue("candidates").jsonArray.size)
                        assertTrue(rerankPayload.getValue("candidates").jsonArray.all {
                            it.jsonObject.keys == setOf("index", "text")
                        })
                        val candidateTexts = rerankPayload.getValue("candidates").jsonArray.map {
                            it.jsonObject.getValue("text").jsonPrimitive.content
                        }
                        assertEquals(4, enhanced.size)
                        enhanced.forEach { citation ->
                            val matchingRaw = raw.single {
                                it.getValue("chunkId") == citation.getValue("chunkId")
                            }
                            assertEquals(matchingRaw.getValue("sourceId"), citation.getValue("sourceId"))
                            assertEquals(matchingRaw.getValue("source"), citation.getValue("source"))
                            assertEquals(matchingRaw.getValue("section"), citation.getValue("section"))
                            assertEquals(matchingRaw.getValue("location"), citation.getValue("location"))
                            val candidateIndex = raw.indexOf(matchingRaw)
                            val quote = citation.getValue("quote").jsonPrimitive.content
                            assertTrue(quote.length <= 300)
                            assertTrue(candidateTexts[candidateIndex].contains(quote))
                        }

                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }

    @Test
    fun emptyEnhancedRetrievalKeepsBaselineAndAbstainsForRag() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("empty-retrieval-app-data"))
            val source = temporaryDirectory.resolve("synthetic-observatory.txt")
            Files.writeString(source, "Observatory temperature is 21 degrees.\n", UTF_8)
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        val question = "What are penguin breeding habits?"
                        val chatStart = fake.cloud.chatMessages.size
                        val rerankStart = fake.cloud.rerankMessages.size
                        page.locator("#question").fill(question)
                        val response = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        val result = Json.parseToJsonElement(response.text()).jsonObject
                        val rawSources = result.getValue("rawSources").jsonArray
                        assertTrue(rawSources.isNotEmpty())
                        assertTrue(rawSources.all { it.jsonObject.getValue("score").jsonPrimitive.content.toDouble() < 0.20 })
                        val rag = result.getValue("rag").jsonObject
                        assertTrue(rag.getValue("sources").jsonArray.isEmpty())
                        assertEquals("Synthetic answer.", result.getValue("baseline").jsonObject
                            .getValue("answer").jsonPrimitive.content)
                        assertEquals(
                            "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник.",
                            rag.getValue("answer").jsonPrimitive.content,
                        )
                        val cloudCalls = fake.cloud.chatMessages.drop(chatStart)
                        assertEquals(1, cloudCalls.size)
                        assertEquals(2, cloudCalls.single().size)
                        assertEquals(question, cloudCalls.single().last())
                        assertTrue(cloudCalls.single().none { "Observatory temperature" in it || "synthetic-observatory.txt" in it })
                        assertEquals(rerankStart, fake.cloud.rerankMessages.size)
                        assertEquals(
                            0,
                            page.locator("#query-results .answer-card").nth(2)
                                .locator("blockquote.citation-quote").count(),
                        )
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }

    @Test
    fun rerankerFailureDoesNotReturnRawAsEnhanced() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("reranker-failure-app-data"))
            val source = temporaryDirectory.resolve("synthetic-greenhouse.txt")
            Files.writeString(source, "Greenhouse humidity target is 64 percent.\n", UTF_8)
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        fake.cloud.failReranker.set(true)
                        val chatStart = fake.cloud.chatMessages.size
                        val rerankStart = fake.cloud.rerankMessages.size
                        page.locator("#question").fill("What is the greenhouse humidity target?")
                        val response = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        assertEquals(502, response.status())
                        assertFalse("Synthetic cloud failure" in response.text())
                        assertFalse("fixture-secret" in response.text())
                        assertEquals(2, fake.cloud.chatMessages.size - chatStart)
                        assertEquals(1, fake.cloud.rerankMessages.size - rerankStart)
                        assertTrue(
                            Json.parseToJsonElement(fake.cloud.rerankMessages.last().last()).jsonObject
                                .getValue("candidates").jsonArray.isNotEmpty(),
                        )
                        assertThat(page.locator("#query-error")).containsText("HTTP 503")
                        assertThat(page.locator("#query-results")).containsText("нового ответа нет")
                        assertFalse(page.locator("#query-results").innerText().contains("Greenhouse humidity target"))
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
            }
        }
    }

    @Test
    fun benchmarkPersistsRawAndRerankedRetrievalAndMetrics() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("benchmark-comparison-app-data"))
            val sourceFiles = (1..12).map { number ->
                temporaryDirectory.resolve("record-${number.toString().padStart(2, '0')}.txt").also { file ->
                    Files.writeString(file, "Synthetic record marker $number contains local event $number.\n", UTF_8)
                }
            }
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    var fixture: BrowserAppFixture? =
                        BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud))
                    try {
                        val app = fixture!!
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        val benchmarkDisclosure = page.getByText("Один запуск отправляет провайдеру")
                        assertThat(benchmarkDisclosure).containsText("до 20 пар reranker/RAG-запросов")
                        assertThat(benchmarkDisclosure).containsText("максимум 50 запросов")
                        assertThat(benchmarkDisclosure).containsText("запросы пропускаются")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(sourceFiles.toTypedArray())
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("12 файл(ов) добавлено")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        val sources = Json.parseToJsonElement(get("http://127.0.0.1:${app.server.port}/api/sources"))
                            .jsonArray.map { it.jsonObject }
                        assertEquals(12, sources.size)
                        val sourceNumberById = sources.associate { source ->
                            source.getValue("sourceId").jsonPrimitive.content to
                                source.getValue("name").jsonPrimitive.content
                                    .substringAfter("record-").substringBefore(".txt").toInt()
                        }
                        val rawByStrategy = mutableMapOf<String, List<JsonObject>>()
                        for (strategy in listOf("FIXED_SIZE", "STRUCTURAL")) {
                            page.locator("#strategy").selectOption(strategy)
                            page.locator("#top-k").fill("4")
                            page.locator("#question").fill("What observations are recorded?")
                            val queryResponse = page.waitForResponse("**/api/query") {
                                page.locator("#query-button").click()
                            }
                            rawByStrategy[strategy] = Json.parseToJsonElement(queryResponse.text()).jsonObject
                                .getValue("rawSources").jsonArray.map { it.jsonObject }
                        }
                        val fixedRaw = rawByStrategy.getValue("FIXED_SIZE")
                        val structuralIds = rawByStrategy.getValue("STRUCTURAL")
                            .map { it.getValue("sourceId").jsonPrimitive.content }.toSet()
                        val commonCitations = fixedRaw.filter {
                            it.getValue("sourceId").jsonPrimitive.content in structuralIds
                        }
                        assertEquals(4, commonCitations.size)
                        val retrievedIds = rawByStrategy.values.flatten()
                            .map { it.getValue("sourceId").jsonPrimitive.content }.toSet()
                        val missingSourceId = sources.first {
                            it.getValue("sourceId").jsonPrimitive.content !in retrievedIds
                        }.getValue("sourceId").jsonPrimitive.content
                        assertEquals(4, fixedRaw.size)
                        assertEquals(4, rawByStrategy.getValue("STRUCTURAL").size)

                        val emptyQuestion = "What are penguin breeding habits?"
                        val questionEntries = (1..10).joinToString(",") { number ->
                            val expectedCitation = if (number <= 4) {
                                commonCitations[(number - 1) % commonCitations.size]
                            } else {
                                null
                            }
                            val targetSourceId = expectedCitation?.getValue("sourceId")?.jsonPrimitive?.content
                                ?: missingSourceId
                            val targetNumber = sourceNumberById.getValue(targetSourceId)
                            val expectedSection = expectedCitation?.get("section")?.jsonPrimitive?.content
                                ?: "Document body"
                            val location = expectedCitation?.getValue("location")?.toString()
                                ?: """{"lineStart":1,"lineEnd":1}"""
                            val question = if (number == 5) emptyQuestion
                                else "Question $number: What observations are recorded?"
                            """{"id":"q${number.toString().padStart(2, '0')}","question":${JsonPrimitive(question)},"expectedFacts":[${JsonPrimitive("Synthetic record marker $targetNumber contains local event $targetNumber.")}],"expectedSources":[{"sourceId":${JsonPrimitive(targetSourceId)},"location":$location,"section":${JsonPrimitive(expectedSection)}}]}"""
                        }
                        val benchmarkFile = temporaryDirectory.resolve("comparison-questions.json")
                        Files.writeString(benchmarkFile, """{"questions":[$questionEntries]}""", UTF_8)
                        page.locator("#benchmark-file").setInputFiles(benchmarkFile)
                        page.locator("#benchmark-upload").click()
                        assertThat(page.locator("#benchmark-error")).containsText("Набор сохранён")

                        val benchmarkCallStart = fake.cloud.chatMessages.size
                        val benchmarkRerankStart = fake.cloud.rerankMessages.size
                        page.locator("#benchmark-run").click()
                        assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                        val benchmarkCalls = fake.cloud.chatMessages.drop(benchmarkCallStart)
                        assertEquals(46, benchmarkCalls.size)
                        assertEquals(18, fake.cloud.rerankMessages.size - benchmarkRerankStart)
                        assertEquals(
                            listOf(emptyQuestion),
                            benchmarkCalls.filter { it.last().contains(emptyQuestion) }.map { it.last() },
                        )

                        val before = Json.parseToJsonElement(get("http://127.0.0.1:${app.server.port}/api/benchmark/results"))
                            .jsonObject
                        val beforeRows = before.getValue("results").jsonArray
                        assertEquals(20, beforeRows.size)
                        val allRows = beforeRows.map { it.jsonObject }
                        fun expectedRank(row: JsonObject, retrievalField: String): Int? {
                            val expectedSources = row.getValue("expectedSources").jsonArray.map { it.jsonObject }
                            val retrieved = row.getValue(retrievalField).jsonArray.map { it.jsonObject }
                            val position = retrieved.indexOfFirst { citation ->
                                expectedSources.any { expected ->
                                    citation.getValue("sourceId") == expected.getValue("sourceId") &&
                                        citation.getValue("location") == expected.getValue("location") &&
                                        citation.getValue("section").jsonPrimitive.content.contains(
                                            expected.getValue("section").jsonPrimitive.content,
                                            ignoreCase = true,
                                        )
                                }
                            }
                            return position.takeIf { it >= 0 }?.plus(1)
                        }
                        fun storedRank(row: JsonObject, field: String): Int? =
                            row.getValue(field).jsonPrimitive.content.toIntOrNull()

                        for (row in allRows) {
                            assertEquals(expectedRank(row, "rawRetrievedSources"), storedRank(row, "rawExpectedSourceRank"))
                            assertEquals(expectedRank(row, "retrievedSources"), storedRank(row, "expectedSourceRank"))
                        }
                        val retrievalSummary = before.getValue("summary").jsonObject.getValue("retrieval").jsonArray
                            .map { it.jsonObject }
                        for (strategy in listOf("FIXED_SIZE", "STRUCTURAL")) {
                            val strategyRows = allRows.filter {
                                it.getValue("strategy").jsonPrimitive.content == strategy
                            }
                            assertEquals(10, strategyRows.size)
                            val rawRanks = strategyRows.map { expectedRank(it, "rawRetrievedSources") }
                            val enhancedRanks = strategyRows.map { expectedRank(it, "retrievedSources") }
                            assertTrue(rawRanks.any { it == null } && rawRanks.any { it != null })
                            assertTrue(enhancedRanks.any { it == null } && enhancedRanks.any { it != null })
                            val metrics = retrievalSummary.single {
                                it.getValue("strategy").jsonPrimitive.content == strategy
                            }
                            assertEquals(10, metrics.getValue("completedQuestions").jsonPrimitive.content.toInt())
                            assertEquals(
                                rawRanks.count { it != null && it <= 4 } / 10.0,
                                metrics.getValue("rawHitAtK").jsonPrimitive.content.toDouble(),
                            )
                            assertEquals(
                                rawRanks.sumOf { it?.let { rank -> 1.0 / rank } ?: 0.0 } / 10.0,
                                metrics.getValue("rawMrr").jsonPrimitive.content.toDouble(),
                            )
                            assertEquals(
                                enhancedRanks.count { it != null && it <= 4 } / 10.0,
                                metrics.getValue("enhancedHitAtK").jsonPrimitive.content.toDouble(),
                            )
                            assertEquals(
                                enhancedRanks.sumOf { it?.let { rank -> 1.0 / rank } ?: 0.0 } / 10.0,
                                metrics.getValue("enhancedMrr").jsonPrimitive.content.toDouble(),
                            )
                        }
                        assertThat(page.locator("#benchmark-results")).containsText("Raw retrieval")
                        val benchmarkText = page.locator("#benchmark-results").innerText()
                        fun displayedRank(label: String, rank: Int?): String =
                            if (rank == null) "$label: не найдено · RR 0.000"
                            else "$label: rank $rank · RR ${String.format(Locale.ROOT, "%.3f", 1.0 / rank)}"
                        allRows.forEach { row ->
                            val rawLabel = displayedRank("Raw", expectedRank(row, "rawRetrievedSources"))
                            val enhancedLabel = displayedRank("Enhanced", expectedRank(row, "retrievedSources"))
                            assertTrue(benchmarkText.contains(rawLabel), "Missing $rawLabel")
                            assertTrue(benchmarkText.contains(enhancedLabel), "Missing $enhancedLabel")
                        }

                        assertThat(page.locator("#benchmark-summary")).containsText("Raw Hit@4")
                        assertThat(page.locator("#benchmark-summary")).containsText("Enhanced Hit@4")
                        assertThat(page.locator("#benchmark-results")).containsText("Raw retrieval")
                        assertThat(page.locator("#benchmark-results")).containsText("Enhanced retrieval")
                        val review = page.locator("#benchmark-results .rating-form").first()
                        review.locator("select[name=\"baselineRating\"]").selectOption("PASS")
                        review.locator("select[name=\"ragRating\"]").selectOption("PASS")
                        review.locator("input[name=\"note\"]").fill("Synthetic benchmark review.")
                        page.waitForResponse("**/api/benchmark/review") {
                            review.locator("button[type=\"submit\"]").click()
                        }
                        val beforeRestart = Json.parseToJsonElement(get("http://127.0.0.1:${app.server.port}/api/benchmark/results"))
                            .jsonObject
                        fixture?.close()
                        fixture = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud))
                        val reopened = fixture!!
                        page.navigate("http://127.0.0.1:${reopened.server.port}/")
                        val afterRestart = Json.parseToJsonElement(get("http://127.0.0.1:${reopened.server.port}/api/benchmark/results"))
                            .jsonObject
                        assertEquals(beforeRestart.getValue("results"), afterRestart.getValue("results"))
                        assertEquals(beforeRestart.getValue("summary"), afterRestart.getValue("summary"))
                        assertEquals(
                            "Synthetic benchmark review.",
                            page.locator("#benchmark-results .rating-form").first()
                                .locator("input[name=\"note\"]").inputValue(),
                        )
                        assertThat(page.locator("#benchmark-summary")).containsText("Raw Hit@4")
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
    @Test
    fun singleBookIndexShowsPageProgressWhileEmbeddingIsInFlight() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("single-book-app-data"))
            val pdf = temporaryDirectory.resolve("single-book.pdf")
            writeSyntheticPdf(pdf)
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    var fixture: BrowserAppFixture? = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud))
                    try {
                        val app = fixture!!
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(pdf)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")

                        fake.holdPdfEmbedding.set(true)
                        val indexResponse = page.waitForResponse("**/api/index") {
                            page.locator("#index-button").click()
                        }
                        val jobId = Json.parseToJsonElement(indexResponse.text()).jsonObject.getValue("jobId").jsonPrimitive.content
                        assertTrue(
                            fake.pdfEmbeddingStarted.await(10, TimeUnit.SECONDS),
                            "Index job did not reach embedding: ${get("http://127.0.0.1:${app.server.port}/api/jobs/$jobId")}",
                        )
                        val progress = page.locator("#job-progress")
                        assertThat(progress).containsText("страница 1")
                        assertFalse(progress.textContent().contains("0 / 1"))
                        assertEquals(null, page.locator("#job-meter").getAttribute("value"))

                        fake.releasePdfEmbedding.countDown()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")
                    } finally {
                        fixture?.close()
                        fixture = null
                    }
                } finally {
                    browser.close()
                }
            } finally {
                playwright.close()
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
                    page.addInitScript(
                        """
                            (() => {
                              const originalFetch = window.fetch.bind(window);
                              window.__holdModelSelection = false;
                              window.__modelSelectionStarted = false;
                              window.__releaseModelSelection = null;
                              window.fetch = (input, init) => {
                                const url = new URL(typeof input === "string" ? input : input.url, window.location.href);
                                if (url.pathname === "/api/models/selection" && window.__holdModelSelection) {
                                  window.__modelSelectionStarted = true;
                                  return new Promise((resolve, reject) => {
                                    window.__releaseModelSelection = () => {
                                      window.__holdModelSelection = false;
                                      originalFetch(input, init).then(resolve, reject);
                                    };
                                  });
                                }
                                return originalFetch(input, init);
                              };
                            })();
                        """.trimIndent(),
                    )
                    page.setDefaultTimeout(30_000.0)
                    Files.createDirectories(paths.modelSelection.parent)
                    Files.writeString(paths.modelSelection, """{"providerId":"fixture","modelId":"removed-model"}""")
                    var fixture: BrowserAppFixture? = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud))
                    try {
                        val first = fixture!!
                        page.navigate("http://127.0.0.1:${first.server.port}/")
                        assertThat(page.locator("#model-select")).hasValue("deepseek-flash")
                        assertThat(page.locator("#model-selection-note")).containsText("Сохранённая модель")
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
                        assertTrue(page.locator("#query-button").isEnabled())
                        assertTrue(page.locator("#benchmark-run").isEnabled())
                        page.evaluate("window.__holdModelSelection = true")
                        page.locator("#model-select").selectOption("deepseek-v4-pro")
                        page.waitForFunction("window.__modelSelectionStarted === true")
                        assertTrue(page.locator("#query-button").isDisabled)
                        assertTrue(page.locator("#benchmark-run").isDisabled)
                        assertTrue(page.locator("#model-select").isDisabled)
                        val selectionResponse = page.waitForResponse("**/api/models/selection") {
                            page.evaluate("window.__releaseModelSelection(); true")
                        }
                        assertEquals(200, selectionResponse.status())
                        assertThat(page.locator("#model-select")).hasValue("deepseek-v4-pro")
                        assertThat(page.locator("#benchmark-run")).isEnabled()
                        assertThat(page.locator("#query-button")).isEnabled()
                        val benchmarkModelCallStart = fake.cloud.chatModels.size
                        fake.cloud.holdNextChat.set(true)
                        page.locator("#benchmark-run").click()
                        assertTrue(fake.cloud.firstChatStarted.await(10, TimeUnit.SECONDS))
                        assertThat(page.locator("#job-progress")).isVisible()
                        page.waitForResponse("**/api/models/selection") {
                            page.locator("#model-select").selectOption("deepseek-flash")
                        }
                        assertThat(page.locator("#model-select")).hasValue("deepseek-flash")
                        assertThat(page.locator("#model-select")).isEnabled()
                        assertThat(page.locator("#job-progress")).isVisible()
                        assertTrue(page.locator("#benchmark-upload").isDisabled)
                        assertTrue(page.locator("#query-button").isDisabled)
                        fake.cloud.releaseFirstChat.countDown()
                        assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                        val benchmarkModels = fake.cloud.chatModels.drop(benchmarkModelCallStart)
                        assertEquals(50, benchmarkModels.size)
                        assertTrue(benchmarkModels.all { it == "deepseek-v4-pro" })
                        page.waitForResponse("**/api/models/selection") {
                            page.locator("#model-select").selectOption("deepseek-v4-pro")
                        }
                        val apiResults = Json.parseToJsonElement(get("http://127.0.0.1:${first.server.port}/api/benchmark/results"))
                            .jsonObject.getValue("results").jsonArray
                        assertEquals(20, apiResults.size)
                        assertTrue(apiResults.all {
                            it.jsonObject.getValue("providerId").jsonPrimitive.content == "fixture" &&
                                it.jsonObject.getValue("modelId").jsonPrimitive.content == "deepseek-v4-pro"
                        })
                        assertTrue(apiResults.all {
                            val row = it.jsonObject
                            row.getValue("rawRetrievedSources").jsonArray.all { citation ->
                                citation.jsonObject.getValue("quote").jsonPrimitive.content.length <= 300
                            } &&
                                row.containsKey("rawExpectedSourceRank") && row.containsKey("expectedSourceRank")
                        })
                        assertThat(page.locator("#benchmark-summary")).containsText("Raw Hit@4")
                        assertThat(page.locator("#benchmark-summary")).containsText("Enhanced Hit@4")
                        assertThat(page.locator("#benchmark-results")).containsText("RR")
                        assertThat(page.locator("#benchmark-results")).containsText("Raw retrieval")
                        assertThat(page.locator("#benchmark-results")).containsText("Enhanced retrieval")
                        assertEquals(20, page.locator("#benchmark-results .strategy-card").count(), page.locator("#benchmark-results").innerText())

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
                        fixture = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake.cloud))
                        val reopened = fixture!!
                        page.navigate("http://127.0.0.1:${reopened.server.port}/")
                        assertThat(page.locator("#model-select")).hasValue("deepseek-v4-pro")
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
                        assertLineOneCitation(page, "restart-note.txt")
                        val addedSourceFile = temporaryDirectory.resolve("added-note.txt")
                        Files.writeString(addedSourceFile, "Beacon records remain in the local observatory.\n", UTF_8)
                        page.locator("#source-files").setInputFiles(addedSourceFile)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        assertThat(page.locator("#source-list")).containsText("restart-note.txt")
                        assertThat(page.locator("#source-list")).containsText("added-note.txt")
                        val sourcesAfterImport = Json.parseToJsonElement(get("http://127.0.0.1:${reopened.server.port}/api/sources"))
                            .jsonArray.map { it.jsonObject }
                        assertEquals(2, sourcesAfterImport.size)
                        assertEquals(
                            sourceId,
                            sourcesAfterImport.single { it.getValue("name").jsonPrimitive.content == "restart-note.txt" }
                                .getValue("sourceId").jsonPrimitive.content,
                        )
                        assertThat(page.locator("#source-list")).containsText("ожидает индексации")
                        page.locator("#question").fill("Where are orbit markers?")
                        page.locator("#query-button").click()
                        assertLineOneCitation(page, "restart-note.txt")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        page.locator("#question").fill("Where are beacon records?")
                        page.locator("#query-button").click()
                        assertLineOneCitation(page, "added-note.txt")
                        page.locator("#question").fill("Where are orbit markers?")
                        page.locator("#query-button").click()
                        assertLineOneCitation(page, "restart-note.txt")
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

    private fun formatUiBytes(bytes: Long): String {
        if (bytes < 1024) {
            return java.text.NumberFormat.getIntegerInstance(Locale.forLanguageTag("ru-RU")).format(bytes) + " Б"
        }
        val units = listOf("КиБ", "МиБ", "ГиБ")
        var value = bytes.toDouble()
        var unit = -1
        do {
            value /= 1024
            unit++
        } while (value >= 1024 && unit < units.lastIndex)
        return java.text.NumberFormat.getNumberInstance(Locale.forLanguageTag("ru-RU")).apply {
            maximumFractionDigits = 1
        }.format(value) + " " + units[unit]
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
                assertThat(page.locator("#upload-success")).containsText("invalid-utf8.md: Файл не является корректным UTF-8 текстом.")
                assertThat(page.locator("#upload-success")).containsText("capture.bin: Файл содержит бинарные или управляющие данные, а не UTF-8 текст.")

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
                val pendingTotalBytes = pendingFiles.sumOf { it.jsonObject.getValue("sizeBytes").jsonPrimitive.content.toLong() }
                assertTrue(
                    cancelledIndex.contains("Начать индексацию 6 файлов (${formatUiBytes(pendingTotalBytes)})"),
                    "Index confirmation omitted or misreported the total size: $cancelledIndex",
                )
                pendingFiles.forEach { file ->
                    val item = file.jsonObject
                    val name = item.getValue("name").jsonPrimitive.content
                    val sizeBytes = item.getValue("sizeBytes").jsonPrimitive.content.toLong()
                    assertTrue(
                        cancelledIndex.contains("$name (${formatUiBytes(sizeBytes)})"),
                        "Index confirmation omitted or misreported size for $name: $cancelledIndex",
                    )
                }
                assertEquals(0, fake.embeddingCalls.get())
                assertThat(page.locator("#source-list")).containsText("ожидает индексации")

                acceptNextDialog.set(true)
                page.locator("#index-button").click()
                assertThat(page.locator("#index-success")).containsText("Индексация завершена")
                assertTrue(awaitDialog(page, dialogMessages).startsWith("Начать индексацию"))
                assertThat(page.locator("#source-list")).containsText("страницы без текста: 2")

                val chatStart = fake.cloud.chatMessages.size
                page.locator("#question").fill("What is the greenhouse humidity target?")
                assertThat(page.locator("#query-button")).isEnabled()
                page.locator("#query-button").click()
                val queryResults = page.locator("#query-results")
                assertThat(queryResults).containsText("Baseline · без коллекции")
                assertThat(queryResults).containsText("RAG · fixture/deepseek-flash")
                assertThat(queryResults).containsText("greenhouse.html")
                assertThat(queryResults).containsText("Greenhouse telemetry")
                val queryMessages = fake.cloud.chatMessages.drop(chatStart)
                assertEquals(3, queryMessages.size)
                assertTrue(queryMessages.first().none { "Humidity target is 64 percent" in it })
                assertTrue(queryMessages[1].last().contains("Humidity target is 64 percent"))
                assertTrue(listOf("sourceId", "sourceName", "chunkId", "location").none { it in queryMessages[1].last() })
                assertTrue(queryMessages.last().any { "Humidity target is 64 percent" in it })
                assertTrue(queryMessages.flatten().none { "SCRIPT_ONLY_SECRET" in it || "STYLE_ONLY_SECRET" in it })
                val pdfChatStart = fake.cloud.chatMessages.size
                page.locator("#question").fill("What does the PDF field note say about conductivity?")
                page.locator("#query-button").click()
                assertThat(queryResults).containsText("Baseline · без коллекции")
                assertThat(queryResults).containsText("RAG · fixture/deepseek-flash")
                assertThat(queryResults).containsText("synthetic-field-guide.pdf")
                assertThat(queryResults).containsText("стр. 1")
                val pdfMessages = fake.cloud.chatMessages.drop(pdfChatStart)
                assertEquals(3, pdfMessages.size)
                assertTrue(pdfMessages.first().none { "conductivity is 21 units" in it })
                assertTrue(pdfMessages.last().any { "conductivity is 21 units" in it })
                assertEquals(0, probe.requests.get())

                page.locator("#top-k").fill("10")
                for (strategy in listOf("FIXED_SIZE", "STRUCTURAL")) {
                    page.locator("#strategy").selectOption(strategy)
                    val codeChatStart = fake.cloud.chatMessages.size
                    page.locator("#question").fill("What copper calibration limit is recorded?")
                    page.locator("#query-button").click()
                    assertThat(queryResults).containsText("copper-calibration.kt")
                    assertThat(queryResults).containsText("строки 1")
                    val codeMessages = fake.cloud.chatMessages.drop(codeChatStart)
                    assertEquals(3, codeMessages.size)
                    assertTrue(codeMessages.last().any { "LIMIT = 19" in it })

                    val textChatStart = fake.cloud.chatMessages.size
                    page.locator("#question").fill("What beacon threshold is recorded in the note?")
                    page.locator("#query-button").click()
                    assertThat(queryResults).containsText("version-note.txt")
                    assertThat(queryResults).containsText("строки 1")
                    val textMessages = fake.cloud.chatMessages.drop(textChatStart)
                    assertEquals(3, textMessages.size)
                    assertTrue(textMessages.last().any { "beacon threshold" in it })
                }
                assertEquals(0, probe.requests.get())
                val emptyContextChatStart = fake.cloud.chatMessages.size
                page.locator("#question").fill("What are penguin breeding habits?")
                page.locator("#query-button").click()
                assertThat(queryResults).containsText("Baseline ·")
                assertThat(queryResults).containsText("Synthetic answer.")
                assertThat(queryResults).containsText("Не знаю на основе текущих источников. Уточните вопрос или добавьте источник.")
                assertThat(queryResults).containsText("Проверенные фрагменты не найдены.")
                val emptyContextMessages = fake.cloud.chatMessages.drop(emptyContextChatStart)
                assertEquals(1, emptyContextMessages.size)
                assertTrue(emptyContextMessages.single().last().contains("penguin"))

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
                Files.writeString(malformedFile, """{"questions":"not-an-array"}""", UTF_8)
                page.locator("#benchmark-file").setInputFiles(malformedFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).containsText("Ожидается JSON-объект с массивом questions.")
                assertTrue(page.locator("#benchmark-run").isEnabled())
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")

                val malformedSyntaxFile = temporaryDirectory.resolve("malformed-syntax.json")
                Files.writeString(malformedSyntaxFile, "{\"questions\":[", UTF_8)
                page.locator("#benchmark-file").setInputFiles(malformedSyntaxFile)
                page.locator("#benchmark-upload").click()
                assertThat(page.locator("#benchmark-error")).not().hasText("")
                val malformedSyntaxError = page.locator("#benchmark-error").innerText()
                assertFalse(malformedSyntaxError == "Ожидается JSON-объект с массивом questions.")
                assertFalse(malformedSyntaxError == "Тело запроса не соответствует ожидаемому JSON-формату.")
                assertTrue(
                    malformedSyntaxError.contains("property", ignoreCase = true) ||
                        malformedSyntaxError.contains("position", ignoreCase = true) ||
                        malformedSyntaxError.contains("Unexpected end", ignoreCase = true),
                    "Expected a syntax-specific JSON error, got: $malformedSyntaxError",
                )
                assertTrue(page.locator("#benchmark-run").isEnabled())
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")
                val resultsAfterMalformedJson = Json.parseToJsonElement(get("$origin/api/benchmark/results"))
                    .jsonObject.getValue("results").jsonArray
                assertEquals(preservedRows, resultsAfterMalformedJson, "Invalid JSON replaced or mutated benchmark results.")
                val acceptedQuestions = Json.parseToJsonElement(get("$origin/api/benchmark/questions")).jsonObject
                assertTrue(acceptedQuestions.getValue("runnable").jsonPrimitive.content.toBoolean())
                assertEquals(
                    unresolvedQuestionsState.getValue("questions"),
                    acceptedQuestions.getValue("questions"),
                    "Invalid JSON replaced the previously accepted question set.",
                )

                page.locator("#question").fill("What is the greenhouse humidity target?")
                page.locator("#query-button").click()
                assertThat(page.locator("#query-results")).containsText("greenhouse.html")


                fake.cloud.failChat.set(true)
                page.locator("#benchmark-run").click()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark не завершён")
                assertThat(page.locator("#benchmark-error")).containsText("результаты предыдущих запусков")
                assertTrue(page.locator("#benchmark-results").textContent().orEmpty().isNotBlank())

                page.reload()
                assertThat(page.locator("#benchmark-error")).containsText("Benchmark не завершён")
                assertThat(page.locator("#benchmark-error")).containsText("результаты предыдущих запусков")
                assertThat(page.locator("#benchmark-results")).containsText("Baseline · FAIL")
                assertThat(page.locator("#benchmark-results")).containsText("RAG · PASS")

                fake.cloud.failChat.set(false)
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
                    .containsText("Не знаю на основе текущих источников. Уточните вопрос или добавьте источник.")
                val noEvidenceRows = Json.parseToJsonElement(get("$origin/api/benchmark/results"))
                    .jsonObject.getValue("results").jsonArray
                assertEquals(20, noEvidenceRows.size)
                assertTrue(noEvidenceRows.all {
                    it.jsonObject.getValue("baselineAnswer").jsonPrimitive.content == "Synthetic answer." &&
                        it.jsonObject.getValue("retrievedSources").jsonArray.isEmpty() &&
                        it.jsonObject.getValue("ragAnswer").jsonPrimitive.content ==
                        "Не знаю на основе текущих источников. Уточните вопрос или добавьте источник."
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
                val completedChatCount = fake.cloud.chatMessages.size
                val completedEmbeddingCalls = fake.embeddingCalls.get()
                fake.cloud.credentialConfigured.set(false)
                page.reload()
                assertThat(page.locator("#model-note")).containsText("FIXTURE_API_KEY")
                assertTrue(page.locator("#query-button").isDisabled())
                assertEquals(completedChatCount, fake.cloud.chatMessages.size)
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

    private fun modelConfiguration(paths: LocalRagPaths, fake: FakeDeepSeek): ModelConfiguration {
        val selection = ModelSelection("fixture", "deepseek-flash")
        val catalog = ProviderCatalog.create(
            listOf(
                ProviderDefinition(
                    "fixture",
                    "Fixture provider",
                    fake.uri.toString(),
                    "/chat/completions",
                    "FIXTURE_API_KEY",
                    listOf(
                        ProviderModelDefinition(selection.modelId, "Fixture Flash"),
                        ProviderModelDefinition("deepseek-v4-pro", "Fixture V4 Pro"),
                    ),
                ),
            ),
            selection,
            allowLoopbackHttpForTests = true,
        )
        return ModelConfiguration(
            catalog,
            ModelSelectionStore(paths.modelSelection, catalog),
            ProviderCredentialSource { if (fake.credentialConfigured.get()) "fixture-secret" else null },
        )
    }

    private fun assertLineOneCitation(page: Page, sourceName: String) {
        val citationItems = page.locator("#query-results .citations li")
        citationItems.first().waitFor()
        val citations = citationItems.allInnerTexts()
        assertTrue(
            citations.any { "$sourceName · строки 1 ·" in it },
            "Expected a line 1 citation for $sourceName, got $citations",
        )
    }

    private class BrowserAppFixture(
        paths: LocalRagPaths,
        fake: FakeOllama,
        private val modelConfiguration: ModelConfiguration,
    ) : AutoCloseable {
        private val index = SqliteIndexRepository(paths.database)
        private val benchmarkStore = BenchmarkStore(paths.database)
        private val chatStore = ChatStore(paths.database)
        private val jobs = JobManager()
        private val ollama = OllamaApi(fake.uri)
        private val embeddings = OllamaEmbeddingPort(ollama)
        private val chat = ChatCompletionsApi(modelConfiguration)
        private val rag = RagService(index, embeddings, chat, chat)
        private val chatService = ChatService(chatStore, rag, chat, embeddings)
        private val catalog = SourceCatalog(paths.sources, index, benchmarkStore::deleteSourceReferences)
        private val indexing = IndexWorkflow(
            extractor = SourceExtractorRegistry(),
            chunkers = listOf(FixedSizeChunker(), StructuralChunker()),
            embeddings = embeddings,
            index = index,
            sourcesDirectory = paths.sources,
        )
        private val benchmark = BenchmarkRunner(rag, benchmarkStore, index)
        private val application = ApplicationService(index, benchmarkStore, catalog, jobs, indexing, rag, benchmark, ollama, modelConfiguration, chatStore, chatService)
        val server = LocalHttpServer(application, 0)

        init {
            server.start()
        }

        override fun close() {
            server.close()
            chatStore.close()
            benchmarkStore.close()
            index.close()
        }
    }

    private class FakeOllama : AutoCloseable {
        val cloud = FakeDeepSeek()
        val embeddingInputs = CopyOnWriteArrayList<String>()
        val embeddingCalls = AtomicInteger()
        val holdPdfEmbedding = AtomicBoolean(false)
        val pdfEmbeddingStarted = CountDownLatch(1)
        val releasePdfEmbedding = CountDownLatch(1)
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeOllama.executor
            createContext("/api/tags") { exchange ->
                val models = """{"models":[{"name":"embeddinggemma:300m"}]}"""
                respond(exchange, 200, models)
            }
            createContext("/api/embed") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                val inputs = request.getValue("input").jsonArray.map { it.jsonPrimitive.content }
                embeddingInputs.addAll(inputs)
                embeddingCalls.addAndGet(inputs.size)
                if (holdPdfEmbedding.compareAndSet(true, false)) {
                    pdfEmbeddingStarted.countDown()
                    check(releasePdfEmbedding.await(20, TimeUnit.SECONDS))
                }
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
        }


        val uri: URI get() = URI("http://127.0.0.1:${server.address.port}")

        init {
            server.start()
        }

        override fun close() {
            server.stop(0)
            releasePdfEmbedding.countDown()
            executor.shutdownNow()
            cloud.close()
        }

        private fun respond(exchange: HttpExchange, status: Int, body: String) {
            val bytes = body.toByteArray(UTF_8)
            exchange.responseHeaders.set("Content-Type", "application/json; charset=utf-8")
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
    }
    private class FakeDeepSeek : AutoCloseable {
        val failChat = AtomicBoolean(false)
        val malformedStructuredResponse = AtomicBoolean(false)
        val failReranker = AtomicBoolean(false)
        val credentialConfigured = AtomicBoolean(true)
        val chatMessages = CopyOnWriteArrayList<List<String>>()
        val chatModels = CopyOnWriteArrayList<String>()
        val rerankMessages = CopyOnWriteArrayList<List<String>>()
        val holdNextChat = AtomicBoolean(false)
        val firstChatStarted = CountDownLatch(1)
        val releaseFirstChat = CountDownLatch(1)
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeDeepSeek.executor
            createContext("/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                val messages = request.getValue("messages").jsonArray.map {
                    it.jsonObject.getValue("content").jsonPrimitive.content
                }
                chatMessages.add(messages)
                chatModels.add(request.getValue("model").jsonPrimitive.content)
                val isReranker = messages.firstOrNull()?.contains("Ты reranker") == true
                if (isReranker) rerankMessages.add(messages)
                if (holdNextChat.compareAndSet(true, false)) {
                    firstChatStarted.countDown()
                    check(releaseFirstChat.await(20, TimeUnit.SECONDS))
                }
                if (failChat.get() || (isReranker && failReranker.get())) {
                    respond(exchange, 503, """{"error":"Synthetic cloud failure"}""")
                } else if (isReranker) {
                    val candidateCount = Json.parseToJsonElement(messages.last()).jsonObject
                        .getValue("candidates").jsonArray.size
                    val reversed = (candidateCount - 1 downTo 0).joinToString(",")
                    respond(exchange, 200, """{"choices":[{"message":{"content":"{\"ranked_indices\":[$reversed]}"}}]}""")
                } else if (request.containsKey("response_format")) {
                    val payload = Json.parseToJsonElement(messages.last()).jsonObject
                    val question = payload.getValue("question").jsonPrimitive.content
                    val currentGoal = payload.getValue("task_state").jsonObject.getValue("goal")
                    val firstSessionA = currentGoal == kotlinx.serialization.json.JsonNull && question.contains("session-A", ignoreCase = true)
                    val firstSessionB = currentGoal == kotlinx.serialization.json.JsonNull && question.contains("session-B", ignoreCase = true)
                    val answerGoal = when {
                        firstSessionA -> "Goal A"
                        firstSessionB -> "Goal B"
                        currentGoal != kotlinx.serialization.json.JsonNull -> currentGoal.jsonPrimitive.content
                        else -> null
                    }
                    val delta = buildJsonObject {
                        put("goal", when {
                            firstSessionA -> JsonPrimitive("Goal A")
                            firstSessionB -> JsonPrimitive("Goal B")
                            else -> kotlinx.serialization.json.JsonNull
                        })
                        put("clarifications_to_add", buildJsonArray {
                            if (firstSessionA) add(JsonPrimitive("Clarification A"))
                            if (firstSessionB) add(JsonPrimitive("Clarification B"))
                        })
                        put("constraints_to_add", buildJsonArray {
                            if (firstSessionA) add(JsonPrimitive("Constraint A"))
                            if (firstSessionB) add(JsonPrimitive("Constraint B"))
                        })
                        put("terms_to_add", buildJsonArray {
                            if (firstSessionA) add(buildJsonObject { put("term", "A-term"); put("definition", "Term A") })
                            if (firstSessionB) add(buildJsonObject { put("term", "B-term"); put("definition", "Term B") })
                        })
                    }
                    val structuredContent = if (malformedStructuredResponse.compareAndSet(true, false)) {
                        """{"answer":"Malformed synthetic answer","task_state_delta":{"goal":"broken"}}"""
                    } else {
                        buildJsonObject {
                            put("answer", answerGoal?.let { "Synthetic chat answer. Current goal: $it" } ?: "Synthetic chat answer.")
                            put("task_state_delta", delta)
                        }.toString()
                    }
                    val response = buildJsonObject {
                        put("choices", buildJsonArray {
                            add(buildJsonObject {
                                put("message", buildJsonObject { put("content", JsonPrimitive(structuredContent)) })
                            })
                        })
                    }.toString()
                    respond(exchange, 200, response)
                } else {
                    respond(exchange, 200, """{"choices":[{"message":{"content":"Synthetic answer."}}]}""")
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
            releaseFirstChat.countDown()
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
