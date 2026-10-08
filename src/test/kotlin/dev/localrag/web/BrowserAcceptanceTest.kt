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
import dev.localrag.ollama.OllamaReranker
import dev.localrag.source.LocalRagPaths
import dev.localrag.source.SourceCatalog
import dev.localrag.generation.ChatCompletionsApi
import dev.localrag.generation.ModelConfiguration
import dev.localrag.generation.ModelSelectionStore
import dev.localrag.generation.ProviderCatalog
import dev.localrag.generation.ProviderCredentialSource
import dev.localrag.generation.ProviderDefinition
import dev.localrag.generation.ProviderModelDefinition
import dev.localrag.generation.ProviderKind
import dev.localrag.domain.ModelSelection
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
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
    fun chatDisplaysOnlyModelSelectedDocumentCitations() {
        withChatBrowser { page, app, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            indexSyntheticCitationNoise(page)
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
            assertEquals(
                setOf(
                    "question",
                    "previous_messages",
                    "task_state",
                    "session_memory",
                    "shared_memory",
                    "document_candidates",
                    "generate_title",
                ),
                providerPayload.keys,
            )
            assertEquals("greenhouse payment date?", providerPayload.getValue("question").jsonPrimitive.content)
            assertTrue(providerPayload.getValue("previous_messages").jsonArray.isEmpty())
            val candidates = providerPayload.getValue("document_candidates").jsonArray
            assertTrue(candidates.size >= 2)
            assertTrue(candidates.all { it.jsonObject.keys == setOf("ref_id", "text") })
            assertTrue(candidates.any { quote in it.jsonObject.getValue("text").jsonPrimitive.content })
            assertTrue(candidates.all { it.jsonObject.getValue("ref_id").jsonPrimitive.content.matches(Regex("D\\d+")) })
            assertFalse(providerPayload.toString().contains("greenhouse-chat.md"))
            assertFalse(providerPayload.toString().contains(expectedSourceId))
            val citation = page.locator(".chat-bubble.assistant .chat-evidence .citations")
                .first().locator("li").first()
            assertThat(citation).containsText("greenhouse-chat.md · Greenhouse · строки 1–2")
            assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).not().containsText("Synthetic Observatory")
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
    fun chatSessionCanSwitchAfterAssistantTurnCompletes() {
        withChatBrowser { page, _, _ ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()

            val sessionA = createChatSession(page)
            submitChatQuestion(page, "session-A switch marker")
            val sessionB = createChatSession(page)
            submitChatQuestion(page, "session-B switch marker")
            assertNotEquals(sessionA, sessionB)

            val sessionAButton = page.locator("#chat-sessions button:not([aria-current='page'])").first()
            assertFalse(sessionAButton.isDisabled())
            sessionAButton.click()

            page.waitForFunction(
                "() => new URL(location.href).searchParams.get('session') === '$sessionA'",
            )
            assertThat(page.locator("#chat-messages")).containsText("session-A switch marker")
            assertThat(page.locator("#chat-messages")).not().containsText("session-B switch marker")
            assertEquals("Goal A", page.locator("#task-goal").inputValue())
            assertThat(page.locator("#task-goal")).not().hasValue("Goal B")

            val sessionBButton = page.locator("#chat-sessions button:not([aria-current='page'])").first()
            sessionBButton.click()
            page.waitForFunction(
                "() => new URL(location.href).searchParams.get('session') === '$sessionB'",
            )
            assertThat(page.locator("#chat-messages")).containsText("session-B switch marker")
            assertThat(page.locator("#chat-messages")).not().containsText("session-A switch marker")
            assertEquals("Goal B", page.locator("#task-goal").inputValue())
        }
    }

    @Test
    fun chatSessionsKeepIndependentProviderAndModelChoices() {
        withChatBrowser { page, app, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val sectionTabs = page.locator("nav[aria-label='Разделы приложения'] button").allTextContents()
                .map { it.trim() }
            assertEquals(listOf("Источники и поиск", "Чат"), sectionTabs)
            val comparisonQuestion = "What is the greenhouse payment date?"
            val cloudSession = createChatSession(page)
            assertThat(page.locator("#chat-provider-select")).hasValue("fixture")
            assertThat(page.locator("#chat-model-select")).hasValue("deepseek-flash")
            val cloudStart = fake.cloud.chatModels.size
            val cloudResponse = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill(comparisonQuestion)
                page.locator("#chat-send").click()
            }
            assertEquals(200, cloudResponse.status())
            val cloudMessage = Json.parseToJsonElement(cloudResponse.text()).jsonObject
                .getValue("message").jsonObject
            val cloudElapsedMs = cloudMessage.getValue("elapsedMs").jsonPrimitive.content.toLong()
            assertTrue(cloudElapsedMs >= 0)
            assertEquals("fixture", cloudMessage.getValue("modelSelection").jsonObject
                .getValue("providerId").jsonPrimitive.content)
            assertEquals("deepseek-flash", cloudMessage.getValue("modelSelection").jsonObject
                .getValue("modelId").jsonPrimitive.content)
            assertEquals(listOf("deepseek-flash"), fake.cloud.chatModels.drop(cloudStart))
            val cloudPayload = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertEquals(comparisonQuestion, cloudPayload.getValue("question").jsonPrimitive.content)
            val cloudAssistant = page.locator(".chat-bubble.assistant").last()
            assertThat(cloudAssistant).containsText("fixture/deepseek-flash")
            assertThat(cloudAssistant)
                .containsText("Время обработки хода: $cloudElapsedMs мс (retrieval + генерация).")

            val localSession = createChatSession(page)
            val providerResponse = page.waitForResponse("**/api/chat/sessions/*/model-selection") {
                page.locator("#chat-provider-select").selectOption("ollama")
            }
            assertEquals(200, providerResponse.status())
            assertThat(page.locator("#chat-model-select")).hasValue("qwen2.5:3b")
            val modelResponse = page.waitForResponse("**/api/chat/sessions/*/model-selection") {
                page.locator("#chat-model-select").selectOption("qwen2.5:7b")
            }
            assertEquals(200, modelResponse.status())
            val localGenerationStart = fake.localGenerationCalls.get()
            val localModelsStart = fake.localModels.size
            val localResponse = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill(comparisonQuestion)
                page.locator("#chat-send").click()
            }
            assertEquals(200, localResponse.status())
            assertEquals(comparisonQuestion, fake.localChatPayloads.last().getValue("question").jsonPrimitive.content)
            val localMessage = Json.parseToJsonElement(localResponse.text()).jsonObject
                .getValue("message").jsonObject
            val localElapsedMs = localMessage.getValue("elapsedMs").jsonPrimitive.content.toLong()
            val localSelection = localMessage.getValue("modelSelection").jsonObject
            assertEquals("ollama", localSelection.getValue("providerId").jsonPrimitive.content)
            assertEquals("qwen2.5:7b", localSelection.getValue("modelId").jsonPrimitive.content)
            assertTrue(localElapsedMs >= 0)
            assertEquals(1, fake.localGenerationCalls.get() - localGenerationStart)
            assertTrue(fake.localModels.drop(localModelsStart).contains("qwen2.5:7b"))
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:7b")
            assertThat(page.locator(".chat-bubble.assistant").last())
                .containsText("Время обработки хода: $localElapsedMs мс (retrieval + генерация).")
            val changedModelResponse = page.waitForResponse("**/api/chat/sessions/*/model-selection") {
                page.locator("#chat-model-select").selectOption("qwen2.5:3b")
            }
            assertEquals(200, changedModelResponse.status())
            val changedGenerationStart = fake.localGenerationCalls.get()
            val changedModelsStart = fake.localModels.size
            val changedTurnResponse = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("Second turn after changing the session model")
                page.locator("#chat-send").click()
            }
            assertEquals(200, changedTurnResponse.status())
            val changedMessage = Json.parseToJsonElement(changedTurnResponse.text()).jsonObject
                .getValue("message").jsonObject
            assertEquals(
                "qwen2.5:3b",
                changedMessage.getValue("modelSelection").jsonObject.getValue("modelId").jsonPrimitive.content,
            )
            val changedElapsedMs = changedMessage.getValue("elapsedMs").jsonPrimitive.content.toLong()
            assertTrue(changedElapsedMs >= 0)
            assertEquals(1, fake.localGenerationCalls.get() - changedGenerationStart)
            assertTrue(fake.localModels.drop(changedModelsStart).contains("qwen2.5:3b"))
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:3b")
            assertThat(page.locator(".chat-bubble.assistant").last())
                .containsText("Время обработки хода: $changedElapsedMs мс (retrieval + генерация).")

            page.locator("#collection-tab").click()
            assertThat(page.locator("#provider-select")).hasValue("fixture")
            assertThat(page.locator("#model-select")).hasValue("deepseek-flash")
            page.locator("#chat-tab").click()
            val cloudSessionButton = page.locator("#chat-sessions button:not([aria-current='page'])").first()
            cloudSessionButton.click()
            page.waitForFunction("() => new URL(location.href).searchParams.get('session') === '$cloudSession'")
            assertThat(page.locator("#chat-provider-select")).hasValue("fixture")
            assertThat(page.locator("#chat-model-select")).hasValue("deepseek-flash")
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("fixture/deepseek-flash")

            page.locator("#chat-sessions button:not([aria-current='page'])").first().click()
            page.waitForFunction("() => new URL(location.href).searchParams.get('session') === '$localSession'")
            assertThat(page.locator("#chat-provider-select")).hasValue("ollama")
            assertThat(page.locator("#chat-model-select")).hasValue("qwen2.5:3b")
            assertThat(page.locator(".chat-bubble.assistant").first()).containsText("ollama/qwen2.5:7b")
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:3b")

            val origin = "http://127.0.0.1:${app.server.port}"
            val firstChoice = Json.parseToJsonElement(
                page.request().get("$origin/api/chat/sessions/$cloudSession").text(),
            ).jsonObject.getValue("session").jsonObject.getValue("modelSelection").jsonObject
            val secondChoice = Json.parseToJsonElement(
                page.request().get("$origin/api/chat/sessions/$localSession").text(),
            ).jsonObject.getValue("session").jsonObject.getValue("modelSelection").jsonObject
            assertEquals("fixture", firstChoice.getValue("providerId").jsonPrimitive.content)
            assertEquals("ollama", secondChoice.getValue("providerId").jsonPrimitive.content)
        }
    }

    @Test
    fun chatRetryWaitsForModelSelectionAndIgnoresStaleSessionRefresh() {
        withChatBrowser { page, _, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val sessionId = createChatSession(page)

            fake.cloud.failChat.set(true)
            val failedTurn = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("Retry after selecting a local provider")
                page.locator("#chat-send").click()
            }
            assertTrue(failedTurn.status() >= 400)
            fake.cloud.failChat.set(false)
            val retry = page.locator(".chat-bubble.user").last().locator("button")
            assertThat(retry).isEnabled()

            page.evaluate(
                """sessionId => {
                    const gate = window.__chatDetailGate = {
                        path: `/api/chat/sessions/${sessionId}`,
                        held: false,
                        delivered: false,
                        release: null,
                        selectionHeld: false,
                        releaseSelection: null,
                    };
                    const originalFetch = window.fetch.bind(window);
                    window.fetch = async (input, init = {}) => {
                        const url = typeof input === "string" ? input : input.url;
                        const method = (init.method || (typeof input === "string" ? "GET" : input.method)).toUpperCase();
                        const path = new URL(url, location.href).pathname;
                        if (method === "PUT" && path.endsWith("/model-selection")) {
                            gate.selectionHeld = true;
                            await new Promise(resolve => { gate.releaseSelection = resolve; });
                        }
                        const response = await originalFetch(input, init);
                        if (method === "GET" && path === gate.path && !gate.held) {
                            gate.held = true;
                            await new Promise(resolve => {
                                gate.release = () => {
                                    gate.delivered = true;
                                    resolve();
                                };
                            });
                        }
                        return response;
                    };
                }""".trimIndent(),
                sessionId,
            )
            page.locator("#collection-tab").click()
            page.locator("#chat-tab").click()
            page.waitForFunction("() => window.__chatDetailGate.held === true")
            page.locator("#chat-provider-select").selectOption("ollama")
            page.waitForFunction("() => window.__chatDetailGate.selectionHeld === true")
            assertThat(retry).isDisabled()

            val providerResponse = page.waitForResponse("**/api/chat/sessions/*/model-selection") {
                page.evaluate("() => window.__chatDetailGate.releaseSelection()")
            }
            assertEquals(200, providerResponse.status())
            assertThat(page.locator("#chat-provider-select")).hasValue("ollama")
            assertThat(retry).isEnabled()

            page.evaluate("() => window.__chatDetailGate.release()")
            page.waitForFunction(
                "() => window.__chatDetailGate.delivered && document.querySelector('#chat-provider-select').value === 'ollama'",
            )
            val cloudCallsBeforeRetry = fake.cloud.chatModels.size
            val localCallsBeforeRetry = fake.localGenerationCalls.get()
            val retryResponse = page.waitForResponse("**/api/chat/sessions/*/turns") {
                retry.click()
            }
            assertEquals(200, retryResponse.status())
            assertEquals(cloudCallsBeforeRetry, fake.cloud.chatModels.size)
            assertTrue(fake.localGenerationCalls.get() > localCallsBeforeRetry)
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:3b")
            fake.failLocalGeneration.set(true)
            val cloudCallsBeforeLocalFailure = fake.cloud.chatModels.size
            val localFailure = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("Local generation failure must not route to Cloud")
                page.locator("#chat-send").click()
            }
            assertTrue(localFailure.status() >= 400)
            page.locator("#chat-error").waitFor()
            assertTrue(page.locator("#chat-error").isVisible)
            assertThat(page.locator("#chat-error")).containsText("Ollama вернул HTTP 503")
            assertEquals(cloudCallsBeforeLocalFailure, fake.cloud.chatModels.size)
        }
    }

    @Test
    fun chatBackNavigationLoadsAnotherSessionDuringSelectionUpdate() {
        withChatBrowser { page, _, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val firstSession = createChatSession(page)
            page.locator("#task-goal").fill("Goal from the first session")
            page.locator("#task-clarifications").fill("Clarification from the first session")
            page.locator("#task-constraints").fill("Constraint from the first session")
            page.locator("#task-terms").fill("term-first — definition-first")
            val firstSaveResponse = page.waitForResponse("**/api/chat/sessions/$firstSession/task-state") {
                page.locator("#task-state-save").click()
            }
            assertEquals(200, firstSaveResponse.status())
            assertThat(page.locator("#chat-error")).containsText("Состояние задачи сохранено.")
            page.waitForFunction("() => !document.querySelector('#task-state-save').disabled")

            val secondSession = createChatSession(page)
            page.locator("#task-goal").fill("Goal from the second session")
            page.locator("#task-clarifications").fill("Clarification from the second session")
            page.locator("#task-constraints").fill("Constraint from the second session")
            page.locator("#task-terms").fill("term-second — definition-second")
            val secondSaveResponse = page.waitForResponse("**/api/chat/sessions/$secondSession/task-state") {
                page.locator("#task-state-save").click()
            }
            assertEquals(200, secondSaveResponse.status())
            assertThat(page.locator("#chat-error")).containsText("Состояние задачи сохранено.")
            page.waitForFunction("() => !document.querySelector('#task-state-save').disabled")
            fake.cloud.failChat.set(true)
            val failedTurn = page.waitForResponse("**/api/chat/sessions/$secondSession/turns") {
                page.locator("#chat-question").fill("Retry this unfinished turn after changing models")
                page.locator("#chat-send").click()
            }
            assertTrue(failedTurn.status() >= 400)
            fake.cloud.failChat.set(false)
            val retry = page.locator(".chat-bubble.user").last().locator("button")
            assertThat(retry).isEnabled()


            page.evaluate(
                """
                    () => {
                      const gate = window.__selectionUpdateGate = {
                        selectionHeld: false,
                        releaseSelection: null,
                        detailHeld: false,
                        detailHeldOnce: false,
                        releaseDetail: null,
                        detailProcessed: false
                      };
                      const originalFetch = window.fetch.bind(window);
                      window.fetch = async (input, init = {}) => {
                        const url = typeof input === "string" ? input : input.url;
                        const method = (init.method || (typeof input === "string" ? "GET" : input.method)).toUpperCase();
                        const path = new URL(url, location.href).pathname;
                        if (method === "PUT" && path.endsWith("/model-selection")) {
                          gate.selectionHeld = true;
                          await new Promise(resolve => { gate.releaseSelection = resolve; });
                        }
                        const isDestinationDetail =
                          method === "GET" &&
                          path === "/api/chat/sessions/$secondSession" &&
                          !gate.detailHeldOnce;
                        if (isDestinationDetail) {
                          gate.detailHeldOnce = true;
                          gate.detailHeld = true;
                          await new Promise(resolve => { gate.releaseDetail = resolve; });
                        }
                        const response = await originalFetch(input, init);
                        if (isDestinationDetail) {
                          setTimeout(() => { gate.detailProcessed = true; }, 0);
                        }
                        return response;
                      };
                    }
                """.trimIndent(),
            )
            page.locator("#chat-provider-select").selectOption("ollama")
            page.waitForFunction("() => window.__selectionUpdateGate.selectionHeld")

            page.evaluate("() => window.history.back()")
            page.waitForFunction(
                "() => new URL(location.href).searchParams.get('session') === '$firstSession'",
            )
            assertThat(page.locator("#task-goal")).hasValue("Goal from the first session")
            assertThat(page.locator("#task-clarifications")).hasValue("Clarification from the first session")
            assertThat(page.locator("#task-constraints")).hasValue("Constraint from the first session")
            assertThat(page.locator("#task-terms")).hasValue("term-first — definition-first")

            page.evaluate("() => window.history.forward()")
            page.waitForFunction(
                "() => new URL(location.href).searchParams.get('session') === '$secondSession'",
            )
            page.waitForFunction("() => window.__selectionUpdateGate.detailHeld")
            val detailResponse = page.waitForResponse("**/api/chat/sessions/$secondSession") {
                page.evaluate("() => window.__selectionUpdateGate.releaseDetail()")
            }
            assertEquals(200, detailResponse.status())
            page.waitForFunction("() => window.__selectionUpdateGate.detailProcessed")
            assertThat(page.locator("#task-goal")).hasValue("")

            val selectionResponse = page.waitForResponse("**/api/chat/sessions/$secondSession/model-selection") {
                page.evaluate("() => window.__selectionUpdateGate.releaseSelection()")
            }
            assertEquals(200, selectionResponse.status())
            page.waitForFunction(
                "() => document.querySelector('#chat-provider-select').value === 'ollama' && " +
                    "!document.querySelector('#chat-provider-select').disabled",
            )
            assertThat(page.locator("#task-goal")).hasValue("Goal from the second session")
            assertThat(page.locator("#task-clarifications")).hasValue("Clarification from the second session")
            assertThat(page.locator("#task-constraints")).hasValue("Constraint from the second session")
            assertThat(page.locator("#task-terms")).hasValue("term-second — definition-second")
            assertThat(retry).isEnabled()
            assertThat(page.locator("#chat-send")).isDisabled()
            assertThat(page.locator("#chat-question")).isDisabled()


            val saveResponse = page.waitForResponse("**/api/chat/sessions/$secondSession/task-state") {
                page.locator("#task-state-save").click()
            }
            assertEquals(200, saveResponse.status())
            assertThat(page.locator("#chat-error")).containsText("Состояние задачи сохранено.")
            page.reload()
            assertThat(page.locator("#task-goal")).hasValue("Goal from the second session")
            assertThat(page.locator("#task-clarifications")).hasValue("Clarification from the second session")
            assertThat(page.locator("#task-constraints")).hasValue("Constraint from the second session")
            assertThat(page.locator("#task-terms")).hasValue("term-second — definition-second")
            assertThat(page.locator(".chat-bubble.user").last().locator("button")).isEnabled()
            assertThat(page.locator("#chat-send")).isDisabled()
            assertThat(page.locator("#chat-question")).isDisabled()
            val cloudCallsBeforeNavigatedRetry = fake.cloud.chatModels.size
            val localCallsBeforeNavigatedRetry = fake.localGenerationCalls.get()
            val navigatedRetryResponse = page.waitForResponse("**/api/chat/sessions/$secondSession/turns") {
                page.locator(".chat-bubble.user").last().locator("button").click()
            }
            assertEquals(200, navigatedRetryResponse.status())
            assertEquals(cloudCallsBeforeNavigatedRetry, fake.cloud.chatModels.size)
            assertEquals(localCallsBeforeNavigatedRetry + 1, fake.localGenerationCalls.get())
            val navigatedRetryMessage = Json.parseToJsonElement(navigatedRetryResponse.text()).jsonObject
                .getValue("message").jsonObject
            assertEquals(
                "ollama",
                navigatedRetryMessage.getValue("modelSelection").jsonObject.getValue("providerId").jsonPrimitive.content,
            )
            assertEquals(
                "qwen2.5:3b",
                navigatedRetryMessage.getValue("modelSelection").jsonObject.getValue("modelId").jsonPrimitive.content,
            )
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:3b")
            assertThat(page.locator(".chat-bubble.user")).hasCount(1)
            assertThat(page.locator(".chat-bubble.assistant")).hasCount(1)
            assertNotEquals(firstSession, secondSession)
        }
    }

    @Test
    fun legacyChatMigrationPreservesUnknownProvenanceStateAndMemory() {
        val paths = LocalRagPaths(temporaryDirectory.resolve("legacy-chat-browser-data"))
        Files.createDirectories(paths.dataDirectory)
        val sessionId = java.util.UUID.randomUUID().toString()
        val turnId = java.util.UUID.randomUUID().toString()
        val userMessageId = java.util.UUID.randomUUID().toString()
        val assistantMessageId = java.util.UUID.randomUUID().toString()
        val sessionFactId = java.util.UUID.randomUUID().toString()
        val sharedFactId = java.util.UUID.randomUUID().toString()
        val timestamp = "2026-10-01T00:00:00Z"
        val assistantTimestamp = "2026-10-01T00:00:01Z"
        val vector = byteArrayOf(0, 0, -128, 63)

        DriverManager.getConnection("jdbc:sqlite:${paths.database.toAbsolutePath()}").use { connection ->
            connection.createStatement().use { statement ->
                statement.execute(
                    """CREATE TABLE chat_sessions(
                        session_id TEXT PRIMARY KEY,title TEXT NOT NULL,title_generated INTEGER NOT NULL,
                        created_at TEXT NOT NULL,updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE chat_messages(
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        turn_id TEXT NOT NULL,role TEXT NOT NULL CHECK(role IN ('USER','ASSISTANT')),
                        content TEXT NOT NULL,document_citations_json TEXT NOT NULL DEFAULT '[]',
                        memory_references_json TEXT NOT NULL DEFAULT '[]',created_at TEXT NOT NULL,
                        UNIQUE(session_id,turn_id,role)
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE chat_task_states(
                        session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        goal TEXT,clarifications_json TEXT NOT NULL DEFAULT '[]',
                        constraints_json TEXT NOT NULL DEFAULT '[]',terms_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE chat_shared_facts(
                        memory_id TEXT PRIMARY KEY,content TEXT NOT NULL,embedding BLOB NOT NULL,
                        embedding_model TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL
                    )""".trimIndent(),
                )
                statement.execute(
                    """CREATE TABLE chat_session_facts(
                        memory_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,embedding BLOB NOT NULL,embedding_model TEXT NOT NULL,
                        created_at TEXT NOT NULL,UNIQUE(session_id,content)
                    )""".trimIndent(),
                )
                statement.execute("PRAGMA user_version=14")
            }
            connection.prepareStatement(
                "INSERT INTO chat_sessions(session_id,title,title_generated,created_at,updated_at) VALUES (?,?,1,?,?)",
            ).use { statement ->
                statement.setString(1, sessionId)
                statement.setString(2, "Legacy Chat")
                statement.setString(3, timestamp)
                statement.setString(4, timestamp)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                "INSERT INTO chat_messages(message_id,session_id,turn_id,role,content,created_at) VALUES (?,?,?,'USER',?,?)",
            ).use { statement ->
                statement.setString(1, userMessageId)
                statement.setString(2, sessionId)
                statement.setString(3, turnId)
                statement.setString(4, "Legacy question")
                statement.setString(5, timestamp)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                """INSERT INTO chat_messages(
                    message_id,session_id,turn_id,role,content,document_citations_json,
                    memory_references_json,created_at
                ) VALUES (?,?,?,'ASSISTANT',?,?,?,?)""".trimIndent(),
            ).use { statement ->
                statement.setString(1, assistantMessageId)
                statement.setString(2, sessionId)
                statement.setString(3, turnId)
                statement.setString(4, "Legacy answer")
                statement.setString(
                    5,
                    """[{"sourceId":"legacy-source-id","source":"legacy-source.md","section":"Legacy section","chunkId":"legacy-chunk","location":{"pageStart":null,"pageEnd":null,"lineStart":4,"lineEnd":4},"score":0.91,"quote":"Legacy citation excerpt."}]""",
                )
                statement.setString(
                    6,
                    """[{"scope":"SESSION","id":"$sessionFactId","text":"Legacy session fact"}]""",
                )
                statement.setString(7, assistantTimestamp)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                "INSERT INTO chat_task_states(session_id,goal,clarifications_json,constraints_json,terms_json,updated_at) VALUES (?,?,?,?,?,?)",
            ).use { statement ->
                statement.setString(1, sessionId)
                statement.setString(2, "Legacy task goal")
                statement.setString(3, """["Legacy clarification"]""")
                statement.setString(4, """["Legacy constraint"]""")
                statement.setString(5, """[{"term":"Legacy term","definition":"Legacy definition"}]""")
                statement.setString(6, timestamp)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                "INSERT INTO chat_session_facts(memory_id,session_id,content,embedding,embedding_model,created_at) VALUES (?,?,?,?,?,?)",
            ).use { statement ->
                statement.setString(1, sessionFactId)
                statement.setString(2, sessionId)
                statement.setString(3, "Legacy session fact")
                statement.setBytes(4, vector)
                statement.setString(5, "embeddinggemma:300m")
                statement.setString(6, timestamp)
                statement.executeUpdate()
            }
            connection.prepareStatement(
                "INSERT INTO chat_shared_facts(memory_id,content,embedding,embedding_model,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            ).use { statement ->
                statement.setString(1, sharedFactId)
                statement.setString(2, "Legacy shared fact")
                statement.setBytes(3, vector)
                statement.setString(4, "embeddinggemma:300m")
                statement.setString(5, timestamp)
                statement.setString(6, timestamp)
                statement.executeUpdate()
            }
        }

        FakeOllama().use { fake ->
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    val urlFor: (Int) -> String = { port ->
                        "http://127.0.0.1:$port/?tab=chat&session=$sessionId"
                    }

                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
                        page.navigate(urlFor(app.server.port))
                        assertThat(page.locator("#chat-messages")).containsText("Legacy question")
                        assertThat(page.locator("#chat-messages")).containsText("Legacy answer")
                        assertThat(page.locator(".chat-bubble.assistant")).containsText(
                            "Модель: не записана для исторического ответа",
                        )
                        assertThat(page.locator(".chat-bubble.assistant")).containsText(
                            "Время обработки исторического ответа не записано.",
                        )
                        assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText(
                            "legacy-source.md · Legacy section · строки 4",
                        )
                        assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText(
                            "Legacy citation excerpt.",
                        )
                        assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText(
                            "Legacy session fact",
                        )
                        assertThat(page.locator("#task-goal")).hasValue("Legacy task goal")
                        assertThat(page.locator("#task-clarifications")).hasValue("Legacy clarification")
                        assertThat(page.locator("#task-constraints")).hasValue("Legacy constraint")
                        assertThat(page.locator("#task-terms")).hasValue("Legacy term — Legacy definition")
                        assertThat(page.locator("#shared-memory-list")).containsText("Legacy shared fact")
                        assertThat(page.locator("#chat-provider-select")).hasValue("fixture")
                        assertThat(page.locator("#chat-model-select")).hasValue("deepseek-flash")

                        page.locator("#collection-tab").click()
                        val providerResponse = page.waitForResponse("**/api/models/selection") {
                            page.locator("#provider-select").selectOption("ollama")
                        }
                        assertEquals(200, providerResponse.status())
                        val modelResponse = page.waitForResponse("**/api/models/selection") {
                            page.locator("#model-select").selectOption("qwen2.5:7b")
                        }
                        assertEquals(200, modelResponse.status())
                        page.locator("#chat-tab").click()
                        assertThat(page.locator("#chat-provider-select")).hasValue("fixture")
                        assertThat(page.locator("#chat-model-select")).hasValue("deepseek-flash")

                        val migrated = Json.parseToJsonElement(
                            page.request().get(
                                "http://127.0.0.1:${app.server.port}/api/chat/sessions/$sessionId",
                            ).text(),
                        ).jsonObject.getValue("session").jsonObject
                        assertEquals("fixture", migrated.getValue("modelSelection").jsonObject
                            .getValue("providerId").jsonPrimitive.content)
                        val legacyAnswer = Json.parseToJsonElement(
                            page.request().get(
                                "http://127.0.0.1:${app.server.port}/api/chat/sessions/$sessionId",
                            ).text(),
                        ).jsonObject.getValue("messages").jsonArray.single {
                            it.jsonObject.getValue("role").jsonPrimitive.content == "ASSISTANT"
                        }.jsonObject
                        assertEquals(assistantMessageId, legacyAnswer.getValue("id").jsonPrimitive.content)
                        assertEquals(JsonNull, legacyAnswer.getValue("modelSelection"))
                        assertEquals(JsonNull, legacyAnswer.getValue("elapsedMs"))
                    }

                    ChatStore(paths.database).use { store ->
                        val persistedFacts = store.memoryFacts(sessionId)
                        assertEquals(sessionFactId, persistedFacts.single { it.text == "Legacy session fact" }.id)
                        assertEquals(sharedFactId, store.sharedFacts().single { it.text == "Legacy shared fact" }.id)
                    }

                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { restarted ->
                        page.navigate(urlFor(restarted.server.port))
                        assertThat(page.locator(".chat-bubble.assistant")).containsText(
                            "Модель: не записана для исторического ответа",
                        )
                        assertThat(page.locator(".chat-bubble.assistant .chat-evidence")).containsText(
                            "Legacy citation excerpt.",
                        )
                        assertThat(page.locator("#task-goal")).hasValue("Legacy task goal")
                        assertThat(page.locator("#shared-memory-list")).containsText("Legacy shared fact")
                        assertThat(page.locator("#chat-provider-select")).hasValue("fixture")
                        assertThat(page.locator("#chat-model-select")).hasValue("deepseek-flash")
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
    fun chatRequiresExplicitSelectionAndReconcilesLostSelectionResponse() {
        val paths = LocalRagPaths(temporaryDirectory.resolve("chat-browser-app-data"))
        Files.createDirectories(paths.modelSelection.parent)
        Files.writeString(paths.modelSelection, """{"providerId":"removed","modelId":"removed-model"}""")
        withChatBrowser { page, _, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            createChatSession(page)
            assertThat(page.locator("#chat-provider-select")).hasValue("")
            assertThat(page.locator("#chat-model-selection-note")).containsText("Выберите провайдера и модель")
            assertThat(page.locator("#chat-send")).isDisabled()

            page.locator("#collection-tab").click()
            val collectionRepair = page.waitForResponse("**/api/models/selection") {
                page.locator("#model-select").selectOption("deepseek-v4-pro")
            }
            assertEquals(200, collectionRepair.status())
            assertThat(page.locator("#model-note")).hasText("")
            page.locator("#chat-tab").click()
            assertThat(page.locator("#chat-provider-select")).hasValue("")
            assertThat(page.locator("#chat-model-selection-note")).containsText("Выберите провайдера и модель")
            assertThat(page.locator("#chat-send")).isDisabled()
            page.route("**/api/chat/sessions/*/model-selection") { route ->
                route.fetch()
                route.abort()
            }
            page.locator("#chat-provider-select").selectOption("ollama")
            assertThat(page.locator("#chat-error")).containsText("Сохранённый выбор перечитан с сервера.")
            assertThat(page.locator("#chat-provider-select")).hasValue("ollama")
            assertThat(page.locator("#chat-model-select")).hasValue("qwen2.5:3b")
            assertThat(page.locator("#chat-send")).isEnabled()

            val cloudCallsBefore = fake.cloud.chatModels.size
            val localCallsBefore = fake.localGenerationCalls.get()
            val response = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("What is the greenhouse payment date?")
                page.locator("#chat-send").click()
            }
            assertEquals(200, response.status())
            assertEquals(cloudCallsBefore, fake.cloud.chatModels.size)
            assertTrue(fake.localGenerationCalls.get() > localCallsBefore)
            assertThat(page.locator(".chat-bubble.assistant").last()).containsText("ollama/qwen2.5:3b")
        }
    }

    @Test
    fun chatAutoClassifiesMemoryAndIsolatesSessions() {
        withChatBrowser { page, _, fake ->
            val dialogs = CopyOnWriteArrayList<String>()
            page.onDialog {
                dialogs.add(it.message())
                it.accept()
            }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()

            val sessionA = createChatSession(page)
            val seedResponse = submitChatQuestion(page, "auto-memory-seed session-A")
            assertLongTurnCitation(page, seedResponse)
            assertThat(page.locator("#shared-memory-list")).containsText("Shared synthetic preference")
            assertThat(page.locator("#chat-view")).not().containsText("Факты этой сессии")
            assertEquals(0, page.locator("#session-memory-list").count())
            assertEquals(0, page.locator("button").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Поделиться с другими сессиями"),
            ).count())
            assertFalse(dialogs.any { "Скопировать этот факт" in it })

            val sessionOnlyResponse = submitChatQuestion(page, "session-only memory")
            assertLongTurnCitation(page, sessionOnlyResponse)
            val sessionOnlyReference = sessionOnlyResponse.getValue("message").jsonObject
                .getValue("memoryReferences").jsonArray.single().jsonObject
            assertEquals("SESSION", sessionOnlyReference.getValue("scope").jsonPrimitive.content)
            assertEquals("Session-only synthetic marker", sessionOnlyReference.getValue("text").jsonPrimitive.content)

            repeat(10) {
                assertLongTurnCitation(page, submitChatQuestion(page, "session-A turn ${it + 3}"))
            }
            assertThat(page.locator(".chat-bubble.user")).hasCount(12)
            assertThat(page.locator(".chat-bubble.assistant")).hasCount(12)
            assertThat(page.locator("#task-goal")).hasValue("Goal A")
            val lastRequestA = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertTrue("Session-only synthetic marker" in lastRequestA.toString())
            assertTrue(lastRequestA.getValue("previous_messages").jsonArray.size <= 8)

            val sessionB = createChatSession(page)
            assertNotEquals(sessionA, sessionB)
            val sharedResponse = submitChatQuestion(page, "session-B shared memory")
            assertLongTurnCitation(page, sharedResponse)
            val sharedReference = sharedResponse.getValue("message").jsonObject
                .getValue("memoryReferences").jsonArray.single().jsonObject
            assertEquals("SHARED", sharedReference.getValue("scope").jsonPrimitive.content)
            assertEquals("Shared synthetic preference", sharedReference.getValue("text").jsonPrimitive.content)
            val firstRequestB = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertTrue("Shared synthetic preference" in firstRequestB.toString())
            assertFalse("Session-only synthetic marker" in firstRequestB.toString())
            assertTrue(firstRequestB.getValue("previous_messages").jsonArray.isEmpty())
            assertFalse("session-A" in firstRequestB.toString())

            repeat(11) {
                assertLongTurnCitation(page, submitChatQuestion(page, "session-B turn ${it + 2}"))
            }
            assertThat(page.locator(".chat-bubble.user")).hasCount(12)
            assertThat(page.locator(".chat-bubble.assistant")).hasCount(12)
            val lastRequestB = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertTrue("Shared synthetic preference" in lastRequestB.toString())
            assertThat(page.locator("#task-goal")).hasValue("Goal B")
            assertThat(page.locator("#shared-memory-list")).containsText("Shared synthetic preference")
            val origin = page.url().substringBefore('?').trimEnd('/')
            page.navigate("$origin?tab=chat&session=$sessionA")
            assertThat(page.locator("#task-goal")).hasValue("Goal A")
            page.locator("#chat-delete-session").click()
            page.waitForFunction("() => !new URL(location.href).searchParams.has('session')")
            assertTrue(dialogs.contains("Удалить эту сессию, сообщения, факты и состояние задачи? Общая память останется."))
            val remainingSessionIds = Json.parseToJsonElement(
                page.request().get("$origin/api/chat/sessions").text(),
            ).jsonArray.map { it.jsonObject.getValue("id").jsonPrimitive.content }
            assertEquals(listOf(sessionB), remainingSessionIds)
            assertThat(page.locator("#shared-memory-list")).containsText("Shared synthetic preference")
            assertThat(page.locator("#chat-sessions button")).hasCount(1)
            page.locator("#chat-sessions button").click()
            page.waitForFunction(
                "() => new URL(location.href).searchParams.get('session') === '$sessionB'",
            )
            assertThat(page.locator("#chat-messages")).containsText("session-B shared memory")
            page.reload()
            assertThat(page.locator("#chat-messages")).containsText("session-B shared memory")
            assertThat(page.locator("#shared-memory-list")).containsText("Shared synthetic preference")
            assertEquals("Goal B", page.locator("#task-goal").inputValue())
        }
    }

    @Test
    fun chatUpdatesGoalAndFreezesModelGeneratedSessionTitle() {
        withChatBrowser { page, _, fake ->
            page.onDialog {
                if (it.type() == "prompt") it.accept("Edited synthetic preference") else it.accept()
            }
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            createChatSession(page)

            submitChatQuestion(page, "auto-memory-seed describe initial context")
            assertThat(page.locator("#chat-status")).containsText("Synthetic context summary")
            val title = page.locator("#chat-status").textContent()
            assertTrue(title.orEmpty().contains("auto-memory-seed"))
            assertEquals(1, fake.cloud.titleGenerationRequests.get())
            submitChatQuestion(page, "explicit goal-shift to a new synthetic objective")
            assertThat(page.locator("#task-goal")).hasValue("Updated synthetic goal")
            assertThat(page.locator("#chat-status")).containsText("Synthetic context summary")
            assertEquals(title, page.locator("#chat-status").textContent())
            assertEquals(1, fake.cloud.titleGenerationRequests.get())

            page.locator("#task-goal").fill("Manually refined synthetic goal")
            page.locator("#task-state-save").click()
            assertThat(page.locator("#chat-error")).containsText("Состояние задачи сохранено.")
            assertThat(page.locator("#task-goal")).hasValue("Manually refined synthetic goal")
            assertEquals(title, page.locator("#chat-status").textContent())
            assertTrue(page.locator("#chat-send").isEnabled())
            submitChatQuestion(page, "conversation-answer after manual state edit")
            assertThat(page.locator("#task-goal")).hasValue("Manually refined synthetic goal")
            assertEquals(1, fake.cloud.titleGenerationRequests.get())
            assertEquals(0, page.locator("#session-memory-heading").count())

            val sharedFact = page.locator("#shared-memory-list .memory-item").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Shared synthetic preference"),
            )
            assertThat(sharedFact).containsText("Shared synthetic preference")
            sharedFact.locator("button").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Изменить"),
            ).click()
            assertThat(page.locator("#shared-memory-list")).containsText("Edited synthetic preference")
            val editedFact = page.locator("#shared-memory-list .memory-item").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Edited synthetic preference"),
            )
            editedFact.locator("button").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Удалить"),
            ).click()
            assertThat(page.locator("#shared-memory-list")).not().containsText("Edited synthetic preference")

            page.reload()
            assertThat(page.locator("#task-goal")).hasValue("Manually refined synthetic goal")
            assertThat(page.locator("#chat-status")).containsText("Synthetic context summary")
            assertEquals(title, page.locator("#chat-status").textContent())
            assertEquals(1, fake.cloud.titleGenerationRequests.get())
            assertThat(page.locator("#chat-view")).not().containsText("Факты этой сессии")
            assertEquals(0, page.locator("button").filter(
                com.microsoft.playwright.Locator.FilterOptions().setHasText("Поделиться с другими сессиями"),
            ).count())
        }
    }

    @Test
    fun chatAbstainsWhenNoDocumentOrMemoryEvidence() {
        withChatBrowser { page, app, fake ->
            page.locator("#chat-tab").click()
            page.onDialog { it.accept() }
            val sessionId = createChatSession(page)
            val response = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("Unsupported synthetic factual question")
                page.locator("#chat-send").click()
            }
            assertEquals(200, response.status())
            val message = Json.parseToJsonElement(response.text()).jsonObject
                .getValue("message").jsonObject
            assertEquals(ChatService.ABSTENTION, message.getValue("content").jsonPrimitive.content)
            assertTrue(message.getValue("documentCitations").jsonArray.isEmpty())
            assertTrue(message.getValue("memoryReferences").jsonArray.isEmpty())
            assertTrue(fake.cloud.chatMessages.isNotEmpty())
            val emptyEvidencePayload = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            assertTrue(emptyEvidencePayload.getValue("document_candidates").jsonArray.isEmpty())
            assertTrue(emptyEvidencePayload.getValue("session_memory").jsonArray.isEmpty())
            assertTrue(emptyEvidencePayload.getValue("shared_memory").jsonArray.isEmpty())
            assertThat(page.locator(".chat-bubble.assistant .answer-text").last())
                .hasText(ChatService.ABSTENTION)
            assertThat(page.locator(".chat-bubble.assistant .chat-evidence"))
                .containsText("Документные фрагменты не найдены.")
            assertThat(page.locator(".chat-bubble.assistant .chat-evidence"))
                .containsText("Факты памяти не передавались.")

            val session = Json.parseToJsonElement(
                page.request().get("http://127.0.0.1:${app.server.port}/api/chat/sessions").text(),
            ).jsonArray.single().jsonObject
            assertEquals(sessionId, session.getValue("id").jsonPrimitive.content)
            page.reload()
            assertEquals(sessionId, page.url().substringAfter("session="))
            assertThat(page.locator(".chat-bubble.assistant .answer-text").last())
                .hasText(ChatService.ABSTENTION)
            assertEquals(1, page.locator(".chat-bubble.user").count())
            assertEquals(1, page.locator(".chat-bubble.assistant").count())

            page.locator("#collection-tab").click()
            indexSyntheticGreenhouseSource(page)
            page.locator("#chat-tab").click()
            val unsupportedResponse = page.waitForResponse("**/api/chat/sessions/*/turns") {
                page.locator("#chat-question").fill("Unsupported synthetic factual question")
                page.locator("#chat-send").click()
            }
            val unsupportedMessage = Json.parseToJsonElement(unsupportedResponse.text()).jsonObject
                .getValue("message").jsonObject
            assertEquals(ChatService.ABSTENTION, unsupportedMessage.getValue("content").jsonPrimitive.content)
            assertTrue(unsupportedMessage.getValue("documentCitations").jsonArray.isEmpty())
            assertTrue(unsupportedMessage.getValue("memoryReferences").jsonArray.isEmpty())
            val requestPayload = Json.parseToJsonElement(fake.cloud.chatMessages.last().last()).jsonObject
            val candidates = requestPayload.getValue("document_candidates").jsonArray
            assertTrue(candidates.isNotEmpty())
            assertTrue(candidates.none {
                "Europa" in it.jsonObject.getValue("text").jsonPrimitive.content
            })
            val unsupportedBubble = page.locator(".chat-bubble.assistant").last()
            assertThat(unsupportedBubble.locator(".answer-text")).hasText(ChatService.ABSTENTION)
            assertThat(unsupportedBubble.locator(".chat-evidence .citations").first())
                .containsText("Документные фрагменты не найдены.")
        }
    }

    @Test
    fun malformedChatResponseKeepsUserTurnAndCanRetryWithoutDuplicate() {
        withChatBrowser { page, _, fake ->
            page.onDialog { it.accept() }
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
            assertEquals(
                ChatService.ABSTENTION,
                page.locator(".chat-bubble.assistant .answer-text").innerText(),
            )
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
        withChatBrowser { page, app, fake ->
            page.onDialog { it.accept() }
            indexSyntheticGreenhouseSource(page)
            indexSyntheticCollectionDecoy(page)
            page.locator("#chat-tab").click()
            createChatSession(page)
            submitChatQuestion(page, "greenhouse payment date?")

            val cloudRequestsBeforeCollection = fake.cloud.chatRequests.size
            page.locator("#collection-tab").click()
            assertThat(page.locator("#source-list")).containsText("greenhouse-chat.md")
            assertThat(page.locator("#source-list")).containsText("collection-decoy.md")
            assertThat(page.locator("#model-select")).hasValue("deepseek-flash")
            val collectionQuestion = "What is the greenhouse payment date?"
            page.locator("#question").fill(collectionQuestion)
            assertEquals(cloudRequestsBeforeCollection, fake.cloud.chatRequests.size)
            val cloudRequestsBeforeQuery = fake.cloud.chatRequests.size
            val queryResponse = page.waitForResponse("**/api/query") {
                page.locator("#query-button").click()
            }
            assertEquals(200, queryResponse.status())
            assertThat(page.locator("#query-results")).containsText("greenhouse-chat.md")
            assertThat(page.locator("#query-results")).containsText("строки 1–2")
            assertThat(page.locator("#query-results")).containsText("2026-03-15")

            val sourceRecords = Json.parseToJsonElement(
                page.request().get("http://127.0.0.1:${app.server.port}/api/sources").text(),
            ).jsonArray.map { it.jsonObject }
            assertEquals(
                setOf("greenhouse-chat.md", "collection-decoy.md"),
                sourceRecords.map { it.getValue("name").jsonPrimitive.content }.toSet(),
            )
            assertTrue(sourceRecords.all { it.getValue("status").jsonPrimitive.content == "READY" })
            val retrievedSources = Json.parseToJsonElement(queryResponse.text()).jsonObject
                .getValue("rag").jsonObject.getValue("sources").jsonArray.map { it.jsonObject }
            assertEquals(1, retrievedSources.size)
            assertEquals("greenhouse-chat.md", retrievedSources.single().getValue("source").jsonPrimitive.content)
            val retrievedQuote = retrievedSources.single().getValue("quote").jsonPrimitive.content
            val collectionRequests = fake.cloud.chatRequests.drop(cloudRequestsBeforeQuery)
            assertEquals(2, collectionRequests.size)
            val collectionMessages = collectionRequests.map { request ->
                assertEquals(setOf("model", "stream", "messages"), request.keys)
                assertEquals("deepseek-flash", request.getValue("model").jsonPrimitive.content)
                val messages = request.getValue("messages").jsonArray.map { it.jsonObject }
                assertEquals(2, messages.size)
                messages.forEach { assertEquals(setOf("role", "content"), it.keys) }
                assertEquals(
                    listOf("system", "user"),
                    messages.map { it.getValue("role").jsonPrimitive.content },
                )
                messages
            }
            val collectionSystemMessages = collectionMessages.map {
                it.first().getValue("content").jsonPrimitive.content
            }
            val collectionUserMessages = collectionMessages.map {
                it.last().getValue("content").jsonPrimitive.content
            }
            assertEquals(collectionQuestion, collectionUserMessages.first())
            assertTrue(collectionQuestion in collectionUserMessages.last())
            assertTrue(retrievedQuote in collectionUserMessages.last())
            assertTrue(collectionSystemMessages.none { collectionQuestion in it })
            assertTrue(collectionSystemMessages.none { retrievedQuote in it })
            val serializedCollectionRequests = collectionRequests.joinToString("\n") { it.toString() }
            assertFalse("UNRELATED_PRIVATE_INDEX_SENTINEL" in serializedCollectionRequests)
            sourceRecords.forEach { source ->
                assertFalse(source.getValue("name").jsonPrimitive.content in serializedCollectionRequests)
                assertFalse(source.getValue("sourceId").jsonPrimitive.content in serializedCollectionRequests)
            }
            assertEquals(cloudRequestsBeforeQuery + 2, fake.cloud.chatRequests.size)

            val localProviderResponse = page.waitForResponse("**/api/models/selection") {
                page.locator("#provider-select").selectOption("ollama")
            }
            assertEquals(200, localProviderResponse.status())
            assertThat(page.locator("#model-select")).hasValue("qwen2.5:3b")
            val benchmarkCloudStart = fake.cloud.chatModels.size
            val benchmarkGenerationStart = fake.localGenerationCalls.get()
            val benchmarkRerankStart = fake.rerankRequests.size
            val source = sourceRecords.single {
                it.getValue("name").jsonPrimitive.content == "greenhouse-chat.md"
            }
            val sourceId = source.getValue("sourceId").jsonPrimitive.content
            val questionsFile = temporaryDirectory.resolve("chat-followup-benchmark.json")
            Files.writeString(
                questionsFile,
                benchmarkQuestions(
                    sourceId,
                    2,
                    "The greenhouse payment date is 2026-03-15.",
                    questionText = collectionQuestion,
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
            assertEquals(benchmarkCloudStart, fake.cloud.chatModels.size)
            assertEquals(30, fake.localGenerationCalls.get() - benchmarkGenerationStart)
            assertEquals(20, fake.rerankRequests.size - benchmarkRerankStart)
            val review = page.locator("#benchmark-results .rating-form").first()
            review.locator("select[name=\"baselineRating\"]").selectOption("PASS")
            review.locator("select[name=\"ragRating\"]").selectOption("PARTIAL")
            review.locator("input[name=\"note\"]").fill("Local benchmark review.")
            page.waitForResponse("**/api/benchmark/review") {
                review.locator("button[type=\"submit\"]").click()
            }
            val origin = "http://127.0.0.1:${app.server.port}"
            val resultsBeforeModelChange = Json.parseToJsonElement(get("$origin/api/benchmark/results")).jsonObject
            assertTrue(resultsBeforeModelChange.getValue("results").jsonArray.all { result ->
                result.jsonObject.getValue("providerId").jsonPrimitive.content == "ollama" &&
                    result.jsonObject.getValue("modelId").jsonPrimitive.content == "qwen2.5:3b"
            })
            val modelResponse = page.waitForResponse("**/api/models/selection") {
                page.locator("#model-select").selectOption("qwen2.5:7b")
            }
            assertEquals(200, modelResponse.status())
            val resultsAfterModelChange = Json.parseToJsonElement(get("$origin/api/benchmark/results")).jsonObject
            assertEquals(resultsBeforeModelChange, resultsAfterModelChange)
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
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
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
    private fun indexSyntheticCollectionDecoy(page: Page) {
        val source = temporaryDirectory.resolve("collection-decoy.md")
        Files.writeString(
            source,
            "# Penguin archive\nUNRELATED_PRIVATE_INDEX_SENTINEL penguin breeding notes.\n",
            UTF_8,
        )
        page.locator("#source-files").setInputFiles(source)
        page.locator("#upload-button").click()
        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
        page.locator("#index-button").click()
        assertThat(page.locator("#index-success")).containsText("Индексация завершена")
    }
    private fun indexSyntheticCitationNoise(page: Page) {
        val source = temporaryDirectory.resolve("synthetic-observatory.md")
        Files.writeString(source, "# Synthetic Observatory\nCalibration note: the beacon threshold is 14 units.\n")
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

    private fun assertLongTurnCitation(page: Page, response: JsonObject) {
        val citations = response.getValue("message").jsonObject.getValue("documentCitations").jsonArray
        assertTrue(citations.isNotEmpty())
        val sources = citations.map { it.jsonObject.getValue("source").jsonPrimitive.content }
        assertThat(page.locator(".chat-bubble.assistant").last().locator(".chat-evidence .citations").first())
            .containsText(sources.joinToString())
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
                        val modelConfiguration = modelConfiguration(paths, fake)
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
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
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
    fun collectionUsesSelectedLocalGenerationModelWithoutCloudFallback() {
        FakeOllama().use { fake ->
            val paths = LocalRagPaths(temporaryDirectory.resolve("local-provider-app-data"))
            val source = temporaryDirectory.resolve("local-provider-greenhouse.txt")
            Files.writeString(source, "Greenhouse payment date is 17 June.\n", UTF_8)
            val playwright = Playwright.create()
            try {
                val browser = playwright.chromium().launch(
                    BrowserType.LaunchOptions().setHeadless(true).setArgs(listOf("--no-sandbox")),
                )
                try {
                    val page = browser.newPage()
                    page.setDefaultTimeout(30_000.0)
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        val origin = "http://127.0.0.1:${app.server.port}"
                        val sourceRecordsBefore = Json.parseToJsonElement(get("$origin/api/sources")).jsonArray
                        val indexRowsBefore = indexSnapshot(paths.database)
                        val sourceFilesBefore = sourceFilesSnapshot(paths.sources)
                        assertEquals(1, sourceRecordsBefore.size)
                        assertTrue(indexRowsBefore.second.isNotEmpty())
                        assertEquals(1, sourceFilesBefore.size)
                        val providerResponse = page.waitForResponse("**/api/models/selection") {
                            page.locator("#provider-select").selectOption("ollama")
                        }
                        assertEquals(200, providerResponse.status())
                        assertThat(page.locator("#model-select")).hasValue("qwen2.5:3b")
                        val modelResponse = page.waitForResponse("**/api/models/selection") {
                            page.locator("#model-select").selectOption("qwen2.5:7b")
                        }
                        assertEquals(200, modelResponse.status())

                        assertEquals(
                            sourceRecordsBefore,
                            Json.parseToJsonElement(get("$origin/api/sources")).jsonArray,
                        )
                        assertEquals(indexRowsBefore, indexSnapshot(paths.database))
                        assertEquals(sourceFilesBefore, sourceFilesSnapshot(paths.sources))
                        val activeJob = Json.parseToJsonElement(get("$origin/api/status")).jsonObject
                            .getValue("activeJob")
                        assertEquals(JsonNull, activeJob)
                        val cloudStart = fake.cloud.chatMessages.size
                        val localGenerationStart = fake.localGenerationCalls.get()
                        val localModelStart = fake.localModels.size
                        val localRerankStart = fake.rerankRequests.size
                        page.locator("#question").fill("What is the greenhouse payment date?")
                        val queryResponse = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        assertEquals(200, queryResponse.status())
                        val result = Json.parseToJsonElement(queryResponse.text()).jsonObject
                        val selection = result.getValue("modelSelection").jsonObject
                        assertEquals("ollama", selection.getValue("providerId").jsonPrimitive.content)
                        assertEquals("qwen2.5:7b", selection.getValue("modelId").jsonPrimitive.content)
                        val rag = result.getValue("rag").jsonObject
                        assertEquals("Synthetic local answer.", rag.getValue("answer").jsonPrimitive.content)
                        assertTrue(rag.getValue("sources").jsonArray.isNotEmpty())
                        assertEquals(cloudStart, fake.cloud.chatMessages.size)
                        assertEquals(2, fake.localGenerationCalls.get() - localGenerationStart)
                        val localModels = fake.localModels.drop(localModelStart)
                        assertTrue(localModels.contains("qwen2.5:7b"))
                        assertTrue(localModels.contains("qwen2.5:3b"))
                        assertEquals(1, fake.rerankRequests.size - localRerankStart)
                        val cloudCallsBeforeGenerationFailure = fake.cloud.chatMessages.size
                        fake.failLocalGeneration.set(true)
                        val generationFailure = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        assertEquals(503, generationFailure.status())
                        page.locator("#query-error").waitFor()
                        assertTrue(page.locator("#query-error").isVisible)
                        assertThat(page.locator("#query-error")).containsText("Ollama вернул HTTP 503")
                        assertEquals(cloudCallsBeforeGenerationFailure, fake.cloud.chatMessages.size)

                        val cloudCallsBeforeRerankerFailure = fake.cloud.chatMessages.size
                        fake.failReranker.set(true)
                        val rerankerFailure = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        assertEquals(503, rerankerFailure.status())
                        page.locator("#query-error").waitFor()
                        assertTrue(page.locator("#query-error").isVisible)
                        assertThat(page.locator("#query-error")).containsText("Ollama вернул HTTP 503")
                        assertEquals(cloudCallsBeforeRerankerFailure, fake.cloud.chatMessages.size)
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
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
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

                        val rerankPayload = fake.rerankRequests.last()
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
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        assertThat(page.locator("#upload-success")).containsText("1 файл(ов) добавлено")
                        page.locator("#index-button").click()
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        val question = "What are penguin breeding habits?"
                        val chatStart = fake.cloud.chatMessages.size
                        val rerankStart = fake.rerankRequests.size
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
                        assertEquals(rerankStart, fake.rerankRequests.size)
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
                    BrowserAppFixture(paths, fake, modelConfiguration(paths, fake)).use { app ->
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        page.onDialog { it.accept() }
                        page.locator("#source-files").setInputFiles(source)
                        page.locator("#upload-button").click()
                        val indexStart = page.waitForResponse("**/api/index") {
                            page.locator("#index-button").click()
                        }
                        assertEquals(202, indexStart.status())
                        page.locator("#index-success").waitFor(
                            com.microsoft.playwright.Locator.WaitForOptions().setTimeout(30_000.0),
                        )
                        assertThat(page.locator("#index-success")).containsText("Индексация завершена")

                        fake.failReranker.set(true)
                        val chatStart = fake.cloud.chatMessages.size
                        val rerankStart = fake.rerankRequests.size
                        page.locator("#question").fill("What is the greenhouse humidity target?")
                        val response = page.waitForResponse("**/api/query") {
                            page.locator("#query-button").click()
                        }
                        assertEquals(503, response.status())
                        assertFalse("SYNTHETIC_LOCAL_RERANK_FAILURE" in response.text())
                        assertFalse("fixture-secret" in response.text())
                        assertEquals(1, fake.cloud.chatMessages.size - chatStart)
                        assertEquals(1, fake.rerankRequests.size - rerankStart)
                        assertTrue(fake.rerankRequests.last().getValue("candidates").jsonArray.isNotEmpty())
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
                        BrowserAppFixture(paths, fake, modelConfiguration(paths, fake))
                    try {
                        val app = fixture!!
                        page.navigate("http://127.0.0.1:${app.server.port}/")
                        val benchmarkDisclosure = page.getByText("С выбранной моделью выполняются")
                        assertThat(benchmarkDisclosure).containsText("10 baseline-вопросов и до 20 RAG-запросов")
                        assertThat(benchmarkDisclosure).containsText("максимум 30 вызовов")
                        assertThat(benchmarkDisclosure).containsText("Локальный reranker")
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
                        val benchmarkRerankStart = fake.rerankRequests.size
                        page.locator("#benchmark-run").click()
                        assertThat(page.locator("#benchmark-error")).containsText("Benchmark завершён")
                        val benchmarkCalls = fake.cloud.chatMessages.drop(benchmarkCallStart)
                        assertEquals(28, benchmarkCalls.size)
                        assertEquals(18, fake.rerankRequests.size - benchmarkRerankStart)
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
                        fixture = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake))
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
                    var fixture: BrowserAppFixture? = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake))
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
                    var fixture: BrowserAppFixture? = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake))
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
                        val benchmarkRerankStart = fake.rerankRequests.size
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
                        assertEquals(30, benchmarkModels.size)
                        assertEquals(20, fake.rerankRequests.size - benchmarkRerankStart)
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
                        fixture = BrowserAppFixture(paths, fake, modelConfiguration(paths, fake))
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

    private fun indexSnapshot(database: Path): Pair<List<List<String>>, List<List<String>>> =
        DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}").use { connection ->
            val sources = connection.createStatement().use { statement ->
                statement.executeQuery(
                    "SELECT source_id,name,type,size_bytes,status,content_sha256 FROM sources ORDER BY source_id",
                ).use { rows ->
                    buildList {
                        while (rows.next()) add((1..6).map { rows.getString(it) ?: "<NULL>" })
                    }
                }
            }
            val chunks = connection.createStatement().use { statement ->
                statement.executeQuery(
                    """SELECT chunk_id,source_id,strategy,section,page_start,page_end,line_start,line_end,
                        token_units,text,hex(embedding),embedding_model,embedding_dimension
                        FROM chunks ORDER BY source_id,chunk_id""".trimIndent(),
                ).use { rows ->
                    buildList {
                        while (rows.next()) add((1..13).map { rows.getString(it) ?: "<NULL>" })
                    }
                }
            }
            sources to chunks
        }

    private fun sourceFilesSnapshot(directory: Path): Map<String, List<Byte>> =
        Files.list(directory).use { files ->
            files.sorted().map { path ->
                path.fileName.toString() to Files.readAllBytes(path).toList()
            }.toList().toMap()
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
                assertEquals(2, queryMessages.size)
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
                assertEquals(2, pdfMessages.size)
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
                    assertEquals(2, codeMessages.size)
                    assertTrue(codeMessages.last().any { "LIMIT = 19" in it })

                    val textChatStart = fake.cloud.chatMessages.size
                    page.locator("#question").fill("What beacon threshold is recorded in the note?")
                    page.locator("#query-button").click()
                    assertThat(queryResults).containsText("version-note.txt")
                    assertThat(queryResults).containsText("строки 1")
                    val textMessages = fake.cloud.chatMessages.drop(textChatStart)
                    assertEquals(2, textMessages.size)
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

                val benchmarkStart = page.waitForResponse("**/api/benchmark/run") {
                    page.locator("#benchmark-run").click()
                }
                assertEquals(202, benchmarkStart.status())
                page.locator("#benchmark-error").waitFor(
                    com.microsoft.playwright.Locator.WaitForOptions().setTimeout(30_000.0),
                )
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

    private fun modelConfiguration(paths: LocalRagPaths, fake: FakeOllama): ModelConfiguration {
        val selection = ModelSelection("fixture", "deepseek-flash")
        val catalog = ProviderCatalog.create(
            listOf(
                ProviderDefinition(
                    "fixture",
                    "Fixture provider",
                    fake.cloud.uri.toString(),
                    "/chat/completions",
                    "FIXTURE_API_KEY",
                    listOf(
                        ProviderModelDefinition(selection.modelId, "Fixture Flash"),
                        ProviderModelDefinition("deepseek-v4-pro", "Fixture V4 Pro"),
                    ),
                ),
                ProviderDefinition(
                    "ollama",
                    "Local / Ollama",
                    fake.uri.toString(),
                    "/v1/chat/completions",
                    null,
                    listOf(
                        ProviderModelDefinition("qwen2.5:3b", "Qwen 2.5 3B"),
                        ProviderModelDefinition("qwen2.5:7b", "Qwen 2.5 7B"),
                    ),
                    ProviderKind.OLLAMA,
                ),
            ),
            selection,
            allowLoopbackHttpForTests = true,
        )
        return ModelConfiguration(
            catalog,
            ModelSelectionStore(paths.modelSelection, catalog),
            ProviderCredentialSource { if (fake.cloud.credentialConfigured.get()) "fixture-secret" else null },
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
        private val rag = RagService(index, embeddings, chat, OllamaReranker(ollama))
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
        val failReranker = AtomicBoolean(false)
        val failLocalGeneration = AtomicBoolean(false)
        val rerankRequests = CopyOnWriteArrayList<JsonObject>()
        val localGenerationCalls = AtomicInteger()
        val localModels = CopyOnWriteArrayList<String>()
        val localChatPayloads = CopyOnWriteArrayList<JsonObject>()
        val embeddingInputs = CopyOnWriteArrayList<String>()
        val embeddingCalls = AtomicInteger()
        val holdPdfEmbedding = AtomicBoolean(false)
        val pdfEmbeddingStarted = CountDownLatch(1)
        val releasePdfEmbedding = CountDownLatch(1)
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeOllama.executor
            createContext("/api/tags") { exchange ->
                val models = """{"models":[{"name":"embeddinggemma:300m"},{"name":"qwen2.5:3b"},{"name":"qwen2.5:7b"}]}"""
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
            createContext("/v1/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                localModels += request.getValue("model").jsonPrimitive.content
                val messages = request.getValue("messages").jsonArray.map {
                    it.jsonObject.getValue("content").jsonPrimitive.content
                }
                val isReranker = messages.firstOrNull()?.contains("локальный reranker") == true
                val content = if (isReranker) {
                    val payload = Json.parseToJsonElement(messages.last()).jsonObject
                    rerankRequests += payload
                    if (failReranker.compareAndSet(true, false)) {
                        respond(exchange, 503, """{"error":"SYNTHETIC_LOCAL_RERANK_FAILURE"}""")
                        return@createContext
                    }
                    val count = payload.getValue("candidates").jsonArray.size
                    buildJsonObject {
                        put("ranked_indices", JsonArray((count - 1 downTo 0).map { JsonPrimitive(it) }))
                    }.toString()
                } else {
                    localGenerationCalls.incrementAndGet()
                    if (failLocalGeneration.compareAndSet(true, false)) {
                        respond(exchange, 503, """{"error":"SYNTHETIC_LOCAL_GENERATION_FAILURE"}""")
                        return@createContext
                    }
                    if (request.containsKey("response_format")) {
                        val payload = Json.parseToJsonElement(messages.last()).jsonObject
                        localChatPayloads += payload
                        val documentCandidates = payload.getValue("document_candidates").jsonArray
                        buildJsonObject {
                            put("answer", if (documentCandidates.isEmpty()) ChatService.ABSTENTION else "Synthetic local chat answer.")
                            put("task_state_delta", buildJsonObject {
                                put("goal", kotlinx.serialization.json.JsonNull)
                                put("clarifications_to_add", buildJsonArray { })
                                put("constraints_to_add", buildJsonArray { })
                                put("terms_to_add", buildJsonArray { })
                            })
                            put("memory_updates", buildJsonArray { })
                            put("document_citation_refs", buildJsonArray {
                                documentCandidates.firstOrNull()?.jsonObject?.get("ref_id")?.let { add(it) }
                            })
                            put("memory_reference_refs", buildJsonArray { })
                            put(
                                "session_title",
                                if (payload.getValue("generate_title").jsonPrimitive.content.toBoolean()) {
                                    JsonPrimitive("Local synthetic session")
                                } else {
                                    kotlinx.serialization.json.JsonNull
                                },
                            )
                        }.toString()
                    } else {
                        "Synthetic local answer."
                    }
                }
                val response = buildJsonObject {
                    put("choices", buildJsonArray {
                        add(buildJsonObject {
                            put("message", buildJsonObject { put("content", JsonPrimitive(content)) })
                        })
                    })
                }.toString()
                respond(exchange, 200, response)
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
        val credentialConfigured = AtomicBoolean(true)
        val chatMessages = CopyOnWriteArrayList<List<String>>()
        val chatRequests = CopyOnWriteArrayList<JsonObject>()
        val titleGenerationRequests = AtomicInteger(0)
        val chatModels = CopyOnWriteArrayList<String>()
        val holdNextChat = AtomicBoolean(false)
        val firstChatStarted = CountDownLatch(1)
        val releaseFirstChat = CountDownLatch(1)
        private val executor = Executors.newCachedThreadPool()
        private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            this.executor = this@FakeDeepSeek.executor
            createContext("/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(exchange.requestBody.bufferedReader(UTF_8).use { it.readText() }).jsonObject
                chatRequests.add(request)
                val messages = request.getValue("messages").jsonArray.map {
                    it.jsonObject.getValue("content").jsonPrimitive.content
                }
                chatMessages.add(messages)
                chatModels.add(request.getValue("model").jsonPrimitive.content)
                if (holdNextChat.compareAndSet(true, false)) {
                    firstChatStarted.countDown()
                    check(releaseFirstChat.await(20, TimeUnit.SECONDS))
                }
                if (failChat.get()) {
                    respond(exchange, 503, """{"error":"Synthetic cloud failure"}""")
                } else if (request.containsKey("response_format")) {
                    val payload = Json.parseToJsonElement(messages.last()).jsonObject
                    val question = payload.getValue("question").jsonPrimitive.content
                    val generateTitle = payload.getValue("generate_title").jsonPrimitive.content.toBoolean()
                    val titleRequestNumber = if (generateTitle) titleGenerationRequests.incrementAndGet() else 0
                    val currentGoal = payload.getValue("task_state").jsonObject.getValue("goal")
                    val firstSessionA = currentGoal == kotlinx.serialization.json.JsonNull && question.contains("session-A", ignoreCase = true)
                    val firstSessionB = currentGoal == kotlinx.serialization.json.JsonNull && question.contains("session-B", ignoreCase = true)
                    val answerGoal = when {
                        firstSessionA -> "Goal A"
                        firstSessionB -> "Goal B"
                        question.contains("goal-shift", ignoreCase = true) -> "Updated synthetic goal"
                        currentGoal != kotlinx.serialization.json.JsonNull -> currentGoal.jsonPrimitive.content
                        else -> null
                    }
                    val delta = buildJsonObject {
                        put("goal", when {
                            firstSessionA -> JsonPrimitive("Goal A")
                            firstSessionB -> JsonPrimitive("Goal B")
                            question.contains("goal-shift", ignoreCase = true) -> JsonPrimitive("Updated synthetic goal")
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
                    val sessionMemory = payload.getValue("session_memory").jsonArray
                    val sharedMemory = payload.getValue("shared_memory").jsonArray
                    val documentCandidates = payload.getValue("document_candidates").jsonArray
                    val memoryUpdates = buildJsonArray {
                        if (firstSessionA) {
                            add(buildJsonObject {
                                put("scope", "SHARED")
                                put("content", "Цель задачи: Goal A")
                            })
                        }
                        if (question.contains("auto-memory-seed", ignoreCase = true)) {
                            add(buildJsonObject {
                                put("scope", "SESSION")
                                put("content", "Session-only synthetic marker")
                            })
                            add(buildJsonObject {
                                put("scope", "SHARED")
                                put("content", "Shared synthetic preference")
                            })
                        }
                    }
                    val selectedMemory = when {
                        firstSessionB -> sharedMemory.firstOrNull {
                            it.jsonObject.getValue("text").jsonPrimitive.content == "Shared synthetic preference"
                        }
                        question.contains("memory", ignoreCase = true) ->
                            (sessionMemory + sharedMemory).firstOrNull()
                        else -> null
                    }
                    val selectedDocument = when {
                        question.contains("Unsupported synthetic factual question", ignoreCase = true) -> null
                        question.contains("payment date", ignoreCase = true) -> documentCandidates.firstOrNull {
                            "payment date" in it.jsonObject.getValue("text").jsonPrimitive.content.lowercase()
                        } ?: documentCandidates.firstOrNull()
                        else -> documentCandidates.firstOrNull()
                    }
                    val structuredContent = if (malformedStructuredResponse.compareAndSet(true, false)) {
                        """{"answer":"Malformed synthetic answer","task_state_delta":{"goal":"broken"}}"""
                    } else {
                        buildJsonObject {
                            val answer = when {
                                question.contains("Unsupported synthetic factual question", ignoreCase = true) ->
                                    ChatService.ABSTENTION
                                question.contains("conversation-answer", ignoreCase = true) ->
                                    "Synthetic conversational answer."
                                documentCandidates.isNotEmpty() || selectedMemory != null ->
                                    answerGoal?.let { "Synthetic chat answer. Current goal: $it" } ?: "Synthetic chat answer."
                                else -> ChatService.ABSTENTION
                            }
                            put("answer", answer)
                            put("task_state_delta", delta)
                            put("memory_updates", memoryUpdates)
                            put("document_citation_refs", buildJsonArray {
                                selectedDocument?.jsonObject?.get("ref_id")?.let { add(it) }
                            })
                            put("memory_reference_refs", buildJsonArray {
                                selectedMemory?.jsonObject?.get("ref_id")?.let { add(it) }
                            })
                            put(
                                "session_title",
                                if (generateTitle) {
                                    JsonPrimitive("Synthetic context summary ${question.take(24)} #$titleRequestNumber")
                                } else {
                                    kotlinx.serialization.json.JsonNull
                                },
                            )
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
