package dev.localrag.generation

import dev.localrag.domain.ModelSelection
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertFailsWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertTrue
import org.junit.jupiter.api.io.TempDir

class ModelSelectionStoreTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `selected provider and model persist while first use keeps configured defaults`() {
        val catalog = catalog()
        val settingsPath = temporaryDirectory.resolve("settings/model-selection.json")
        val first = ModelSelectionStore(settingsPath, catalog)
        assertEquals(catalog.defaultSelection, first.currentSelection())
        assertNull(first.warning())

        val selected = ModelSelection("deepseek", "deepseek-v4-pro")
        first.select(selected)

        val restarted = ModelSelectionStore(settingsPath, catalog)
        assertEquals(selected, restarted.currentSelection())
        assertNull(restarted.warning())
    }

    @Test
    fun `invalid saved model falls back to catalog default and reports warning`() {
        val catalog = catalog()
        val settingsPath = temporaryDirectory.resolve("model-selection.json")
        Files.writeString(settingsPath, """{"providerId":"deepseek","modelId":"removed-model"}""")

        val store = ModelSelectionStore(settingsPath, catalog)

        assertEquals(catalog.defaultSelection, store.currentSelection())
        assertNotNull(store.warning())
    }

    @Test
    fun `production catalog includes cloud and local generation models`() {
        val catalog = ProviderCatalog.load(Path.of("config/providers.json"), Path.of("config/agent.json"))
        val cloud = catalog.providers.single { it.id == "deepseek" }
        val ollama = catalog.providers.single { it.id == "ollama" }

        assertEquals("https://api.deepseek.com", cloud.baseUrl)
        assertEquals("/chat/completions", cloud.chatCompletionsPath)
        assertEquals("DEEPSEEK_API_KEY", cloud.credentialEnv)
        assertEquals(listOf("deepseek-flash", "deepseek-v4-pro"), cloud.models.map { it.id })
        assertEquals("http://127.0.0.1:11434", ollama.baseUrl)
        assertEquals(ProviderKind.OLLAMA, ollama.kind)
        assertNull(ollama.credentialEnv)
        assertEquals(listOf("qwen2.5:3b", "qwen2.5:3b-instruct-q5_K_M", "qwen2.5:7b"), ollama.models.map { it.id })
        assertEquals(ModelSelection("deepseek", "deepseek-flash"), catalog.defaultSelection)
    }

    @Test
    fun `provider model options reflect Ollama tags and cloud credentials`() {
        val catalog = catalog()
        val configuration = ModelConfiguration(
            catalog,
            ModelSelectionStore(temporaryDirectory.resolve("options.json"), catalog),
            ProviderCredentialSource { "synthetic-cloud-key" },
        )
        configuration.select(ModelSelection("ollama", "qwen2.5:7b"))

        val response = configuration.options(setOf("qwen2.5:7b"))

        assertTrue(response.selectedModelAvailable)
        assertEquals(
            listOf(false, true),
            response.providers.single { it.id == "ollama" }.models.map { it.available },
        )
        assertTrue(response.providers.single { it.id == "deepseek" }.models.all { it.available })
    }

    @Test
    fun `invalid defaults and selections are rejected`() {
        val valid = catalog()
        assertFailsWith<ModelConfigurationException> {
            ProviderCatalog.create(valid.providers, ModelSelection("deepseek", "removed-model"))
        }

        val settingsPath = temporaryDirectory.resolve("invalid-selection.json")
        val store = ModelSelectionStore(settingsPath, valid)
        assertFailsWith<IllegalArgumentException> {
            store.select(ModelSelection("deepseek", "removed-model"))
        }
        assertEquals(valid.defaultSelection, store.currentSelection())
        assertEquals(false, Files.exists(settingsPath))
    }

    @Test
    fun `production catalog load rejects loopback HTTP`() {
        val providersPath = temporaryDirectory.resolve("providers.json")
        val agentPath = temporaryDirectory.resolve("agent.json")
        Files.writeString(
            providersPath,
            """{"providers":[{"id":"deepseek","display_name":"DeepSeek","base_url":"http://127.0.0.1:8080","chat_completions_path":"/chat/completions","credential_env":"DEEPSEEK_API_KEY","models":[{"id":"deepseek-flash","display_name":"DeepSeek Flash"}]}]}""",
        )
        Files.writeString(agentPath, """{"default_provider_id":"deepseek","default_model_id":"deepseek-flash"}""")

        assertFailsWith<IllegalArgumentException> { ProviderCatalog.load(providersPath, agentPath) }
    }

    @Test
    fun `concurrent selections leave in-memory and persisted values aligned`() {
        val catalog = catalog()
        val settingsPath = temporaryDirectory.resolve("concurrent/model-selection.json")
        val store = ModelSelectionStore(settingsPath, catalog)
        val selections = listOf(
            ModelSelection("deepseek", "deepseek-flash"),
            ModelSelection("deepseek", "deepseek-v4-pro"),
        )
        val writers = 12
        val ready = CountDownLatch(writers)
        val start = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(writers)

        try {
            val updates = (0 until writers).map { writer ->
                executor.submit {
                    ready.countDown()
                    check(start.await(10, TimeUnit.SECONDS))
                    repeat(25) { iteration ->
                        store.select(selections[(writer + iteration) % selections.size])
                    }
                }
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS))
            start.countDown()
            updates.forEach { it.get(30, TimeUnit.SECONDS) }

            assertEquals(store.currentSelection(), ModelSelectionStore(settingsPath, catalog).currentSelection())
        } finally {
            executor.shutdownNow()
        }
    }

    private fun catalog() = ProviderCatalog.create(
        listOf(
            ProviderDefinition(
                id = "deepseek",
                displayName = "DeepSeek",
                baseUrl = "https://api.deepseek.com",
                chatCompletionsPath = "/chat/completions",
                credentialEnv = "DEEPSEEK_API_KEY",
                models = listOf(
                    ProviderModelDefinition("deepseek-flash", "DeepSeek Flash"),
                    ProviderModelDefinition("deepseek-v4-pro", "DeepSeek V4 Pro"),
                ),
            ),
            ProviderDefinition(
                id = "ollama",
                displayName = "Local / Ollama",
                baseUrl = "http://127.0.0.1:11434",
                chatCompletionsPath = "/v1/chat/completions",
                credentialEnv = null,
                models = listOf(
                    ProviderModelDefinition("qwen2.5:3b", "Qwen 2.5 3B"),
                    ProviderModelDefinition("qwen2.5:7b", "Qwen 2.5 7B"),
                ),
                kind = ProviderKind.OLLAMA,
            ),
        ),
        ModelSelection("deepseek", "deepseek-flash"),
    )
}
