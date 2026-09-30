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
    fun `production catalog contains only approved DeepSeek models`() {
        val catalog = ProviderCatalog.load(Path.of("config/providers.json"), Path.of("config/agent.json"))
        val provider = catalog.providers.single()

        assertEquals("deepseek", provider.id)
        assertEquals("https://api.deepseek.com", provider.baseUrl)
        assertEquals("/chat/completions", provider.chatCompletionsPath)
        assertEquals("DEEPSEEK_API_KEY", provider.credentialEnv)
        assertEquals(listOf("deepseek-flash", "deepseek-v4-pro"), provider.models.map { it.id })
        assertEquals(ModelSelection("deepseek", "deepseek-flash"), catalog.defaultSelection)
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
        ),
        ModelSelection("deepseek", "deepseek-flash"),
    )
}
