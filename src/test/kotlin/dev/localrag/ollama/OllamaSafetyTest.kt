package dev.localrag.ollama

import java.net.URI
import kotlin.test.Test
import kotlin.test.assertFailsWith

class OllamaSafetyTest {

    @Test
    fun `Ollama endpoint rejects non-loopback and non-HTTP services`() {
        listOf(
            "http://ollama.example:11434",
            "https://127.0.0.1:11434",
            "http://user@127.0.0.1:11434",
        ).forEach { endpoint ->
            assertFailsWith<IllegalArgumentException>(endpoint) { OllamaApi(URI(endpoint)) }
        }
    }

}
