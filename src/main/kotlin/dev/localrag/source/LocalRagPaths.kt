package dev.localrag.source

import java.nio.file.Path

 data class LocalRagPaths(val dataDirectory: Path) {
    val database: Path get() = dataDirectory.resolve("local-rag-v11.sqlite")
    val sources: Path get() = dataDirectory.resolve("sources")

    companion object {
        fun current(): LocalRagPaths {
            val configured = System.getenv("LOCAL_RAG_DATA_DIR")?.trim()?.takeIf(String::isNotEmpty)
            val root = configured?.let(Path::of)
                ?: Path.of(System.getProperty("user.home"), ".local", "share", "local-rag")
            return LocalRagPaths(root.toAbsolutePath().normalize())
        }
    }
}
