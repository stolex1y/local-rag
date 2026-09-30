package dev.localrag.generation

import dev.localrag.domain.ModelSelection
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.PosixFilePermission

@Serializable
private data class StoredModelSelection(val providerId: String, val modelId: String)

private data class SelectionState(val selection: ModelSelection, val warning: String? = null)

class ModelSelectionStore(
    path: Path,
    private val catalog: ProviderCatalog,
) {
    private val path = path.toAbsolutePath().normalize()
    private val lock = Any()
    private var state = load()

    fun currentSelection(): ModelSelection = synchronized(lock) { state.selection }

    fun warning(): String? = synchronized(lock) { state.warning }

    fun select(selection: ModelSelection) = synchronized(lock) {
        catalog.requireSelection(selection)
        persist(selection)
        state = SelectionState(selection)
    }

    private fun load(): SelectionState {
        if (!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) return SelectionState(catalog.defaultSelection)
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)) {
            return SelectionState(catalog.defaultSelection, "Локальная настройка модели недоступна; выбрано значение по умолчанию.")
        }
        return try {
            val stored = Json.decodeFromString<StoredModelSelection>(Files.readString(path))
            val selection = ModelSelection(stored.providerId, stored.modelId)
            catalog.requireSelection(selection)
            SelectionState(selection)
        } catch (_: Exception) {
            SelectionState(catalog.defaultSelection, "Сохранённая модель больше не входит в каталог; выбрано значение по умолчанию.")
        }
    }

    private fun persist(selection: ModelSelection) {
        val parent = path.parent ?: throw IllegalStateException("Model settings path must have a parent directory.")
        Files.createDirectories(parent)
        setPermissions(parent, setOf(
            PosixFilePermission.OWNER_READ,
            PosixFilePermission.OWNER_WRITE,
            PosixFilePermission.OWNER_EXECUTE,
        ))
        val temporary = Files.createTempFile(parent, ".model-selection-", ".tmp")
        try {
            Files.writeString(temporary, Json.encodeToString(StoredModelSelection(selection.providerId, selection.modelId)))
            setPermissions(temporary, setOf(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE))
            try {
                Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            Files.deleteIfExists(temporary)
        }
    }

    private fun setPermissions(path: Path, permissions: Set<PosixFilePermission>) {
        try {
            Files.setPosixFilePermissions(path, permissions)
        } catch (_: UnsupportedOperationException) {
            // The platform does not expose POSIX permissions.
        }
    }
}
