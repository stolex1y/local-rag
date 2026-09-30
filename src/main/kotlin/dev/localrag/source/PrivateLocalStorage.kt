package dev.localrag.source

import java.io.IOException
import java.nio.file.FileAlreadyExistsException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermission
import java.nio.file.attribute.PosixFilePermissions

internal object PrivateLocalStorage {
    private val DIRECTORY_PERMISSIONS = PosixFilePermissions.fromString("rwx------")
    private val FILE_PERMISSIONS = PosixFilePermissions.fromString("rw-------")

    fun prepareDirectory(path: Path) {
        val directory = path.toAbsolutePath().normalize()
        try {
            Files.createDirectories(directory, PosixFilePermissions.asFileAttribute(DIRECTORY_PERMISSIONS))
        } catch (_: UnsupportedOperationException) {
            Files.createDirectories(directory)
        }
        restrict(directory, isDirectory = true)
    }

    fun createPrivateFile(path: Path) {
        val file = path.toAbsolutePath().normalize()
        try {
            Files.createFile(file, PosixFilePermissions.asFileAttribute(FILE_PERMISSIONS))
        } catch (_: UnsupportedOperationException) {
            if (!Files.exists(file, LinkOption.NOFOLLOW_LINKS)) Files.createFile(file)
        }
        restrict(file, isDirectory = false)
    }

    fun prepareFile(path: Path) {
        val file = path.toAbsolutePath().normalize()
        if (!Files.exists(file, LinkOption.NOFOLLOW_LINKS)) {
            try {
                Files.createFile(file, PosixFilePermissions.asFileAttribute(FILE_PERMISSIONS))
            } catch (_: UnsupportedOperationException) {
                if (!Files.exists(file, LinkOption.NOFOLLOW_LINKS)) {
                    try {
                        Files.createFile(file)
                    } catch (_: FileAlreadyExistsException) {
                        // Another local process created the configured database first.
                    }
                }
            } catch (_: FileAlreadyExistsException) {
                // Another local process created the configured database first.
            }
        }
        restrict(file, isDirectory = false)
    }

    private fun restrict(path: Path, isDirectory: Boolean) {
        try {
            Files.setPosixFilePermissions(path, if (isDirectory) DIRECTORY_PERMISSIONS else FILE_PERMISSIONS)
        } catch (_: UnsupportedOperationException) {
            val file = path.toFile()
            val removedInheritedPermissions = file.setReadable(false, false) &&
                file.setWritable(false, false) &&
                file.setExecutable(false, false)
            val restoredOwnerPermissions = file.setReadable(true, true) &&
                file.setWritable(true, true) &&
                (!isDirectory || file.setExecutable(true, true))
            if (!removedInheritedPermissions || !restoredOwnerPermissions) {
                throw IOException("Не удалось ограничить локальное хранилище владельцем приложения.")
            }
        }
    }
}
