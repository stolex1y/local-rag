package dev.localrag.web

import java.net.InetAddress
import java.net.URI
import java.nio.file.Path

class LocalHttpServerConfig(
    val bindAddress: InetAddress,
    val port: Int,
    val publicOrigin: URI?,
    val keyStorePath: Path?,
    keyStorePassword: CharArray?,
    sharedSecret: String?,
) {
    internal val keyStorePassword: CharArray? = keyStorePassword?.copyOf()
    internal val sharedSecret: ByteArray? = sharedSecret?.toByteArray(Charsets.UTF_8)
    val usesHttps: Boolean get() = keyStorePath != null
    val requiresSharedSecret: Boolean get() = sharedSecret != null

    init {
        require(port in 0..65535) { "Порт Local RAG должен быть от 0 до 65535." }
        require((keyStorePath == null) == (keyStorePassword == null)) {
            "Для HTTPS задайте и путь к PKCS12 keystore, и его пароль."
        }
        require(!requiresSharedSecret || usesHttps) { "Общий секрет разрешён только вместе с HTTPS." }
        require(!requiresSharedSecret || requireNotNull(sharedSecret).toByteArray(Charsets.UTF_8).size >= MIN_SHARED_SECRET_BYTES) {
            "Общий секрет должен содержать не менее $MIN_SHARED_SECRET_BYTES байт."
        }
        require(publicOrigin == null || usesHttps) { "Публичный origin разрешён только для HTTPS." }
        if (!bindAddress.isLoopbackAddress) {
            require(usesHttps && requiresSharedSecret && publicOrigin != null) {
                "LAN-доступ требует HTTPS keystore, общий секрет и LOCAL_RAG_PUBLIC_ORIGIN."
            }
        }
        publicOrigin?.let { origin ->
            require(origin.scheme.equals("https", ignoreCase = true) && origin.host != null) {
                "LOCAL_RAG_PUBLIC_ORIGIN должен быть HTTPS origin с hostname."
            }
            require(origin.rawUserInfo == null && origin.rawPath.isNullOrEmpty() && origin.rawQuery == null && origin.rawFragment == null) {
                "LOCAL_RAG_PUBLIC_ORIGIN должен содержать только scheme, host и port."
            }
            require(origin.port == -1 || port == 0 || origin.port == port) {
                "Порт LOCAL_RAG_PUBLIC_ORIGIN должен совпадать с LOCAL_RAG_PORT."
            }
        }
    }

    fun originFor(actualPort: Int): String? {
        val host = publicOrigin?.host ?: return null
        val scheme = if (usesHttps) "https" else "http"
        val defaultPort = if (usesHttps) 443 else 80
        val usePort = publicOrigin.port.takeIf { it > 0 } ?: actualPort
        return if (usePort == defaultPort) "$scheme://$host" else "$scheme://$host:$usePort"
    }

    fun acceptsHost(hostHeader: String?, actualPort: Int, remoteAddress: InetAddress? = null): Boolean {
        val host = hostHeader?.trim()?.lowercase() ?: return false
        val publicHost = publicOrigin?.host?.lowercase()
        val allowed = if (publicHost != null) {
            buildSet {
                add("$publicHost:$actualPort")
                if ((usesHttps && actualPort == 443) || (!usesHttps && actualPort == 80)) add(publicHost)
            }
        } else {
            setOf("127.0.0.1:$actualPort", "localhost:$actualPort")
        }
        if (host in allowed) return true
        return usesHttps && remoteAddress?.isLoopbackAddress == true &&
            host in setOf("127.0.0.1:$actualPort", "localhost:$actualPort")
    }

    fun acceptsOrigin(origin: String?, actualPort: Int, remoteAddress: InetAddress? = null): Boolean {
        val expected = originFor(actualPort)
        val accepted = if (expected != null) {
            setOf(expected)
        } else {
            val scheme = if (usesHttps) "https" else "http"
            setOf("$scheme://127.0.0.1:$actualPort", "$scheme://localhost:$actualPort")
        }
        if (origin != null && accepted.any { it.equals(origin, ignoreCase = true) }) return true
        if (!usesHttps || remoteAddress?.isLoopbackAddress != true || origin == null) return false
        return setOf("https://127.0.0.1:$actualPort", "https://localhost:$actualPort")
            .any { it.equals(origin, ignoreCase = true) }
    }

    companion object {
        const val MIN_SHARED_SECRET_BYTES = 32

        fun fromEnvironment(): LocalHttpServerConfig {
            val bindText = System.getenv("LOCAL_RAG_BIND_ADDRESS")?.trim()?.takeIf(String::isNotEmpty)
                ?: "127.0.0.1"
            val bindAddress = InetAddress.getByName(bindText)
            val port = System.getenv("LOCAL_RAG_PORT")?.toIntOrNull() ?: 8765
            val keyStore = System.getenv("LOCAL_RAG_HTTPS_KEYSTORE")?.trim()?.takeIf(String::isNotEmpty)?.let(Path::of)
            val password = System.getenv("LOCAL_RAG_HTTPS_KEYSTORE_PASSWORD")?.takeIf(String::isNotEmpty)?.toCharArray()
            val secret = System.getenv("LOCAL_RAG_SHARED_SECRET")?.takeIf(String::isNotEmpty)
            val originText = System.getenv("LOCAL_RAG_PUBLIC_ORIGIN")?.trim()?.takeIf(String::isNotEmpty)
            val origin = originText?.let(URI::create)
            require(bindAddress.isLoopbackAddress || secret != null) {
                "Внешняя привязка запрещена без LOCAL_RAG_SHARED_SECRET."
            }
            return LocalHttpServerConfig(bindAddress, port, origin, keyStore, password, secret)
        }
    }
}
