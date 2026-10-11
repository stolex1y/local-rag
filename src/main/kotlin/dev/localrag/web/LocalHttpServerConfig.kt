package dev.localrag.web

import java.net.Inet4Address
import java.net.Inet6Address
import java.net.InetAddress

class LocalHttpServerConfig(
    val bindAddress: InetAddress,
    val port: Int,
) {
    init {
        require(port in 0..65535) { "Порт Local RAG должен быть от 0 до 65535." }
        require(!bindAddress.isAnyLocalAddress) {
            "LOCAL_RAG_BIND_ADDRESS должен быть конкретным адресом интерфейса, не 0.0.0.0/::."
        }
        if (!bindAddress.isLoopbackAddress) {
            require(bindAddress is Inet4Address && bindAddress.hostAddress == LAN_BIND_ADDRESS) {
                "LAN bind разрешён только на Wi-Fi-IP $LAN_BIND_ADDRESS."
            }
            require(port == LAN_PORT) { "LAN bind разрешён только на порт $LAN_PORT." }
        }
    }

    fun originFor(actualPort: Int): String = "http://${formatHost(bindAddress)}:$actualPort"

    fun acceptsHost(hostHeader: String?, actualPort: Int): Boolean {
        val host = hostHeader?.trim()?.lowercase() ?: return false
        val expected = "${formatHost(bindAddress)}:$actualPort".lowercase()
        if (host == expected) return true
        return bindAddress.isLoopbackAddress && host == "localhost:$actualPort"
    }

    fun acceptsOrigin(origin: String?, actualPort: Int): Boolean {
        if (origin == null) return false
        val accepted = if (bindAddress.isLoopbackAddress) {
            setOf(originFor(actualPort), "http://localhost:$actualPort")
        } else {
            setOf(originFor(actualPort))
        }
        return accepted.any { it.equals(origin, ignoreCase = true) }
    }

    companion object {
        fun fromEnvironment(): LocalHttpServerConfig {
            val bindText = System.getenv("LOCAL_RAG_BIND_ADDRESS")?.trim()?.takeIf(String::isNotEmpty)
                ?: "127.0.0.1"
            val bindAddress = InetAddress.getByName(bindText)
            val defaultPort = if (bindAddress.isLoopbackAddress) DEFAULT_LOOPBACK_PORT else LAN_PORT
            val port = System.getenv("LOCAL_RAG_PORT")?.toIntOrNull() ?: defaultPort
            return LocalHttpServerConfig(bindAddress, port)
        }

        private const val LAN_BIND_ADDRESS = "192.168.2.160"
        private const val LAN_PORT = 8767
        private const val DEFAULT_LOOPBACK_PORT = 8765

        private fun formatHost(address: InetAddress): String = when (address) {
            is Inet4Address -> address.hostAddress
            is Inet6Address -> "[${address.hostAddress.substringBefore('%')}]"
        }
    }
}
