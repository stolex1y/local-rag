package dev.localrag.generation

import dev.localrag.domain.ModelSelection
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path

@Serializable
data class ProviderCatalogDocument(val providers: List<ProviderDefinition>)

@Serializable
data class ProviderDefinition(
    val id: String,
    @SerialName("display_name") val displayName: String,
    @SerialName("base_url") val baseUrl: String,
    @SerialName("chat_completions_path") val chatCompletionsPath: String,
    @SerialName("credential_env") val credentialEnv: String,
    val models: List<ProviderModelDefinition>,
)

@Serializable
data class ProviderModelDefinition(
    val id: String,
    @SerialName("display_name") val displayName: String,
)

@Serializable
data class AgentDefaults(
    @SerialName("default_provider_id") val defaultProviderId: String,
    @SerialName("default_model_id") val defaultModelId: String,
)

@Serializable
data class ModelOption(val id: String, val displayName: String)

@Serializable
data class ProviderOption(
    val id: String,
    val displayName: String,
    val models: List<ModelOption>,
)

@Serializable
data class ModelConfigurationResponse(
    val providers: List<ProviderOption>,
    val defaultProviderId: String,
    val defaultModelId: String,
    val selectedProviderId: String,
    val selectedModelId: String,
    val credentialEnv: String,
    val credentialConfigured: Boolean,
    val warning: String? = null,
)

fun interface ProviderCredentialSource {
    fun read(environmentName: String): String?
}

object EnvironmentProviderCredentials : ProviderCredentialSource {
    override fun read(environmentName: String): String? = System.getenv(environmentName)
}

class ModelConfigurationException(message: String) : RuntimeException(message)

class ProviderCatalog private constructor(
    val providers: List<ProviderDefinition>,
    val defaultSelection: ModelSelection,
) {
    private val byId = providers.associateBy(ProviderDefinition::id)

    fun requireSelection(selection: ModelSelection) {
        val provider = byId[selection.providerId]
            ?: throw IllegalArgumentException("Выберите провайдера из настроенного каталога.")
        require(provider.models.any { it.id == selection.modelId }) {
            "Выберите модель из настроенного каталога провайдера."
        }
    }

    fun resolve(selection: ModelSelection): ResolvedProviderModel {
        requireSelection(selection)
        val provider = byId.getValue(selection.providerId)
        return ResolvedProviderModel(provider, provider.models.first { it.id == selection.modelId })
    }

    fun options(): List<ProviderOption> = providers.map { provider ->
        ProviderOption(
            id = provider.id,
            displayName = provider.displayName,
            models = provider.models.map { ModelOption(it.id, it.displayName) },
        )
    }

    companion object {
        private val json = Json

        fun load(providersPath: Path, agentPath: Path): ProviderCatalog {
            val providers = try {
                json.decodeFromString<ProviderCatalogDocument>(Files.readString(providersPath)).providers
            } catch (_: Exception) {
                throw ModelConfigurationException("Не удалось прочитать config/providers.json.")
            }
            val defaults = try {
                json.decodeFromString<AgentDefaults>(Files.readString(agentPath))
            } catch (_: Exception) {
                throw ModelConfigurationException("Не удалось прочитать config/agent.json.")
            }
            return create(
                providers,
                ModelSelection(defaults.defaultProviderId, defaults.defaultModelId),
            )
        }

        fun create(
            providers: List<ProviderDefinition>,
            defaultSelection: ModelSelection,
            allowLoopbackHttpForTests: Boolean = false,
        ): ProviderCatalog {
            require(providers.isNotEmpty()) { "Каталог моделей не должен быть пустым." }
            require(providers.map(ProviderDefinition::id).toSet().size == providers.size) {
                "Идентификаторы провайдеров должны быть уникальными."
            }
            providers.forEach { validateProvider(it, allowLoopbackHttpForTests) }
            val catalog = ProviderCatalog(providers.toList(), defaultSelection)
            try {
                catalog.requireSelection(defaultSelection)
            } catch (error: IllegalArgumentException) {
                throw ModelConfigurationException("Значения defaults в config/agent.json отсутствуют в config/providers.json.")
            }
            return catalog
        }

        private fun validateProvider(provider: ProviderDefinition, allowLoopbackHttpForTests: Boolean) {
            require(provider.id.matches(Regex("[a-z][a-z0-9-]{0,63}"))) { "Некорректный provider id." }
            require(provider.displayName.isNotBlank() && provider.displayName.length <= 80) { "Некорректное display name провайдера." }
            val base = try {
                URI.create(provider.baseUrl)
            } catch (_: Exception) {
                throw IllegalArgumentException("Некорректный base_url провайдера.")
            }
            val loopbackHttp = allowLoopbackHttpForTests && base.scheme == "http" &&
                base.host in setOf("127.0.0.1", "localhost", "::1")
            require(base.isAbsolute && (base.scheme == "https" || loopbackHttp) && base.host != null) {
                "Provider base_url должен использовать HTTPS; HTTP допустим только для loopback fake-сервиса."
            }
            require(base.userInfo == null && base.query == null && base.fragment == null &&
                (base.path.isNullOrEmpty() || base.path == "/") && base.port in -1..65535 && base.port != 0) {
                "Provider base_url содержит недопустимые компоненты."
            }
            val endpoint = try {
                URI.create(provider.chatCompletionsPath)
            } catch (_: Exception) {
                throw IllegalArgumentException("Некорректный chat_completions_path провайдера.")
            }
            require(endpoint.scheme == null && endpoint.rawAuthority == null && endpoint.path.startsWith("/") &&
                endpoint.query == null && endpoint.fragment == null && !endpoint.path.split('/').contains("..")) {
                "chat_completions_path должен быть абсолютным путём без host, query и traversal."
            }
            require(provider.credentialEnv.matches(Regex("[A-Z][A-Z0-9_]{0,127}"))) {
                "Некорректное имя credential environment variable."
            }
            require(provider.models.isNotEmpty()) { "Провайдер должен содержать хотя бы одну модель." }
            require(provider.models.map(ProviderModelDefinition::id).toSet().size == provider.models.size) {
                "Идентификаторы моделей должны быть уникальными у провайдера."
            }
            provider.models.forEach { model ->
                require(model.id.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}"))) { "Некорректный model id." }
                require(model.displayName.isNotBlank() && model.displayName.length <= 120) { "Некорректное display name модели." }
            }
        }
    }
}

data class ResolvedProviderModel(
    val provider: ProviderDefinition,
    val model: ProviderModelDefinition,
)

class ModelConfiguration(
    val catalog: ProviderCatalog,
    private val selections: ModelSelectionStore,
    private val credentials: ProviderCredentialSource,
) {
    fun currentSelection(): ModelSelection = selections.currentSelection()

    fun select(selection: ModelSelection) {
        catalog.requireSelection(selection)
        selections.select(selection)
    }

    fun options(): ModelConfigurationResponse {
        val selection = currentSelection()
        val provider = catalog.resolve(selection).provider
        return ModelConfigurationResponse(
            providers = catalog.options(),
            defaultProviderId = catalog.defaultSelection.providerId,
            defaultModelId = catalog.defaultSelection.modelId,
            selectedProviderId = selection.providerId,
            selectedModelId = selection.modelId,
            credentialEnv = provider.credentialEnv,
            credentialConfigured = hasCredential(selection),
            warning = selections.warning(),
        )
    }

    fun hasCredential(selection: ModelSelection = currentSelection()): Boolean {
        val envName = catalog.resolve(selection).provider.credentialEnv
        return !credentials.read(envName).isNullOrBlank()
    }

    fun requireCredential(selection: ModelSelection) {
        val envName = catalog.resolve(selection).provider.credentialEnv
        if (credentials.read(envName).isNullOrBlank()) {
            throw CloudModelException("Для генерации задайте переменную окружения $envName и перезапустите приложение.")
        }
    }

    fun credential(selection: ModelSelection): String? =
        credentials.read(catalog.resolve(selection).provider.credentialEnv)?.takeIf(String::isNotBlank)
}
