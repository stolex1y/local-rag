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
enum class ProviderKind {
    CLOUD,
    OLLAMA,
}

@Serializable
data class ProviderCatalogDocument(val providers: List<ProviderDefinition>)

@Serializable
data class ProviderDefinition(
    val id: String,
    @SerialName("display_name") val displayName: String,
    @SerialName("base_url") val baseUrl: String,
    @SerialName("chat_completions_path") val chatCompletionsPath: String,
    @SerialName("credential_env") val credentialEnv: String?,
    val models: List<ProviderModelDefinition>,
    val kind: ProviderKind = ProviderKind.CLOUD,
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
data class ModelOption(val id: String, val displayName: String, val available: Boolean = true)

@Serializable
data class ProviderOption(
    val id: String,
    val displayName: String,
    val kind: ProviderKind,
    val models: List<ModelOption>,
)

@Serializable
data class ModelConfigurationResponse(
    val providers: List<ProviderOption>,
    val defaultProviderId: String,
    val defaultModelId: String,
    val selectedProviderId: String,
    val selectedModelId: String,
    val selectedModelAvailable: Boolean,
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
            kind = provider.kind,
            models = provider.models.map { ModelOption(it.id, it.displayName) },
        )
    }

    companion object {
        private val json = Json

        fun load(
            providersPath: Path,
            agentPath: Path,
            ollamaBaseUrl: URI = URI("http://127.0.0.1:11434"),
        ): ProviderCatalog {
            val cloudProviders = try {
                json.decodeFromString<ProviderCatalogDocument>(Files.readString(providersPath)).providers
            } catch (_: Exception) {
                throw ModelConfigurationException("Не удалось прочитать config/providers.json.")
            }
            val defaults = try {
                json.decodeFromString<AgentDefaults>(Files.readString(agentPath))
            } catch (_: Exception) {
                throw ModelConfigurationException("Не удалось прочитать config/agent.json.")
            }
            val ollama = ProviderDefinition(
                id = "ollama",
                displayName = "Local / Ollama",
                baseUrl = ollamaBaseUrl.toString(),
                chatCompletionsPath = "/v1/chat/completions",
                credentialEnv = null,
                models = listOf(
                    ProviderModelDefinition("qwen2.5:3b", "Qwen 2.5 3B (Q4_K_M)"),
                    ProviderModelDefinition("qwen2.5:3b-instruct-q5_K_M", "Qwen 2.5 3B (Q5_K_M)"),
                    ProviderModelDefinition("qwen2.5:7b", "Qwen 2.5 7B"),
                ),
                kind = ProviderKind.OLLAMA,
            )
            return create(
                cloudProviders + ollama,
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
            val isOllama = provider.kind == ProviderKind.OLLAMA
            val loopbackHttp = base.scheme == "http" && base.host == "127.0.0.1" &&
                (isOllama || allowLoopbackHttpForTests)
            require(base.isAbsolute && (base.scheme == "https" || loopbackHttp) && base.host != null) {
                "Provider base_url должен использовать HTTPS; HTTP разрешён только для Ollama loopback и loopback fake-сервиса."
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
            if (isOllama) {
                require(provider.id == "ollama" && provider.chatCompletionsPath == "/v1/chat/completions" &&
                    provider.credentialEnv == null) {
                    "Ollama provider должен использовать loopback OpenAI-compatible endpoint без credential."
                }
            } else {
                require(provider.credentialEnv?.matches(Regex("[A-Z][A-Z0-9_]{0,127}")) == true) {
                    "Некорректное имя credential environment variable."
                }
            }
            require(provider.models.isNotEmpty()) { "Провайдер должен содержать хотя бы одну модель." }
            require(provider.models.map(ProviderModelDefinition::id).toSet().size == provider.models.size) {
                "Идентификаторы моделей должны быть уникальными у провайдера."
            }
            provider.models.forEach { model ->
                require(model.id.matches(Regex("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"))) { "Некорректный model id." }
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

    fun options(availableLocalModels: Set<String> = emptySet()): ModelConfigurationResponse {
        val selection = currentSelection()
        val provider = catalog.resolve(selection).provider
        val credentialConfigured = hasCredential(selection)
        val availableFor = { candidate: ProviderDefinition ->
            when (candidate.kind) {
                ProviderKind.CLOUD -> candidate.credentialEnv?.let { !credentials.read(it).isNullOrBlank() } == true
                ProviderKind.OLLAMA -> candidate.models.any { it.id in availableLocalModels }
            }
        }
        val providerOptions = catalog.providers.map { candidate ->
            ProviderOption(
                id = candidate.id,
                displayName = candidate.displayName,
                kind = candidate.kind,
                models = candidate.models.map { model ->
                    val available = when (candidate.kind) {
                        ProviderKind.CLOUD -> availableFor(candidate)
                        ProviderKind.OLLAMA -> model.id in availableLocalModels
                    }
                    ModelOption(model.id, model.displayName, available)
                },
            )
        }
        val selectedModelAvailable = when (provider.kind) {
            ProviderKind.CLOUD -> credentialConfigured
            ProviderKind.OLLAMA -> selection.modelId in availableLocalModels
        }
        return ModelConfigurationResponse(
            providers = providerOptions,
            defaultProviderId = catalog.defaultSelection.providerId,
            defaultModelId = catalog.defaultSelection.modelId,
            selectedProviderId = selection.providerId,
            selectedModelId = selection.modelId,
            selectedModelAvailable = selectedModelAvailable,
            credentialEnv = provider.credentialEnv.orEmpty(),
            credentialConfigured = credentialConfigured,
            warning = selections.warning(),
        )
    }

    fun hasCredential(selection: ModelSelection = currentSelection()): Boolean {
        val provider = catalog.resolve(selection).provider
        val envName = provider.credentialEnv ?: return false
        return !credentials.read(envName).isNullOrBlank()
    }

    fun requireCredential(selection: ModelSelection) {
        val provider = catalog.resolve(selection).provider
        val envName = provider.credentialEnv ?: return
        if (credentials.read(envName).isNullOrBlank()) {
            throw CloudModelException("Для генерации задайте переменную окружения $envName и перезапустите приложение.")
        }
    }

    fun credential(selection: ModelSelection): String? {
        val provider = catalog.resolve(selection).provider
        return provider.credentialEnv?.let(credentials::read)?.takeIf(String::isNotBlank)
    }
}
