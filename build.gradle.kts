import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.gradle.api.tasks.testing.Test

plugins {
    kotlin("jvm") version "2.2.21"
    kotlin("plugin.serialization") version "2.2.21"
    application
}

group = "dev.localrag"
version = "0.1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

application {
    mainClass.set("dev.localrag.MainKt")
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
    implementation("org.xerial:sqlite-jdbc:3.46.1.0")

    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")
    testImplementation("com.microsoft.playwright:playwright:1.62.0")
}

tasks.test {
    useJUnitPlatform { excludeTags("browser") }
}

tasks.register<Test>("browserTest") {
    group = "verification"
    description = "Runs isolated Playwright acceptance tests."
    dependsOn("testClasses")
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    outputs.upToDateWhen { false }
    useJUnitPlatform { includeTags("browser") }
}
