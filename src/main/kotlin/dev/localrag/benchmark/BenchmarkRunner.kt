package dev.localrag.benchmark

import dev.localrag.app.RagService
import dev.localrag.domain.ChunkStrategy
import dev.localrag.domain.IndexProgressUpdate
import dev.localrag.domain.IndexRepository
import dev.localrag.domain.ScoredChunk
import dev.localrag.domain.ModelSelection

class BenchmarkRunner(
    private val rag: RagService,
    private val store: BenchmarkStore,
    private val index: IndexRepository,
) {
    fun run(questions: List<BenchmarkQuestion>, selection: ModelSelection, progress: (IndexProgressUpdate) -> Unit) {
        require(questions.size == BenchmarkStore.EXPECTED_QUESTION_COUNT) {
            "Benchmark должен содержать ровно ${BenchmarkStore.EXPECTED_QUESTION_COUNT} вопросов."
        }
        val totalCases = questions.size * ChunkStrategy.entries.size
        var completed = 0
        questions.forEach { question ->
            progress(IndexProgressUpdate("baseline", completed, totalCases, 0, 0, question.question, 0, 0))
            val baseline = rag.baseline(question.question, selection)
            val sourcesByStrategy = rag.retrieveForStrategies(question.question)
            ChunkStrategy.entries.forEach { strategy ->
                progress(IndexProgressUpdate("rag", completed, totalCases, 0, 0, question.question, 0, 0))
                val sources = sourcesByStrategy.getValue(strategy)
                store.saveResult(
                    question = question,
                    strategy = strategy,
                    sources = sources,
                    baselineAnswer = baseline,
                    ragAnswer = rag.ragAnswer(question.question, sources, selection),
                    referenceHit = hasExpectedSourceHit(question, sources),
                    selection = selection,
                )
                completed++
                progress(IndexProgressUpdate("benchmark", completed, totalCases, 0, 0, null, 0, 0))
            }
        }
    }

    private fun hasExpectedSourceHit(question: BenchmarkQuestion, retrieved: List<ScoredChunk>): Boolean =
        question.expectedSources.any { expected ->
            index.hasSourceLocation(expected.sourceId, expected.location, expected.section) &&
                retrieved.any { candidate ->
                    val draft = candidate.chunk.draft
                    draft.sourceId == expected.sourceId && locationMatches(expected, draft.location, draft.section)
                }
        }

    private fun locationMatches(expected: BenchmarkExpectedSource, actual: dev.localrag.domain.SourceLocation, section: String): Boolean {
        val wanted = expected.location
        val rangeMatches = when {
            wanted.pageStart != null -> actual.pageStart != null &&
                actual.pageEnd!! >= wanted.pageStart && actual.pageStart <= wanted.pageEnd!!
            wanted.lineStart != null -> actual.lineStart != null &&
                actual.lineEnd!! >= wanted.lineStart && actual.lineStart <= wanted.lineEnd!!
            else -> true
        }
        val sectionMatches = expected.section.isNullOrBlank() || section.contains(expected.section, ignoreCase = true)
        return rangeMatches && sectionMatches
    }
}
