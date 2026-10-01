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
            val validExpectedSources = question.expectedSources.filter { expected ->
                index.hasSourceLocation(expected.sourceId, expected.location, expected.section)
            }
            val comparisonsByStrategy = rag.retrieveForStrategies(question.question, selection)
            ChunkStrategy.entries.forEach { strategy ->
                progress(IndexProgressUpdate("rag", completed, totalCases, 0, 0, question.question, 0, 0))
                val comparison = comparisonsByStrategy.getValue(strategy)
                store.saveResult(
                    question = question,
                    strategy = strategy,
                    rawSources = comparison.raw,
                    sources = comparison.enhanced,
                    baselineAnswer = baseline,
                    ragAnswer = rag.ragAnswer(question.question, comparison.enhanced, selection),
                    rawExpectedSourceRank = expectedSourceRank(validExpectedSources, comparison.raw),
                    expectedSourceRank = expectedSourceRank(validExpectedSources, comparison.enhanced),
                    selection = selection,
                )
                completed++
                progress(IndexProgressUpdate("benchmark", completed, totalCases, 0, 0, null, 0, 0))
            }
        }
    }

    private fun expectedSourceRank(expectedSources: List<BenchmarkExpectedSource>, retrieved: List<ScoredChunk>): Int? {
        val position = retrieved.indexOfFirst { candidate ->
            val draft = candidate.chunk.draft
            expectedSources.any { expected ->
                draft.sourceId == expected.sourceId && locationMatches(expected, draft.location, draft.section)
            }
        }
        return if (position < 0) null else position + 1
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
