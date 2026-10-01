package dev.localrag.app

import java.text.Normalizer
import java.util.Locale

internal object QueryNormalizer {
    fun normalize(query: String): String {
        val folded = Normalizer.normalize(query, Normalizer.Form.NFKC).lowercase(Locale.ROOT)
        val normalized = StringBuilder(folded.length)
        var pendingSpace = false

        for (character in folded) {
            if (character.isWhitespace()) {
                pendingSpace = normalized.isNotEmpty()
                continue
            }
            if (pendingSpace) {
                normalized.append(' ')
                pendingSpace = false
            }
            when (character) {
                '\u2018', '\u2019' -> normalized.append('\'')
                '\u201c', '\u201d' -> normalized.append('"')
                '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015' -> normalized.append('-')
                '\u2026' -> normalized.append("...")
                else -> normalized.append(character)
            }
        }
        return normalized.toString()
    }
}
