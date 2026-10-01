package dev.localrag.app

import kotlin.test.Test
import kotlin.test.assertEquals

class QueryNormalizerTest {
    @Test
    fun `normalizes compatibility forms case whitespace and typography without rewriting tokens`() {
        assertEquals(
            "what \"café\"- ... c++ --flag",
            QueryNormalizer.normalize("  ＷＨＡＴ\t“Café”—  …  C++   --FLAG  "),
        )
    }

    @Test
    fun `whitespace-only query normalizes to empty`() {
        assertEquals("", QueryNormalizer.normalize(" \t\n "))
    }
}
