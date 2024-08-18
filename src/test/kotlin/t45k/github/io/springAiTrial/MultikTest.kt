package t45k.github.io.springAiTrial

import kotlin.math.sqrt
import kotlin.test.assertEquals
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.junit.jupiter.api.Test

class MultikTest {
    @Test
    fun cosSimilarity() {
        val vector1 = listOf(1.0, 2.2, 0.0, 3.1, 1.0)
        val vector2 = listOf(2.0, 0.0, 1.1, 3.0, 0.0)

        val cosSimilarity1 = calcCosSimilarity(vector1, vector2)

        val ndarray1 = mk.ndarray(vector1)
        val ndarray2 = mk.ndarray(vector2)

        val cosSimilarity2 = (ndarray1 dot ndarray2) / (mk.linalg.norm(ndarray1) * mk.linalg.norm(ndarray2))

        assertEquals(cosSimilarity1, cosSimilarity2)
    }

    private fun LinAlg.norm(mat: MultiArray<Double, D1>): Double = norm(mk.stack(mat, mk.zeros(mat.size)))

    private fun calcCosSimilarity(vector1: List<Double>, vector2: List<Double>): Double {
        val a = vector1.zip(vector2) { a, b -> a * b }.sum()
        val b = sqrt(vector1.sumOf { it * it }) * sqrt(vector2.sumOf { it * it })
        return a / b
    }
}