package t45k.github.io.springAiTrial

import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.embedding.EmbeddingModel
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@SpringBootApplication
class SpringAiTrialApplication

fun main(args: Array<String>) {
    runApplication<SpringAiTrialApplication>(*args)
}

@RestController
class Controller(
    private val embeddingModel: EmbeddingModel,
    private val chatModel: ChatModel,
) {
    @GetMapping("/embedding")
    fun call(): Any {
        val response = embeddingModel.embedForResponse(
            listOf(
                sumMethod1.trimIndent(),
                sumMethod2.trimIndent(),
                sumMethod3.trimIndent(),
                productMethod.trimIndent(),
            )
        )

        val vectors = response.results.map { it.output }
        return vectors.map { vector -> vectors.map { calcCosSimilarity(vector, it) } }
    }

    @GetMapping("/chat")
    fun callChat(): Any {
        return chatModel.call("Please list up 10 famous Japanese food")
    }
}

private fun calcCosSimilarity(vector1: List<Double>, vector2: List<Double>): Double {
    val ndarray1 = mk.ndarray(vector1)
    val ndarray2 = mk.ndarray(vector2)

    return (ndarray1 dot ndarray2) / (mk.linalg.norm(ndarray1) * mk.linalg.norm(ndarray2))
}

private fun LinAlg.norm(mat: MultiArray<Double, D1>): Double = norm(mk.stack(mat, mk.zeros(mat.size)))

private const val sumMethod1 = """
    fun sum(values: List<Int>): Int {
        var sum = 0
        for (value in values) {
            sum += value
        }
        return sum
    }
"""

private const val sumMethod2 = """
    fun sum(values: List<Int>): Int {
        return values.reduce { acc, value -> acc + value }
    }
"""

private const val sumMethod3 = """
    fun sum(values: List<Int>): Int = values.sum()
"""

private const val productMethod = """
    fun product(values: List<Int>): Int {
        var product = 1
        for (value in values) {
            product *= value
        }
        return product
    }
"""
