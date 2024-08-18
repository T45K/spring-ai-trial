package t45k.github.io.springAiTrial

import kotlin.math.sqrt
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
    val a = vector1.zip(vector2) { a, b -> a * b }.sum()
    val b = sqrt(vector1.sumOf { it * it }) * sqrt(vector2.sumOf { it * it })
    return a / b
}

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
