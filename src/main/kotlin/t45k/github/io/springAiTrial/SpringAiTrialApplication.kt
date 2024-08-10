package t45k.github.io.springAiTrial

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
class Controller(private val embeddingModel: EmbeddingModel) {
    @GetMapping("/ai")
    fun call() {
        val response = embeddingModel.embedForResponse(listOf("hello"))
        println(response.results.map { it.output })
    }
}
