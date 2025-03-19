package t45k.github.io.springAiTrial

import org.springframework.ai.chat.model.ChatModel
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@SpringBootApplication
class SpringAiTrialApplication

fun main(args: Array<String>) {
    runApplication<SpringAiTrialApplication>(*args)
}

@RestController
class Controller(private val chatModel: ChatModel) {
    @GetMapping("/chat")
    fun chat(@RequestParam message: String): Map<String, String> {
        val answer = chatModel.call(message)
        return mapOf("answer" to answer)
    }
}
