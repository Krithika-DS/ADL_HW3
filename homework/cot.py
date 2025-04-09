from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that performs unit conversions. "
                           "Be concise. Use step-by-step reasoning. Format your final answer inside <answer> tags."
            },
            {
                "role": "user",
                "content": "How many feet are in 3 meters?"
            },
            {
                "role": "assistant",
                "content": "To convert meters to feet, we use the conversion: 1 meter = 3.28084 feet.\n"
                           "So, 3 meters * 3.28084 = 9.84252 feet.\n"
                           "<answer>9.84252</answer>"
            },
            {
                "role": "user",
                "content": question
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
