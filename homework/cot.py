from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a helpful assistant that performs unit conversions. "
        #                    "Be concise. Use step-by-step reasoning. Format your final answer inside <answer> tags."
        #     },
        #     {
        #         "role": "user",
        #         "content": "How many feet are in 3 meters?"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "To convert meters to feet, we use the conversion: 1 meter = 3.28084 feet.\n"
        #                    "So, 3 meters * 3.28084 = 9.84252 feet.\n"
        #                    "<answer>9.84252</answer>"
        #     },
        #     {
        #         "role": "user",
        #         "content": question
        #     },
        # ]

        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a helpful assistant that performs unit conversions. "
        #                    "Be concise. Use step-by-step reasoning. Format your final answer inside <answer> tags."
        #     },

        #     # EXAMPLE 1: meters to feet
        #     {
        #         "role": "user",
        #         "content": "How many feet are in 3 meters?"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "To convert meters to feet, we use the conversion: 1 meter = 3.28084 feet.\n"
        #                    "So, 3 meters * 3.28084 = 9.84252 feet.\n"
        #                    "<answer>9.84252</answer>"
        #     },

        #     # EXAMPLE 2: meters to yards
        #     {
        #         "role": "user",
        #         "content": "How many yards are in 5 meters?"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "1 meter = 1.09361 yards.\n"
        #                    "So, 5 meters * 1.09361 = 5.46805 yards.\n"
        #                    "<answer>5.46805</answer>"
        #     },

        #     # EXAMPLE 3: yards to feet
        #     {
        #         "role": "user",
        #         "content": "How many feet are in 2 yards?"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "1 yard = 3 feet.\n"
        #                    "So, 2 yards * 3 = 6 feet.\n"
        #                    "<answer>6</answer>"
        #     },

        #     # EXAMPLE 4: feet to meters
        #     {
        #         "role": "user",
        #         "content": "How many meters are in 10 feet?"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "1 foot = 0.3048 meters.\n"
        #                    "So, 10 feet * 0.3048 = 3.048 meters.\n"
        #                    "<answer>3.048</answer>"
        #     },

        #     # The actual test question
        #     {
        #         "role": "user",
        #         "content": question
        #     },
        # ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that performs unit conversions. "
                           "Be concise. Use step-by-step reasoning. Format your final answer inside <answer> tags."
            },

            # MY EXAMPLE 1: GB to MB
            {
                "role": "user",
                "content": "How many MB are there per 9 GB?"
            },
            {
                "role": "assistant",
                "content": "To convert GB to MB, we use the conversion: 1 GB = 1000 MB.\n"
                           "So, 9 GB * 1000 = 9000.0 MB.\n"
                           "<answer>9000.0</answer>"
            },

            # MY EXAMPLE 2: pound to ounce
            {
                "role": "user",
                "content": "How do we express 4 pound in terms of ounce?"
            },
            {
                "role": "assistant",
                "content": "1 pound = 16 ounces.\n"
                           "So, 4 pounds * 16 = 64.0 ounces.\n"
                           "<answer>64.0</answer>"
            },

            # MY EXAMPLE 3: kmh to mih
            {
                "role": "user",
                "content": "Can you change 9 kmh to its equivalent in mi/h"
            },
            {
                "role": "assistant",
                "content": "To convert kmh to mi/h, we use the conversion: 1 km = 0.6213711922373.\n"
                           "So, 9 kmh * 0.6213711922373 = 5.592340730136007 mi/h.\n"
                           "<answer>5.592340730136007</answer>"
            },

            # MY EXAMPLE 4: feet to meters
            {
                "role": "user",
                "content": "What is the equivalent of 10 century in year?"
            },
            {
                "role": "assistant",
                "content": "1 century = 100 years.\n"
                           "So, 10 centuries * 100 = 1000.0 years.\n"
                           "<answer>1000.0</answer>"
            },

            # The actual test question
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
