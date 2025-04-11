from .data import Dataset, is_answer_valid
import json
from tqdm import tqdm
from .cot import CoTModel

# def convert_to_prompt_completion_format(dataset: Dataset) -> list[dict]:
#     data = []
#     for question, answer in dataset:
#         data.append({
#             "question": question, #prompt
#             "answer": f"<answer>{answer}</answer>", #completion
#         })
#     return data


# def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
#     #raise NotImplementedError()

#     train_dataset = Dataset("train")
#     formatted_data = convert_to_prompt_completion_format(train_dataset)

#     model = CoTModel()

#     new_dataset = []

#     for item in tqdm(formatted_data, desc="Generating CoT completions"):
#         question, correct_answer = item["question"], item["answer"]

#         # Generate multiple completions
#         completions = model.batched_generate(
#             prompts=[question],
#             num_return_sequences=oversample,
#             temperature=temperature
#         )[0]  # Get the list for the single prompt

#         for response in completions:
#             parsed = model.parse_answer(response)
#             if is_answer_valid(parsed, correct_answer):
#                 new_dataset.append({
#                     "question": question,
#                     "answer": correct_answer,
#                     "completion": response.strip()
#                 })
#                 break  # Use only the first correct one

#     # Save new dataset to JSON
#     with open(output_json, "w") as f:
#         json.dump(new_dataset, f, indent=2)

#     print(f"Saved {len(new_dataset)} filtered CoT completions to {output_json}")


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):

    # Load original dataset
    dataset = Dataset("train")

    model = CoTModel()
    new_dataset = []

    for item in tqdm(dataset, desc="Generating CoT completions"):
        print(item)
        question = item["question"]
        correct_answer = item["answer"]

        # Generate multiple completions per question
        completions = model.batched_generate(
            prompts=[question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]  # Get list of completions for single prompt

        # Filter for valid completions
        for completion in completions:
            parsed_answer = model.parse_answer(completion)
            if is_answer_valid(parsed_answer, correct_answer):
                new_dataset.append({
                    "question": question,
                    "answer": correct_answer,
                    "completion": completion.strip()
                })
                break  # Keep only first valid completion

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(new_dataset, f, indent=2)

    print(f"Saved {len(new_dataset)} CoT examples to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
