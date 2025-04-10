def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()
    import json
    from datasets import load_dataset
    from tqdm import tqdm

    # Load your original dataset
    original_dataset = load_dataset("nateraw/convert-units")["train"]

    model = CoTModel()

    new_dataset = []

    for item in tqdm(original_dataset, desc="Generating CoT completions"):
        question, correct_answer = item["question"], item["answer"]

        # Generate multiple completions
        completions = model.batched_generate(
            prompts=[question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]  # Get the list for the single prompt

        for response in completions:
            parsed = model.parse_answer(response)
            if is_answer_valid(parsed, correct_answer):
                new_dataset.append({
                    "question": question,
                    "answer": correct_answer,
                    "completion": response.strip()
                })
                break  # Use only the first correct one

    # Save new dataset to JSON
    with open(output_json, "w") as f:
        json.dump(new_dataset, f, indent=2)

    print(f"Saved {len(new_dataset)} filtered CoT completions to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
