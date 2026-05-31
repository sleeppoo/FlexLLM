from vllm import LLM, SamplingParams


def main():
    model_name = "shuyuej/Llama-3.3-70B-Instruct-GPTQ"

    print(f"Loading {model_name} ...")
    llm = LLM(
        model=model_name,
        quantization="gptq",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )

    conversation = []
    system_prompt = "You are a helpful assistant."
    conversation.append({"role": "system", "content": system_prompt})

    print("\n=== Interactive Chat (type 'quit' or 'exit' to stop) ===\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        outputs = llm.chat(
            messages=[conversation],
            sampling_params=sampling_params,
        )

        assistant_reply = outputs[0].outputs[0].text
        conversation.append({"role": "assistant", "content": assistant_reply})

        print(f"\nAssistant: {assistant_reply}\n")


if __name__ == "__main__":
    main()
