


def text_generation(prompt, model, tokenizer, repetition_penalty=1, device='cuda:0'):
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=2048,
        repetition_penalty=repetition_penalty,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        num_return_sequences=4)

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return result




