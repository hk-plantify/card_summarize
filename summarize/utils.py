import torch

def generate_summary(model, tokenizer, text, device, max_len):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_len, truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=input_ids, max_length=150, num_beams=5, early_stopping=True
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
