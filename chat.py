import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_latest_checkpoint(output_dir='outputs'):
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, 'checkpoint-*')),
        key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else -1
    )
    return checkpoints[-1] if checkpoints else None

def main():
    output_dir = 'outputs'
    checkpoint_dir = find_latest_checkpoint(output_dir)
    if not checkpoint_dir:
        print(f"No checkpoint found in '{output_dir}'. Please train the model first.")
        return

    print(f"Loading model from {checkpoint_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.eval()

    print("\nChatbot is ready! Type your message and press Enter. Type 'exit' to quit.\n")
    while True:
        prompt = input('You: ')
        if prompt.strip().lower() in {'exit', 'quit'}:
            print('Exiting chat.')
            break
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Only print the new part of the response
        print('Bot:', response[len(prompt):].strip())

if __name__ == '__main__':
    main() 