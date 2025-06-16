import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from pathlib import Path
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_path = Path("checkpoints/checkpoint-100")
    print(f"Loading model from {checkpoint_path}...")
    
    try:
        print("Loading tokenizer...")
        # First try to load the tokenizer from the checkpoint
        tokenizer_path = checkpoint_path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = GPT2Tokenizer.from_pretrained(
                str(checkpoint_path),
                local_files_only=True
            )
        else:
            print("Tokenizer files not found in checkpoint, using default GPT-2 tokenizer")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model...")
        # Try loading with trust_remote_code first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                trust_remote_code=True,
                torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32,
                device_map='auto'
            )
        except Exception as e:
            print(f"Failed to load with trust_remote_code: {e}")
            print("Trying to load with explicit model class...")
            from moe_model import GPT2WithMoE
            model = GPT2WithMoE.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32,
                device_map='auto',
                local_files_only=True
            )
        
        # Test inference
        print("\nModel loaded successfully! Testing inference...")
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"\nError during model loading or inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
