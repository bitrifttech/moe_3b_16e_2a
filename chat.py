import os
import glob
import argparse
import torch
from transformers import AutoTokenizer

# Import both model types
from moe_model import GPT2WithMoE
from dense_model import GPT2Dense

def find_latest_checkpoint(output_dir='checkpoints'):
    """Find the latest checkpoint in the output directory."""
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, 'checkpoint-*')),
        key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else -1
    )
    return checkpoints[-1] if checkpoints else None

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a fine-tuned GPT-2 model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to the checkpoint directory. If not provided, uses the latest checkpoint in the checkpoints directory.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run the model on (default: cuda if available, else cpu)')
    parser.add_argument('--architecture', type=str, choices=['moe', 'dense'], default='moe',
                       help='Model architecture type (default: moe)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Find and validate checkpoint
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint('checkpoints')
        if args.checkpoint is None:
            print("No checkpoint found. Please specify a checkpoint path or train the model first.")
            return
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print(f"Loading {args.architecture.upper()} model from {args.checkpoint} ...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config
    config_path = os.path.join(args.checkpoint, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Try to load using safetensors first
    safetensors_path = os.path.join(args.checkpoint, 'model.safetensors')
    pytorch_path = os.path.join(args.checkpoint, 'pytorch_model.bin')
    
    # Initialize model with appropriate architecture
    print("Loading model configuration...")
    if args.architecture == 'dense':
        model = GPT2Dense.from_pretrained(
            args.checkpoint,
            config=config_path,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
    else:  # moe
        model = GPT2WithMoE.from_pretrained(
            args.checkpoint,
            config=config_path,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
    
    # Load weights
    print("Loading model weights...")
    if os.path.exists(safetensors_path):
        print("Loading from safetensors...")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device=args.device)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(pytorch_path):
        print("Loading from PyTorch checkpoint...")
        state_dict = torch.load(pytorch_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"No model weights found in {args.checkpoint}")
    
    model.to(args.device)
    model.eval()
    print(f"{args.architecture.upper()} model loaded successfully!")

    print("\nChatbot is ready! Type your message and press Enter. Type 'exit' to quit.\n")
    print(f"Running on device: {args.device}")
    print(f"Model architecture: {args.architecture.upper()}")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            prompt = input('You: ').strip()
            if not prompt:
                continue
                
            if prompt.lower() in {'exit', 'quit'}:
                print('Exiting chat.')
                break
                
            # Encode the input and move to the correct device
            inputs = tokenizer(prompt, return_tensors='pt', return_attention_mask=True)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                )
            
            # Decode and clean up the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()  # Remove input prompt
            
            # Print response with some formatting
            print('\nBot:', response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == '__main__':
    main()