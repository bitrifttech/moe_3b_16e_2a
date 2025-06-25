#!/usr/bin/env python3
"""
Test 800M model memory usage with the recommended training settings.
"""

import torch
from dense_model import create_dense_config, GPT2Dense

def test_800m_memory():
    """Test 800M model memory usage with realistic training settings."""
    print("=== 800M Model Memory Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create 800M model with BF16 and gradient checkpointing
        print("Creating 800M model...")
        cfg = create_dense_config("800m", use_flash_attention=True)
        model = GPT2Dense(cfg).to(device)
        
        # Enable gradient checkpointing (important for memory)
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
        
        # Test with recommended training settings
        batch_size = 3
        seq_len = 1024  # Realistic sequence length for chatbot training
        
        print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
        
        # Create input batch
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        
        # Test forward pass (what happens during training)
        model.train()  # Training mode
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # BF16 autocast
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        
        # Simulate backward pass (gradient computation)
        loss.backward()
        
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # GB
        
        print(f"✓ Forward + backward pass successful")
        print(f"Current memory: {current_memory:.2f} GB")
        print(f"Peak memory: {peak_memory:.2f} GB")
        
        # Check if it fits comfortably (leave 15% buffer)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        memory_usage_percent = (peak_memory / total_memory) * 100
        
        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Memory usage: {memory_usage_percent:.1f}%")
        
        if memory_usage_percent < 85:
            print("✅ 800M model FITS with recommended settings!")
            return True
        else:
            print("⚠️ 800M model might be tight on memory")
            return False
            
    except torch.cuda.OutOfMemoryError:
        print("❌ 800M model OUT OF MEMORY")
        return False
    except Exception as e:
        print(f"❌ Error testing 800M model: {e}")
        return False

def compare_models():
    """Compare 600M vs 800M theoretical memory usage."""
    print("\n=== Model Comparison ===")
    
    # Model sizes (approximate parameters)
    sizes = {
        "600m": 500_000_000,
        "800m": 840_000_000,
    }
    
    for name, params in sizes.items():
        cfg = create_dense_config(name)
        
        # Estimate memory usage
        # Parameters: 2 bytes (BF16) 
        # Gradients: 2 bytes (BF16)
        # Optimizer states: 8 bytes (FP32 for Adam - momentum + variance)
        param_memory = params * 2 / 1024 / 1024 / 1024  # GB
        grad_memory = params * 2 / 1024 / 1024 / 1024   # GB  
        optimizer_memory = params * 8 / 1024 / 1024 / 1024  # GB
        
        total_model_memory = param_memory + grad_memory + optimizer_memory
        
        print(f"\n{name.upper()} Model:")
        print(f"  Parameters: {params/1_000_000:.1f}M")
        print(f"  Hidden size: {cfg.n_embd}")
        print(f"  Layers: {cfg.n_layer}")
        print(f"  Context: {cfg.n_positions}")
        print(f"  Model memory: {total_model_memory:.2f} GB")
        print(f"  (+ activations depending on batch size)")

def main():
    print("Testing 800M Model Memory Usage\n")
    
    compare_models()
    
    fits = test_800m_memory()
    
    print("\n" + "="*50)
    print("RECOMMENDATION:")
    
    if fits:
        print("✅ YES - Use 800M model with recommended settings!")
        print("\nOptimal command:")
        print("python train.py --architecture dense --model_size 800m \\")
        print("  --use_conversation --use_ultrachat --use_openorca --use_lmsys_chat --use_alpaca \\")
        print("  --bf16 --gradient_checkpointing \\")
        print("  --max_samples 80000 --epochs 6 \\")
        print("  --learning_rate 3e-4 --warmup_steps 1500 \\")
        print("  --batch_size 3 --gradient_accumulation_steps 12 \\")
        print("  --save_steps 400 --logging_steps 25 \\")
        print("  --output_dir dense_chatbot_800m_v1")
        
    else:
        print("⚠️  Stick with 600M model for safety")
        print("The 800M model may cause OOM errors with current settings.")
        print("Consider reducing batch_size to 2 if you want to try 800M.")
    
    return 0 if fits else 1

if __name__ == "__main__":
    exit(main())
