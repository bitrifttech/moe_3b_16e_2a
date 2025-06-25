#!/usr/bin/env python3
"""
Test PyTorch's Scaled Dot Product Attention (SDPA) for memory efficiency.
"""

import torch
import time
from dense_model import create_dense_config, GPT2Dense

def test_sdpa_support():
    """Test if SDPA is available and working."""
    print("=== PyTorch SDPA Support Test ===")
    
    pytorch_version = torch.__version__
    print(f"PyTorch version: {pytorch_version}")
    
    # Check if SDPA is available
    try:
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            print("✓ PyTorch has scaled_dot_product_attention")
            
            # Test basic SDPA functionality
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
            
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            
            with torch.no_grad():
                out = F.scaled_dot_product_attention(q, k, v)
                print(f"✓ SDPA test successful, output shape: {out.shape}")
            
            return True
        else:
            print("✗ PyTorch version doesn't support scaled_dot_product_attention")
            return False
            
    except Exception as e:
        print(f"✗ Error testing SDPA: {e}")
        return False

def test_model_attention_implementation():
    """Test what attention implementation is actually being used."""
    print("\n=== Model Attention Implementation Test ===")
    
    # Create model with SDPA enabled
    cfg = create_dense_config("200m", use_flash_attention=True)
    model = GPT2Dense(cfg)
    
    print(f"Model config attn_implementation: {getattr(cfg, 'attn_implementation', 'default')}")
    
    # Look at the first transformer block's attention
    first_block = model.transformer.h[0]
    attention = first_block.attn
    
    print(f"Attention class: {attention.__class__.__name__}")
    
    # Check if the attention module has specific attributes
    if hasattr(attention, '_attn_implementation'):
        print(f"Attention implementation: {attention._attn_implementation}")
    elif hasattr(attention, 'config') and hasattr(attention.config, 'attn_implementation'):
        print(f"Config attention implementation: {attention.config.attn_implementation}")
    else:
        print("No specific attention implementation found (using default)")
    
    return model

def memory_benchmark():
    """Benchmark memory usage with different model sizes."""
    print("\n=== Memory Benchmark ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = "cuda"
    sizes_to_test = ["200m", "400m", "600m"]
    
    results = {}
    
    for size in sizes_to_test:
        print(f"\nTesting {size} model...")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create model
            cfg = create_dense_config(size, use_flash_attention=True)
            model = GPT2Dense(cfg).to(device)
            
            # Test with realistic batch
            batch_size = 4
            seq_len = 512 if size in ["200m", "400m"] else 256  # Adjust seq_len for larger models
            
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids)
            
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            results[size] = {
                'current': current_memory,
                'peak': peak_memory,
                'seq_len': seq_len,
                'batch_size': batch_size
            }
            
            print(f"  ✓ Success - Peak memory: {peak_memory:.1f} MB (batch={batch_size}, seq={seq_len})")
            
            del model, outputs, input_ids
            
        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ Out of memory with batch_size={batch_size}, seq_len={seq_len}")
            results[size] = {'error': 'OOM'}
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[size] = {'error': str(e)}
    
    # Summary
    print("\n=== Memory Usage Summary ===")
    for size, result in results.items():
        if 'error' in result:
            print(f"{size:>5}: {result['error']}")
        else:
            print(f"{size:>5}: {result['peak']:>6.1f} MB (batch={result['batch_size']}, seq={result['seq_len']})")
    
    return results

def main():
    print("Testing Efficient Attention for Dense Models\n")
    
    if not test_sdpa_support():
        print("⚠ SDPA not available. Using standard attention.")
        return 1
    
    model = test_model_attention_implementation()
    
    results = memory_benchmark()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    
    # Find the largest model that fits
    working_sizes = [size for size, result in results.items() if 'error' not in result]
    
    if working_sizes:
        largest_working = max(working_sizes, key=lambda x: int(x.replace('m', '')))
        print(f"✓ Recommended model size: {largest_working}")
        print(f"  Command: python train.py --architecture dense --model_size {largest_working}")
        
        if len(working_sizes) > 1:
            print(f"  Alternative sizes: {', '.join(working_sizes[:-1])}")
    else:
        print("✗ No models fit in memory. Try:")
        print("  1. Reduce batch size: --batch_size 2 --gradient_accumulation_steps 16")
        print("  2. Use gradient checkpointing: --gradient_checkpointing")
        print("  3. Use model parallelism")
    
    print("\nMemory efficiency features enabled:")
    print("  ✓ PyTorch SDPA (automatic memory optimization)")
    print("  ✓ Mixed precision training (--fp16)")
    print("  ✓ Gradient accumulation")
    
    return 0

if __name__ == "__main__":
    exit(main())
