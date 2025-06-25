#!/usr/bin/env python3
"""
Test script to check flash attention availability and performance.
"""

import torch
import time
from dense_model import create_dense_config, GPT2Dense

def test_flash_attention_availability():
    """Test if flash attention is available."""
    print("=== Flash Attention Availability Test ===")
    
    try:
        # Try importing flash attention
        import flash_attn
        print(f"✓ flash-attn package found: version {flash_attn.__version__}")
        flash_available = True
    except ImportError:
        print("✗ flash-attn package not found")
        flash_available = False
    
    # Test Transformers built-in support
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
        print("✓ Transformers GPT2 available")
        
        # Check for flash attention support in newer transformers
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        if hasattr(GPT2Attention, "_flash_attn_2_enabled"):
            print("✓ Transformers has FlashAttention-2 support")
        else:
            print("⚠ Transformers version may not support FlashAttention-2")
            
    except Exception as e:
        print(f"✗ Error checking Transformers: {e}")
    
    return flash_available

def test_model_creation():
    """Test creating models with and without flash attention."""
    print("\n=== Model Creation Test ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test small model with flash attention
    try:
        print("Testing 200M model with flash attention...")
        cfg_flash = create_dense_config("200m", use_flash_attention=True)
        model_flash = GPT2Dense(cfg_flash)
        
        if hasattr(cfg_flash, 'attn_implementation'):
            print(f"✓ Config has attn_implementation: {cfg_flash.attn_implementation}")
        else:
            print("⚠ Config doesn't have attn_implementation (may fall back to default)")
            
        print(f"✓ Model with flash attention created successfully")
        
        # Test moving to device
        model_flash = model_flash.to(device)
        print(f"✓ Model moved to {device}")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 128)).to(device)
        
        with torch.no_grad():
            outputs = model_flash(input_ids)
            print(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        del model_flash, outputs, input_ids
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Flash attention model failed: {e}")
        return False
    
    # Test model without flash attention
    try:
        print("\nTesting 200M model without flash attention...")
        cfg_standard = create_dense_config("200m", use_flash_attention=False)
        model_standard = GPT2Dense(cfg_standard)
        
        if hasattr(cfg_standard, 'attn_implementation'):
            print(f"Config has attn_implementation: {cfg_standard.attn_implementation}")
        else:
            print("✓ Config uses standard attention (no attn_implementation)")
            
        print(f"✓ Model without flash attention created successfully")
        
        del model_standard
        
    except Exception as e:
        print(f"✗ Standard attention model failed: {e}")
        return False
    
    return True

def memory_comparison():
    """Compare memory usage between flash and standard attention."""
    print("\n=== Memory Usage Comparison ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = "cuda"
    
    def get_memory_mb():
        return torch.cuda.memory_allocated() / 1024 / 1024
    
    # Test flash attention memory
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cfg_flash = create_dense_config("400m", use_flash_attention=True)
        model_flash = GPT2Dense(cfg_flash).to(device)
        
        # Simulate training batch
        batch_size, seq_len = 4, 512
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            outputs = model_flash(input_ids)
        
        flash_memory = get_memory_mb()
        flash_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"Flash Attention - Current: {flash_memory:.1f} MB, Peak: {flash_peak:.1f} MB")
        
        del model_flash, outputs, input_ids
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Flash attention memory test failed: {e}")
        return
    
    # Test standard attention memory
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cfg_standard = create_dense_config("400m", use_flash_attention=False)
        model_standard = GPT2Dense(cfg_standard).to(device)
        
        # Simulate training batch
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        
        with torch.no_grad():
            outputs = model_standard(input_ids)
        
        standard_memory = get_memory_mb()
        standard_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"Standard Attention - Current: {standard_memory:.1f} MB, Peak: {standard_peak:.1f} MB")
        
        if flash_peak > 0:
            savings = ((standard_peak - flash_peak) / standard_peak) * 100
            print(f"Memory savings with Flash Attention: {savings:.1f}%")
        
        del model_standard, outputs, input_ids
        
    except Exception as e:
        print(f"Standard attention memory test failed: {e}")

def main():
    print("Testing Flash Attention Support for Dense Models\n")
    
    flash_available = test_flash_attention_availability()
    
    if test_model_creation():
        print("\n✓ All model creation tests passed!")
        
        if flash_available:
            memory_comparison()
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        
        if flash_available:
            print("✓ Flash attention is available and working!")
            print("  Use: python train.py --architecture dense --model_size 400m")
            print("  (flash attention is enabled by default)")
        else:
            print("⚠ Flash attention package not found.")
            print("  Options:")
            print("  1. Install: pip install flash-attn")
            print("  2. Use without: python train.py --no_flash_attention")
            print("  3. Use smaller model: --model_size 200m")
        
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
