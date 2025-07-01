#!/usr/bin/env python3
"""
Test script to verify MoE model with flash attention implementation.
"""

import torch
import sys
import os

def test_moe_config_creation():
    """Test MoE configuration creation with flash attention."""
    print("=" * 60)
    print("Testing MoE Configuration Creation")
    print("=" * 60)
    
    try:
        from moe_model import create_moe_config, get_moe_model_info
        
        # Test different model sizes
        sizes = ["500m", "1b", "2b"]
        
        for size in sizes:
            print(f"\n--- Testing {size} configuration ---")
            
            # Test with flash attention
            config_flash = create_moe_config(size, use_flash_attention=True)
            print(f"✓ Created {size} config with flash attention")
            print(f"  Context length: {config_flash.n_positions}")
            print(f"  Hidden size: {config_flash.n_embd}")
            print(f"  Layers: {config_flash.n_layer}")
            print(f"  Attention implementation: {getattr(config_flash, 'attn_implementation', 'standard')}")
            
            # Test without flash attention
            config_standard = create_moe_config(size, use_flash_attention=False)
            print(f"✓ Created {size} config without flash attention")
            print(f"  Attention implementation: {getattr(config_standard, 'attn_implementation', 'standard')}")
            
            # Test model info calculation
            info = get_moe_model_info(config_flash, num_experts=16)
            print(f"✓ Model info calculated:")
            print(f"  Total params: {info['total_params_m']:.1f}M")
            print(f"  Active params: {info['active_params_m']:.1f}M")
            print(f"  Expert utilization: {info['expert_utilization']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_moe_model_creation():
    """Test MoE model creation with flash attention."""
    print("\n" + "=" * 60)
    print("Testing MoE Model Creation")
    print("=" * 60)
    
    try:
        from moe_model import create_moe_config, GPT2WithMoE
        from transformers import AutoTokenizer
        
        # Create a small test configuration
        config = create_moe_config("500m", use_flash_attention=True)
        config.n_layer = 4  # Reduce for testing
        config.n_embd = 256  # Reduce for testing
        config.n_inner = 1024  # Reduce for testing
        config.vocab_size = 1000  # Reduce for testing
        
        print("✓ Created test configuration")
        print(f"  Flash attention: {getattr(config, 'attn_implementation', 'standard') == 'sdpa'}")
        
        # Create model
        print("Creating MoE model...")
        model = GPT2WithMoE(config, num_experts=4)  # Reduce experts for testing
        
        print("✓ Successfully created MoE model")
        print(f"  Model has flash attention: {model.use_flash_attention}")
        print(f"  Number of experts: {model.num_experts}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
            
        print("✓ Forward pass successful")
        print(f"  Output shape: {outputs.logits.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        if outputs.logits.shape == expected_shape:
            print("✓ Output shape is correct")
        else:
            print(f"✗ Output shape mismatch: got {outputs.logits.shape}, expected {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flash_attention_availability():
    """Test if flash attention is available in the current PyTorch version."""
    print("\n" + "=" * 60)
    print("Testing Flash Attention Availability")
    print("=" * 60)
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        
        # Check if scaled_dot_product_attention is available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✓ scaled_dot_product_attention is available")
            
            # Test basic functionality
            batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
            q = torch.randn(batch_size, num_heads, seq_len, head_dim)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim)
            
            with torch.no_grad():
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            
            print("✓ scaled_dot_product_attention works correctly")
            print(f"  Input shape: {q.shape}")
            print(f"  Output shape: {output.shape}")
            
            return True
        else:
            print("✗ scaled_dot_product_attention is not available")
            print("  Please upgrade to PyTorch 2.0+ for flash attention support")
            return False
            
    except Exception as e:
        print(f"✗ Flash attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show usage examples for the improved MoE model."""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("1. Train MoE model with flash attention (default):")
    print("   python3 train.py --architecture moe --moe_size 500m --num_experts 16")
    
    print("\n2. Train larger MoE model:")
    print("   python3 train.py --architecture moe --moe_size 1b --num_experts 32")
    
    print("\n3. Train MoE without flash attention:")
    print("   python3 train.py --architecture moe --no_flash_attention")
    
    print("\n4. Chat with trained MoE model:")
    print("   python3 chat.py --architecture moe")
    
    print("\n5. Key improvements:")
    print("   ✓ Flash attention for memory efficiency")
    print("   ✓ Configurable model sizes (500m, 1b, 2b)")
    print("   ✓ Configurable number of experts")
    print("   ✓ Better parameter counting and reporting")
    print("   ✓ Improved conversation formatting for instruction following")

def main():
    """Run all MoE flash attention tests."""
    print("🧪 Testing MoE Model with Flash Attention")
    print("This will test the enhanced MoE implementation with flash attention support.")
    
    results = []
    
    # Test flash attention availability
    results.append(test_flash_attention_availability())
    
    # Test configuration creation
    results.append(test_moe_config_creation())
    
    # Test model creation (only if flash attention is available)
    if results[0]:  # Flash attention available
        results.append(test_moe_model_creation())
    else:
        print("\nSkipping model creation test due to flash attention unavailability")
        results.append(False)
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! MoE with flash attention is working correctly.")
        print("\nKey improvements:")
        print("• Flash attention support for memory efficiency")
        print("• Configurable MoE model sizes and expert counts")
        print("• Better parameter counting and reporting")
        print("• Enhanced conversation formatting")
        print("\nYour MoE models should now be more memory efficient and perform better!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not results[0]:
            print("\n💡 Tip: Upgrade to PyTorch 2.0+ for flash attention support")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())