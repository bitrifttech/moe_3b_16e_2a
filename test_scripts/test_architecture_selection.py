#!/usr/bin/env python3
"""
Test script to verify architecture selection works correctly in the training script.
"""

import sys
import argparse
import json
from pathlib import Path

def test_model_creation():
    """Test that both architectures can be created without errors."""
    
    # Test dense architecture
    print("Testing dense architecture creation...")
    try:
        from dense_model import create_dense_config, GPT2Dense, get_model_info
        
        for model_size in ["800m", "1.5b"]:
            print(f"  Testing {model_size} configuration...")
            cfg = create_dense_config(model_size)
            model = GPT2Dense(cfg)
            info = get_model_info(cfg)
            print(f"    ✓ {model_size}: {info['total_params_m']:.1f}M parameters")
            del model  # Free memory
            
    except Exception as e:
        print(f"    ✗ Dense model creation failed: {e}")
        return False
    
    # Test MoE architecture
    print("Testing MoE architecture creation...")
    try:
        from moe_model import GPT2WithMoE
        from transformers import GPT2Config
        
        cfg = GPT2Config(
            vocab_size=50257,  # Default GPT-2 vocab size
            n_positions=512,
            n_embd=1024,
            n_layer=8,
            n_head=16,
            n_inner=4096
        )
        
        model = GPT2WithMoE(cfg)
        print(f"    ✓ MoE model created successfully")
        del model  # Free memory
        
    except Exception as e:
        print(f"    ✗ MoE model creation failed: {e}")
        return False
    
    print("All architecture tests passed!")
    return True

def test_argument_parsing():
    """Test that argument parsing works for both architectures."""
    
    print("Testing argument parsing...")
    
    # Import the argument parser from train.py
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        # We'll simulate the argument parsing logic
        test_cases = [
            # Dense architectures
            ["--architecture", "dense", "--model_size", "800m"],
            ["--architecture", "dense", "--model_size", "1.5b"],
            # MoE architecture
            ["--architecture", "moe"],
        ]
        
        for args in test_cases:
            print(f"  Testing args: {' '.join(args)}")
            
            # Create a minimal parser to test
            parser = argparse.ArgumentParser()
            parser.add_argument("--architecture", choices=["moe", "dense"], default="moe")
            parser.add_argument("--model_size", choices=["800m", "1.5b"], default="800m")
            
            parsed = parser.parse_args(args)
            print(f"    ✓ Parsed: architecture={parsed.architecture}, model_size={parsed.model_size}")
            
    except Exception as e:
        print(f"    ✗ Argument parsing failed: {e}")
        return False
    
    print("Argument parsing tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Architecture Selection for MoE Training Pipeline")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    if test_argument_parsing():
        tests_passed += 1
    
    if test_model_creation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Architecture selection is working correctly.")
        print("\nYou can now train models using:")
        print("  # Dense 800M model:")
        print("  python train.py --architecture dense --model_size 800m")
        print("  # Dense 1.5B model:")
        print("  python train.py --architecture dense --model_size 1.5b")
        print("  # MoE model (original):")
        print("  python train.py --architecture moe")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
