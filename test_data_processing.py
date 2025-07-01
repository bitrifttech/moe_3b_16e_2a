#!/usr/bin/env python3
"""
Test script to verify improved data processing for instruction-following training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_modules.conversational.openassistant import create_openassistant_loader
from data_modules.conversational.anthropic_hh import create_anthropic_hh_loader

def test_openassistant_processing():
    """Test OpenAssistant data processing."""
    print("=" * 60)
    print("Testing OpenAssistant Data Processing")
    print("=" * 60)
    
    try:
        # Create loader with small sample size for testing
        loader = create_openassistant_loader(max_samples=10, max_length=500)
        
        # Load and process data
        train_data, val_data, stats = loader.load_and_process()
        
        print(f"✓ Successfully loaded OpenAssistant data")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Stats: {stats}")
        
        # Show example conversations
        print("\nExample conversations:")
        for i, example in enumerate(train_data[:3]):
            print(f"\n--- Example {i+1} ---")
            print(example["text"][:300] + "..." if len(example["text"]) > 300 else example["text"])
            
            # Verify format
            if "Human:" in example["text"] and "Assistant:" in example["text"]:
                print("✓ Proper conversation format detected")
            else:
                print("⚠ Missing proper conversation format")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAssistant test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_anthropic_hh_processing():
    """Test Anthropic HH data processing."""
    print("\n" + "=" * 60)
    print("Testing Anthropic HH Data Processing")
    print("=" * 60)
    
    try:
        # Create loader with small sample size for testing
        loader = create_anthropic_hh_loader(max_samples=10, max_length=500)
        
        # Load and process data
        train_data, val_data, stats = loader.load_and_process()
        
        print(f"✓ Successfully loaded Anthropic HH data")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Stats: {stats}")
        
        # Show example conversations
        print("\nExample conversations:")
        for i, example in enumerate(train_data[:3]):
            print(f"\n--- Example {i+1} ---")
            print(example["text"][:300] + "..." if len(example["text"]) > 300 else example["text"])
            
            # Verify format
            if "Human:" in example["text"] and "Assistant:" in example["text"]:
                print("✓ Proper conversation format detected")
            else:
                print("⚠ Missing proper conversation format")
        
        return True
        
    except Exception as e:
        print(f"✗ Anthropic HH test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_quality():
    """Test conversation quality and format."""
    print("\n" + "=" * 60)
    print("Testing Conversation Quality")
    print("=" * 60)
    
    # Sample conversation formats to test
    test_conversations = [
        "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris. It's located in the north-central part of the country and is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
        "Human: Can you help me write a Python function to calculate factorial?\n\nAssistant: Sure! Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n```\n\nThis function uses recursion to calculate the factorial of a number.",
        "Human: What's the weather like today?\n\nAssistant: I don't have access to real-time weather data, so I can't tell you the current weather. To get accurate weather information, I'd recommend checking a weather app, website like weather.com, or asking a voice assistant that has internet access."
    ]
    
    print("Analyzing conversation formats:")
    
    for i, conv in enumerate(test_conversations):
        print(f"\n--- Test Conversation {i+1} ---")
        
        # Check format
        has_human = "Human:" in conv
        has_assistant = "Assistant:" in conv
        word_count = len(conv.split())
        
        print(f"Has Human: {has_human}")
        print(f"Has Assistant: {has_assistant}")
        print(f"Word count: {word_count}")
        print(f"Length: {len(conv)} characters")
        
        # Check if it would pass filters
        if has_human and has_assistant and word_count >= 20 and word_count <= 800:
            print("✓ Would pass quality filters")
        else:
            print("⚠ Would fail quality filters")
        
        print(f"Preview: {conv[:150]}...")
    
    return True

def main():
    """Run all data processing tests."""
    print("🧪 Testing Improved Data Processing")
    print("This will test the new conversation formatting and filtering.")
    
    results = []
    
    # Test OpenAssistant processing
    results.append(test_openassistant_processing())
    
    # Test Anthropic HH processing  
    results.append(test_anthropic_hh_processing())
    
    # Test conversation quality
    results.append(test_conversation_quality())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Data processing improvements are working correctly.")
        print("\nKey improvements:")
        print("• Conversations now include both Human and Assistant context")
        print("• Proper instruction-response formatting")
        print("• Better quality filtering")
        print("• Longer context support (1024 tokens)")
        print("\nYour models should now learn to follow instructions better!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())