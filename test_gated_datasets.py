#!/usr/bin/env python3
"""Test the gated datasets now that we're authenticated."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import (
    create_lmsys_chat_loader,
    create_chatbot_arena_loader
)

def test_gated_dataset(name, loader_func, max_samples=50):
    """Test a gated dataset with small sample size."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING GATED DATASET: {name}")
    print(f"{'='*60}")
    
    try:
        # Create loader with small sample
        loader = loader_func(max_samples=max_samples)
        print(f"✅ Loader created successfully")
        
        # Load raw data
        print("🔄 Loading raw data...")
        raw_data = loader.load_raw_data()
        print(f"✅ Raw data: {len(raw_data)} samples")
        
        if len(raw_data) > 0:
            print(f"📝 Raw sample keys: {list(raw_data[0].keys())}")
            print(f"📝 Raw sample preview: {str(raw_data[0])[:300]}...")
            
        # Preprocess
        print("🔄 Preprocessing...")
        processed = loader.preprocess(raw_data)
        print(f"✅ Processed: {len(processed)} samples")
        
        if len(processed) > 0:
            print(f"📝 Processed keys: {list(processed[0].keys())}")
        
        # Process examples (test first 10)
        print("🔄 Processing examples...")
        successes = 0
        failures = 0
        
        for i, example in enumerate(processed[:10]):
            result = loader.process_example(example)
            if result:
                successes += 1
                if successes == 1:  # Show first success
                    text = result['text']
                    print(f"📖 Sample text ({len(text)} chars): {text[:200]}...")
            else:
                failures += 1
        
        total_tested = min(10, len(processed))
        success_rate = (successes / total_tested * 100) if total_tested > 0 else 0
        print(f"✅ Success rate: {successes}/{total_tested} ({success_rate:.1f}%)")
        
        if success_rate > 80:
            print("🎉 DATASET WORKING PERFECTLY!")
        elif success_rate > 50:  
            print("⚠️  DATASET HAS SOME ISSUES")
        else:
            print("❌ DATASET STILL BROKEN")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔓 TESTING GATED DATASETS (POST-AUTHENTICATION)")
    
    gated_datasets = [
        ("LMSYS Chat", create_lmsys_chat_loader),
        ("Chatbot Arena", create_chatbot_arena_loader),
    ]
    
    for name, loader_func in gated_datasets:
        test_gated_dataset(name, loader_func, max_samples=100)
    
    print(f"\n{'='*60}")
    print("🏁 GATED DATASET TESTING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
