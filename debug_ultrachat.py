#!/usr/bin/env python3
"""Debug UltraChat dataset loading specifically."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_ultrachat_loader

def debug_ultrachat():
    print("🔍 DEBUGGING ULTRACHAT")
    
    # Create loader
    loader = create_ultrachat_loader(max_samples=100)  # Small sample
    
    # Load and preprocess
    raw_data = loader.load_raw_data()
    print(f"Raw samples: {len(raw_data)}")
    print(f"First raw sample: {raw_data[0]}")
    
    processed_data = loader.preprocess(raw_data)
    print(f"Processed samples: {len(processed_data)}")
    print(f"First processed sample: {processed_data[0]}")
    
    # Try processing first few examples
    for i in range(min(5, len(processed_data))):
        try:
            example = processed_data[i]
            print(f"\n--- Example {i} ---")
            print(f"Keys: {example.keys()}")
            print(f"Sample data: {str(example)[:300]}...")
            
            result = loader.process_example(example)
            if result:
                print(f"✅ SUCCESS: {len(result['text'])} chars")
                print(f"Text preview: {result['text'][:200]}...")
            else:
                print("❌ FAILED: process_example returned None")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    debug_ultrachat()
