#!/usr/bin/env python3
"""
Dataset Testing Script

Tests each conversational dataset loader individually to debug loading issues.
Shows sample counts, data quality, and actual examples.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import (
    create_ultrachat_loader,
    create_openorca_loader, 
    create_lmsys_chat_loader,
    create_chatbot_arena_loader,
    create_sharegpt_loader,
    create_anthropic_hh_loader
)
from data_modules.base import DatasetConfig

def test_dataset_loader(name: str, loader_func, config_kwargs=None):
    """Test a single dataset loader."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {name}")
    print(f"{'='*60}")
    
    try:
        # Create config if provided
        if config_kwargs:
            print(f"📋 Config: {config_kwargs}")
            loader = loader_func(**config_kwargs)
        else:
            print("📋 Using default config")
            loader = loader_func()
        
        print(f"📊 Expected max_samples: {loader.config.max_samples}")
        print(f"📏 Expected max_length: {loader.config.max_length}")
        print(f"📐 Expected min_length: {loader.config.min_length}")
        
        # Load raw data
        print("\n🔄 Loading raw data...")
        raw_data = loader.load_raw_data()
        print(f"✅ Raw data loaded: {len(raw_data)} samples")
        
        # Show first raw example
        if len(raw_data) > 0:
            print(f"\n📝 Sample raw data:")
            print(f"   Keys: {list(raw_data[0].keys())}")
            first_sample = str(raw_data[0])
            if len(first_sample) > 200:
                first_sample = first_sample[:200] + "..."
            print(f"   Sample: {first_sample}")
        
        # Preprocess data
        print("\n🔄 Preprocessing data...")
        processed_data = loader.preprocess(raw_data)
        print(f"✅ Preprocessed data: {len(processed_data)} samples")
        
        # Process examples
        print("\n🔄 Processing examples...")
        final_examples = []
        failed_count = 0
        
        for i, example in enumerate(processed_data[:100]):  # Test first 100
            try:
                result = loader.process_example(example)
                if result is not None:
                    final_examples.append(result)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if i < 5:  # Show first few errors
                    print(f"   ⚠️  Error processing example {i}: {e}")
        
        success_rate = len(final_examples) / min(100, len(processed_data)) * 100
        print(f"✅ Processed examples: {len(final_examples)}/100 tested ({success_rate:.1f}% success)")
        print(f"❌ Failed examples: {failed_count}")
        
        # Show processed examples
        if final_examples:
            print(f"\n📖 Sample processed data:")
            for i, example in enumerate(final_examples[:3]):
                text = example.get('text', '')
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"   Example {i+1}: {text}")
                print(f"   Length: {len(example.get('text', ''))} chars")
                print("")
        
        # Estimate final count if we processed all data
        total_estimated = int(len(final_examples) * len(processed_data) / min(100, len(processed_data)))
        print(f"📊 Estimated final dataset size: {total_estimated} samples")
        
        return {
            'name': name,
            'success': True,
            'raw_count': len(raw_data),
            'processed_count': len(processed_data),
            'final_count': total_estimated,
            'success_rate': success_rate,
            'config': loader.config.__dict__
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"🔍 Traceback:")
        traceback.print_exc()
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'raw_count': 0,
            'processed_count': 0,
            'final_count': 0,
            'success_rate': 0
        }

def main():
    """Test all dataset loaders."""
    print("🚀 DATASET LOADER TESTING SCRIPT")
    print("Testing each conversational dataset loader individually...")
    
    # Test configurations
    test_configs = [
        ("UltraChat", create_ultrachat_loader, {'max_samples': 1000}),
        ("OpenOrca", create_openorca_loader, {'max_samples': 1000}),
        ("ShareGPT", create_sharegpt_loader, {'max_samples': 1000}),
        ("Anthropic HH", create_anthropic_hh_loader, {'max_samples': 1000}),
        ("Chatbot Arena", create_chatbot_arena_loader, {'max_samples': 1000}),
        ("LMSYS Chat", create_lmsys_chat_loader, {'max_samples': 1000}),
    ]
    
    results = []
    
    for name, loader_func, config in test_configs:
        result = test_dataset_loader(name, loader_func, config)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Dataset':<15} {'Status':<8} {'Raw':<8} {'Processed':<10} {'Final':<8} {'Success%':<8}")
    print("-" * 60)
    
    for result in results:
        status = "✅ OK" if result['success'] else "❌ FAIL"
        name = result['name'][:14]
        raw = str(result['raw_count'])
        processed = str(result['processed_count'])
        final = str(result['final_count'])
        success = f"{result['success_rate']:.1f}%"
        
        print(f"{name:<15} {status:<8} {raw:<8} {processed:<10} {final:<8} {success:<8}")
        
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    failed = [r for r in results if not r['success']]
    working = [r for r in results if r['success'] and r['final_count'] > 100]
    
    if failed:
        print(f"❌ Failed loaders: {', '.join([r['name'] for r in failed])}")
        print("   → Need debugging or alternative datasets")
    
    if working:
        print(f"✅ Working loaders: {', '.join([r['name'] for r in working])}")
        total_samples = sum(r['final_count'] for r in working)
        print(f"   → Total available samples: {total_samples:,}")
    
    print(f"\n🎯 Ready to train with working datasets!")

if __name__ == "__main__":
    main()
