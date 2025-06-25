#!/usr/bin/env python3
"""Debug _format_conversation method directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.base.dataset_config import DatasetConfig

def debug_format_conversation():
    print("🔍 DEBUGGING _format_conversation METHOD")
    
    # Create loader using factory
    loader = create_lmsys_chat_loader(max_samples=10, max_length=4000)
    
    # Get sample data
    raw_data = loader.load_raw_data()
    
    # Test each example manually
    for i, example in enumerate(raw_data[:5]):
        print(f"\n--- Example {i} ---")
        print(f"Language: {example.get('language')}")
        print(f"Model: {example.get('model')}")
        
        # Try to format
        try:
            result = loader._format_conversation(example)
            if result:
                print(f"✅ Success! Length: {len(result)}")
                print(f"Preview: {result[:200]}...")
            else:
                print("❌ Returned None")
                
                # Debug why it failed
                conversation_data = example.get("conversation", [])
                print(f"Conv length: {len(conversation_data)}")
                
                # Check language filter
                language = example.get("language", "en").lower()
                valid_languages = ["en", "english", "american english", "british english"]
                print(f"Language '{language}' valid: {language in valid_languages}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_format_conversation()
