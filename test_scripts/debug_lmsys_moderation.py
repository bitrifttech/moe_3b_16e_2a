#!/usr/bin/env python3
"""Debug LMSYS Chat moderation field structure."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_lmsys_chat_loader

def debug_moderation():
    print("🔍 DEBUGGING LMSYS CHAT MODERATION FIELD")
    
    # Create loader
    loader = create_lmsys_chat_loader(max_samples=5)
    
    # Load raw data
    raw_data = loader.load_raw_data()
    
    for i, example in enumerate(raw_data):
        print(f"\n--- Example {i} ---")
        print(f"Model: {example.get('model')}")
        print(f"Language: {example.get('language')}")
        
        # Check moderation field
        moderation = example.get("openai_moderation")
        print(f"Moderation type: {type(moderation)}")
        print(f"Moderation content: {moderation}")
        
        if isinstance(moderation, list) and len(moderation) > 0:
            print(f"First moderation item: {moderation[0]}")
            if isinstance(moderation[0], dict):
                flagged = moderation[0].get("flagged", False)
                print(f"Flagged: {flagged}")

if __name__ == "__main__":
    debug_moderation()
