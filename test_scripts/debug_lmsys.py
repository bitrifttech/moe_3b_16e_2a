#!/usr/bin/env python3
"""Debug LMSYS Chat dataset specifically."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_lmsys_chat_loader

def debug_lmsys():
    print("🔍 DEBUGGING LMSYS CHAT")
    
    # Create loader
    loader = create_lmsys_chat_loader(max_samples=10)  # Very small sample
    print(f"Config: {loader.config.__dict__}")
    
    # Load raw data
    raw_data = loader.load_raw_data()
    print(f"Raw samples: {len(raw_data)}")
    
    # Look at first few samples
    for i in range(min(3, len(raw_data))):
        sample = raw_data[i]
        print(f"\n--- Raw Sample {i} ---")
        print(f"Keys: {sample.keys()}")
        
        # Check conversation field specifically
        conversation = sample.get("conversation", [])
        print(f"Conversation type: {type(conversation)}")
        print(f"Conversation length: {len(conversation)}")
        
        if len(conversation) > 0:
            print(f"First conversation item type: {type(conversation[0])}")
            print(f"First conversation item: {conversation[0]}")
            
            if len(conversation) > 1:
                print(f"Second conversation item: {conversation[1]}")

if __name__ == "__main__":
    debug_lmsys()
