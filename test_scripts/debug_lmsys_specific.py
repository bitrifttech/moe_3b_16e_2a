#!/usr/bin/env python3
"""Debug LMSYS specific formatting issues."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_lmsys_chat_loader

def debug_specific_issue():
    print("🔍 DEBUGGING LMSYS SPECIFIC FORMATTING")
    
    # Create loader
    loader = create_lmsys_chat_loader(max_samples=10, max_length=4000)
    # Manually set min_length for testing
    loader.config.min_length = 20
    
    # Load raw data
    raw_data = loader.load_raw_data()
    
    for i, example in enumerate(raw_data):
        print(f"\n--- Example {i} ---")
        print(f"Model: {example.get('model')}")
        print(f"Language: {example.get('language')}")
        print(f"Conversation length: {len(example.get('conversation', []))}")
        
        # Try to format manually step by step
        conversation_data = example.get("conversation", [])
        if not conversation_data:
            print("❌ No conversation data")
            continue
            
        # Process conversation parts
        conversation_parts = []
        for turn in conversation_data:
            role = turn.get("role", "").lower()
            content = turn.get("content", "").strip()
            
            if not content:
                print(f"❌ Empty content in turn: {turn}")
                continue
            
            # Handle different role formats
            if role in ["user", "human"]:
                conversation_parts.append(f"Human: {content}")
            elif role in ["assistant", "gpt", "ai"]:
                conversation_parts.append(f"Assistant: {content}")
            elif role == "system":
                # Include system messages if they're informative
                if len(content) < 200 and "helpful assistant" not in content.lower():
                    conversation_parts.append(f"System: {content}")
        
        if len(conversation_parts) < 2:
            print(f"❌ Not enough conversation parts: {len(conversation_parts)}")
            continue
            
        # Join conversation
        text = "\n\n".join(conversation_parts)
        print(f"✅ Generated text: {len(text)} chars")
        print(f"Preview: {text[:200]}...")
        
        # Check filters
        if len(text) < 20:
            print("❌ Too short")
        elif len(text) > 4000:
            print("❌ Too long")
        else:
            print("✅ Length OK")
            
        # Language check
        language = example.get("language", "")
        if language in ["English", "en"]:
            print("✅ Language OK")
        else:
            print(f"❌ Language filtered: {language}")

if __name__ == "__main__":
    debug_specific_issue()
