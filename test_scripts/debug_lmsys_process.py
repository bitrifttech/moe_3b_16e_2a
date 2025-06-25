#!/usr/bin/env python3
"""Test LMSYS Chat preprocessing and processing."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_lmsys_chat_loader

def test_lmsys_processing():
    print("🔍 TESTING LMSYS CHAT PROCESSING")
    
    # Create loader with very small sample
    loader = create_lmsys_chat_loader(max_samples=50)
    print(f"Config: {loader.config.__dict__}")
    
    try:
        # Load raw data
        raw_data = loader.load_raw_data()
        print(f"Raw samples: {len(raw_data)}")
        
        # Test preprocessing
        print("🔄 Preprocessing...")
        print("Debugging preprocessing...")
        
        for i, raw_example in enumerate(raw_data[:5]):
            print(f"\n--- Raw Example {i} ---")
            print(f"Type: {type(raw_example)}")
            print(f"Content preview: {str(raw_example)[:300]}...")
            
            if isinstance(raw_example, dict):
                conversation_text = loader._format_conversation(raw_example)
                print(f"Conversation text result: {conversation_text is not None}")
                if conversation_text:
                    print(f"Text length: {len(conversation_text)}")
                    print(f"Text preview: {conversation_text[:200]}...")
                else:
                    # Debug why it failed
                    conv_data = raw_example.get("conversation", [])
                    model = raw_example.get("model", "unknown")
                    language = raw_example.get("language", "en")
                    moderation_list = raw_example.get("openai_moderation", [])
                    any_flagged = False
                    if isinstance(moderation_list, list) and len(moderation_list) > 0:
                        any_flagged = any(mod.get("flagged", False) for mod in moderation_list if isinstance(mod, dict))
                    print(f"Conv data length: {len(conv_data)}")
                    print(f"Model: {model}")
                    print(f"Language: {language}")
                    print(f"Any flagged: {any_flagged}")
            else:
                print("Raw example is not a dict - dataset loading issue!")
        
        processed = loader.preprocess(raw_data)
        print(f"Processed samples: {len(processed)}")
        
        if len(processed) > 0:
            print(f"First processed keys: {processed[0].keys()}")
            print(f"First processed text: {processed[0]['text'][:300]}...")
        
        # Test process_example
        print("🔄 Testing process_example...")
        successes = 0
        failures = 0
        
        for i, example in enumerate(processed[:10]):
            result = loader.process_example(example)
            if result:
                successes += 1
                if successes == 1:
                    print(f"✅ SUCCESS: {len(result['text'])} chars")
                    print(f"Text: {result['text'][:200]}...")
            else:
                failures += 1
        
        total = min(10, len(processed))
        print(f"Success rate: {successes}/{total} ({successes/total*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lmsys_processing()
