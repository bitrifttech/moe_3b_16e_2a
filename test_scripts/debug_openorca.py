#!/usr/bin/env python3
"""Debug OpenOrca dataset specifically."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_openorca_loader

def debug_openorca():
    print("🔍 DEBUGGING OPENORCA")
    
    # Create loader
    loader = create_openorca_loader(max_samples=100)
    print(f"Config: {loader.config.__dict__}")
    
    # Load and preprocess
    raw_data = loader.load_raw_data()
    print(f"Raw samples: {len(raw_data)}")
    print(f"First raw sample: {raw_data[0]}")
    
    processed_data = loader.preprocess(raw_data)
    print(f"Processed samples: {len(processed_data)}")
    
    # Try processing first few examples
    for i in range(min(5, len(processed_data))):
        try:
            example = processed_data[i]
            print(f"\n--- Example {i} ---")
            print(f"Keys: {example.keys()}")
            print(f"Question: {example.get('question', 'N/A')[:100]}...")
            print(f"Response: {example.get('response', 'N/A')[:100]}...")
            
            result = loader.process_example(example)
            if result:
                print(f"✅ SUCCESS: {len(result['text'])} chars")
                print(f"Text preview: {result['text'][:200]}...")
            else:
                print("❌ FAILED: process_example returned None")
                
                # Debug why it failed
                question = example.get("question", "").strip()
                response = example.get("response", "").strip()
                
                if not question or not response:
                    print("  Reason: Missing question or response")
                    continue
                
                # Build text like the method does
                conversation_parts = []
                system_prompt = example.get("system_prompt", "").strip()
                if system_prompt and len(system_prompt) < 200:
                    if not system_prompt.lower().startswith("you are a helpful assistant"):
                        conversation_parts.append(f"System: {system_prompt}")
                
                conversation_parts.append(f"Human: {question}")
                conversation_parts.append(f"Assistant: {response}")
                text = "\n\n".join(conversation_parts)
                
                print(f"  Text length: {len(text)} (min: {loader.config.min_length}, max: {loader.config.max_length})")
                
                if len(text) < loader.config.min_length:
                    print("  Reason: Too short")
                elif len(text) > loader.config.max_length:
                    print("  Reason: Too long")
                else:
                    # Check for bad patterns
                    lower_response = response.lower()
                    bad_patterns = [
                        "i cannot",
                        "i can't help", 
                        "i'm not able to",
                        "i don't have access",
                        "as an ai language model",
                        "i'm just a computer program"
                    ]
                    bad_count = sum(1 for pattern in bad_patterns if pattern in lower_response)
                    if bad_count > 1 or (bad_count == 1 and len(response) < 100):
                        print(f"  Reason: Bad patterns ({bad_count} found)")
                    else:
                        print("  Reason: Unknown error")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_openorca()
