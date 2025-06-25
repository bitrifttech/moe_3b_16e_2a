#!/usr/bin/env python3
"""Debug _format_conversation step by step."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_modules.conversational import create_lmsys_chat_loader

def debug_step_by_step():
    print("🔍 DEBUGGING _format_conversation STEP BY STEP")
    
    # Create loader
    loader = create_lmsys_chat_loader(max_samples=5, max_length=4000)
    
    # Load raw data
    raw_data = loader.load_raw_data()
    
    for i, example in enumerate(raw_data):
        print(f"\n{'='*60}")
        print(f"🔍 DEBUGGING EXAMPLE {i}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Check conversation data
            conversation_data = example.get("conversation", [])
            print(f"Step 1 - Conversation data: {len(conversation_data)} turns")
            if not conversation_data or len(conversation_data) < 2:
                print("❌ FAIL: Not enough conversation turns")
                continue
            
            # Step 2: Process turns
            conversation_parts = []
            for j, turn in enumerate(conversation_data):
                role = turn.get("role", "").lower()
                content = turn.get("content", "").strip()
                print(f"  Turn {j}: role='{role}', content_len={len(content)}")
                
                if not content:
                    print(f"  ❌ Turn {j}: Empty content")
                    continue
                
                # Handle different role formats
                if role in ["user", "human"]:
                    conversation_parts.append(f"Human: {content}")
                elif role in ["assistant", "gpt", "ai"]:
                    conversation_parts.append(f"Assistant: {content}")
                elif role == "system":
                    if len(content) < 200 and "helpful assistant" not in content.lower():
                        conversation_parts.append(f"System: {content}")
                        
            print(f"Step 2 - Conversation parts: {len(conversation_parts)}")
            if len(conversation_parts) < 2:
                print("❌ FAIL: Not enough valid conversation parts")
                continue
            
            # Step 3: Join conversation
            text = "\n\n".join(conversation_parts)
            print(f"Step 3 - Text length: {len(text)} chars")
            
            # Step 4: Length check
            min_len = loader.config.min_length
            max_len = loader.config.max_length
            print(f"Step 4 - Length check: {min_len} <= {len(text)} <= {max_len}")
            if len(text) < min_len or len(text) > max_len:
                print(f"❌ FAIL: Length out of bounds")
                continue
            
            # Step 5: PII check
            name_count = text.count("NAME_")
            print(f"Step 5 - NAME_ count: {name_count}")
            if "NAME_" in text and name_count > 3:
                print("❌ FAIL: Too many PII redactions")
                continue
            
            # Step 6: Moderation check
            moderation_list = example.get("openai_moderation", [])
            any_flagged = False
            if isinstance(moderation_list, list) and len(moderation_list) > 0:
                any_flagged = any(mod.get("flagged", False) for mod in moderation_list if isinstance(mod, dict))
            print(f"Step 6 - Moderation flagged: {any_flagged}")
            if any_flagged:
                print("❌ FAIL: Moderation flagged")
                continue
            
            # Step 7: Language check
            language = example.get("language", "en").lower()
            valid_languages = ["en", "english", "american english", "british english"]
            language_ok = language in valid_languages
            print(f"Step 7 - Language check: '{language}' in {valid_languages} = {language_ok}")
            if not language_ok:
                print("❌ FAIL: Language filtered")
                continue
            
            # Step 8: Repetition check (simulate)
            print(f"Step 8 - Repetition check: calling _is_repetitive...")
            try:
                is_repetitive = loader._is_repetitive(text)
                print(f"Step 8 - Is repetitive: {is_repetitive}")
                if is_repetitive:
                    print("❌ FAIL: Too repetitive")
                    continue
            except Exception as e:
                print(f"Step 8 - Repetition check failed: {e}")
            
            print("✅ SUCCESS: All checks passed!")
            print(f"Final text preview: {text[:200]}...")
            
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()
