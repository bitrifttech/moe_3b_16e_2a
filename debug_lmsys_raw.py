#!/usr/bin/env python3
"""Debug LMSYS Chat raw dataset structure."""

from datasets import load_dataset as hf_load_dataset

def debug_lmsys_raw():
    print("🔍 DEBUGGING LMSYS CHAT RAW DATASET")
    
    try:
        # Load dataset
        dataset = hf_load_dataset("lmsys/lmsys-chat-1m")
        print(f"Dataset keys: {list(dataset.keys())}")
        
        # Get train split
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            split_name = list(dataset.keys())[0]
            train_data = dataset[split_name]
            
        print(f"Train data type: {type(train_data)}")
        print(f"Train data length: {len(train_data)}")
        
        # Check features
        print(f"Features: {train_data.features}")
        
        # Look at a few examples properly
        for i in range(3):
            example = train_data[i]  # Access by index
            print(f"\n--- Example {i} ---")
            print(f"Type: {type(example)}")
            print(f"Keys: {example.keys()}")
            print(f"Model: {example.get('model', 'N/A')}")
            print(f"Language: {example.get('language', 'N/A')}")
            
            conv = example.get('conversation', [])
            print(f"Conversation type: {type(conv)}")
            print(f"Conversation length: {len(conv)}")
            if len(conv) > 0:
                print(f"First turn: {conv[0]}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lmsys_raw()
