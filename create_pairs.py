import json
import random
from collections import defaultdict
import os

def load_data(jsonl_file):
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_audio_intent_mapping(data_list):
    # Flatten to list of (audio_file, intent, text)
    items = []
    for entry in data_list:
        intent = entry.get("intent")
        sentence = entry.get("sentence")
        recordings = entry.get("recordings", [])
        for rec in recordings:
            if rec.get("file"):
                items.append({
                    "audio_file": rec["file"],
                    "intent": intent,
                    "text": sentence
                })
    return items

def create_pairs(items, num_positive, num_negative):
    # Group by intent
    intent_to_items = defaultdict(list)
    for item in items:
        if item["intent"]:
            intent_to_items[item["intent"]].append(item)
    
    intents = list(intent_to_items.keys())
    
    positive_pairs = []
    # Create positive pairs
    valid_intents = [int_name for int_name, its in intent_to_items.items() if len(its) >= 2]
    
    for _ in range(num_positive):
        if not valid_intents:
            break
        chosen_intent = random.choice(valid_intents)
        item1, item2 = random.sample(intent_to_items[chosen_intent], 2)
        positive_pairs.append({
            "audio_1": item1["audio_file"],
            "audio_2": item2["audio_file"],
            "intent_1": item1["intent"],
            "intent_2": item2["intent"],
            "text_1": item1["text"],
            "text_2": item2["text"],
            "same_intent": True
        })
        
    negative_pairs = []
    # Create negative pairs
    for _ in range(num_negative):
        if len(intents) < 2:
            break
        intent1, intent2 = random.sample(intents, 2)
        if not intent_to_items[intent1] or not intent_to_items[intent2]:
            continue
        item1 = random.choice(intent_to_items[intent1])
        item2 = random.choice(intent_to_items[intent2])
        negative_pairs.append({
            "audio_1": item1["audio_file"],
            "audio_2": item2["audio_file"],
            "intent_1": item1["intent"],
            "intent_2": item2["intent"],
            "text_1": item1["text"],
            "text_2": item2["text"],
            "same_intent": False
        })
        
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs

def process_file(input_file, output_file, num_pos, num_neg):
    print(f"Loading data from {input_file} ...")
    data = load_data(input_file)
    items = extract_audio_intent_mapping(data)
    print(f"Extracted {len(items)} audio recordings.")
    
    pairs = create_pairs(items, num_pos, num_neg)
    
    print(f"Saving {len(pairs)} pairs to {output_file} ...")
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    base_dir = "/research/hal-afsharim/slurp/dataset/slurp"
    out_dir = "/research/hal-afsharim/slurp/dataset/pairs"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Train pairs: generating 5000 positive and 5000 negative samples (can be adjusted)
    process_file(f"{base_dir}/train.jsonl", f"{out_dir}/train_pairs.jsonl", num_pos=5000, num_neg=5000)
    
    # Test pairs: generating 1000 positive and 1000 negative samples
    process_file(f"{base_dir}/test.jsonl", f"{out_dir}/test_pairs.jsonl", num_pos=1000, num_neg=1000)

    print("Dataset generation completed!")
