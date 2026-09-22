import json
import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", 
        torch_dtype=torch.float16
    ).to(device)
    
    model.eval()

    test_file = "/research/hal-afsharim/slurp/dataset/pairs/test_pairs.jsonl"
    print(f"Loading test data from {test_file}...")
    test_data = load_data(test_file)
    
    # Just running a small subset for demonstration (e.g. first 20 records)
    test_data = test_data[:] 

    correct = 0
    predictions = []

    print("Starting evaluation...")
    for idx, sample in enumerate(tqdm(test_data)):
        audio_1_path = f"/research/hal-afsharim/slurp/audio/slurp_real/{sample['audio_1']}"
        audio_2_path = f"/research/hal-afsharim/slurp/audio/slurp_real/{sample['audio_2']}"

        messages = [
            {"role": "system", "content": "You are an expert audio classification assistant. Your task is to determine if two user utterances have the same underlying intent or goal (e.g., both are asking for the weather, both are asking to play music, both are querying an alarm)."},
            {"role": "user", "content": [
                {"type": "text", "text": "Audio 1: "},
                {"type": "audio", "audio_url": audio_1_path},
                {"type": "text", "text": "\nAudio 2: "},
                {"type": "audio", "audio_url": audio_2_path},
                {"type": "text", "text": "\nPlease analyze what the speaker in Audio 1 wants, then analyze what the speaker in Audio 2 wants. Finally, state whether they share the same general intent or goal by writing exactly 'Conclusion: Yes' or 'Conclusion: No' at the very end."}
            ]}
        ]

        # Template formatting
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        try:
            # Explicitly load audio data at the processor's required sampling rate
            audio_arrays = []
            for path in [audio_1_path, audio_2_path]:
                # librosa loads the audios correctly for processing
                audio_arr, _ = librosa.load(path, sr=processor.feature_extractor.sampling_rate)
                audio_arrays.append(audio_arr)
            
            inputs = processor(
                text=text, 
                audio=audio_arrays, 
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            # Predict
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=150)
                
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
            # Extract boolean logic (looking at the conclusion)
            response_lower = response.lower()
            last_yes = response_lower.rfind("yes")
            last_no = response_lower.rfind("no")
            is_same_intent = last_yes > last_no
            actual_same_intent = sample['same_intent']
            
            if is_same_intent == actual_same_intent:
                correct += 1
                
            predictions.append({
                "audio_1": sample["audio_1"],
                "audio_2": sample["audio_2"],
                "text_1": sample.get("text_1"),
                "text_2": sample.get("text_2"),
                "intent_1": sample.get("intent_1"),
                "intent_2": sample.get("intent_2"),
                "actual_same_intent": actual_same_intent,
                "predicted_response": response,
                "predicted_same_intent": is_same_intent
            })

        except Exception as e:
            print(f"Failed on sample {idx}: {e}")

    accuracy = correct / len(test_data)
    print(f"\nEvaluation Complete!")
    print(f"Evaluated {len(test_data)} samples.")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    out_file = "/research/hal-afsharim/slurp/dataset/pairs/predictions.jsonl"
    print(f"Saving predictions to {out_file}")
    with open(out_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

if __name__ == "__main__":
    main()
