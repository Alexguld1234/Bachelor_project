import torch
import torch
from sklearn.metrics import accuracy_score

import nltk

nltk.data.path.append("D:/Bachelor_project/local")
nltk.download('wordnet', download_dir='D:/Bachelor_project/local')
#nltk.data.path.append("/work3/s224228/nltk_data")
#nltk.download('wordnet', download_dir='/work3/s224228/nltk_data')

from data import get_dataloader
from transformers import AutoTokenizer
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from radtex_model import build_model
def evaluate(model_path="radtex_model_epoch1.pth", batch_size=8, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size

    # Load model
    
    model = build_model("densenet", "biogpt", vocab_size=vocab_size).to(device)   # or load names from saved yaml
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_dataloader(mode="test", batch_size=batch_size, shuffle=False)

    prompt = "FINAL REPORT\n\n"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    all_preds, all_labels = [], []
    all_generated_texts, all_reference_texts = [], []

    print("\n📋 Generating Reports from Images...\n")
    with torch.no_grad():
        for images, reports, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device).long()

            class_output, generated_ids = model(
                images,
                text_inputs=prompt_inputs.repeat(images.size(0), 1),
                generate=True,
                max_length=128
            )

            predicted_labels = class_output.argmax(dim=1).cpu().numpy()
            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())

            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reports)

            for i, text in enumerate(generated_texts[:5]):
                print(f"\nGenerated Report {i+1}:\n{text}\n----\nOriginal:\n{reports[i]}\n")

    # Classification Metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Text Metrics
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_rouge = {k: sum(d[k].fmeasure for d in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]}

    meteor_scores = [
        meteor_score([ref.split()], gen.split()) for ref, gen in zip(all_reference_texts, all_generated_texts)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print("\n✅ **Evaluation Results:**")
    print(f"🔹 Accuracy: {accuracy:.4f}")
    print(f"🔹 ROUGE: {avg_rouge}")
    print(f"🔹 METEOR: {avg_meteor:.4f}")

if __name__ == "__main__":
    evaluate()
