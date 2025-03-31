import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from data import get_dataloader
from model import RadTexModel
from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# ✅ Load SciBERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ✅ Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = tokenizer.vocab_size
model = RadTexModel(vocab_size=VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("radtex_model.pth", map_location=DEVICE))
model.eval()

# ✅ Load Data
test_loader = get_dataloader(mode="test", batch_size=8, shuffle=False)

# ✅ Set prompt for generation
prompt = """                               FINAL REPORT
 
"""
prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

def evaluate():
    all_preds, all_labels = [], []
    all_generated_texts, all_reference_texts = [], []

    print("\n📋 Generating Reports from Images Only...\n")

    with torch.no_grad():
        for images, reports, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            class_output, generated_ids = model(
                images,
                text_inputs=prompt_inputs.repeat(images.size(0), 1),
                generate=True,
                max_length=128
            )

            predicted_labels = (class_output > 0.5).float().cpu().numpy()
            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())

            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reports)

            for i, text in enumerate(generated_texts):
                print(f"Generated Report {i+1}: {text}\n---------------------\n{reports[i]}\n---------------------")

    accuracy = accuracy_score(all_labels, all_preds)
    if len(set(x[0] for x in all_labels)) > 1:
        auc_roc = roc_auc_score(all_labels, all_preds)
    else:
        auc_roc = "N/A (Only one class present)"

    # ✅ BLEU Score (NLTK)
    #bleu_score = corpus_bleu([[ref.split()] for ref in all_reference_texts], [gen.split() for gen in all_generated_texts])

    # ✅ ROUGE Score (NLTK)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_rouge = {k: sum(d[k].fmeasure for d in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]}

    # ✅ METEOR Score (NLTK)
    meteor_scores = [
        meteor_score([ref.split()], gen.split()) for ref, gen in zip(all_reference_texts, all_generated_texts)
    ]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

    print("\n✅ **Evaluation Results:**")
    print(f"🔹 Classification Accuracy: {accuracy:.4f}")
    if isinstance(auc_roc, str):
        print(f"🔹 AUC-ROC: {auc_roc}")
    else:
        print(f"🔹 AUC-ROC: {auc_roc:.4f}")

    #print(f"🔹 BLEU Score: {bleu_score:.4f}")
    print(f"🔹 ROUGE Scores: {avg_rouge}")
    print(f"🔹 METEOR Score: {avg_meteor_score:.4f}")

if __name__ == "__main__":
    evaluate()