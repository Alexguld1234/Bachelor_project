import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from data import get_dataloader
from model import RadTexModel
from transformers import AutoTokenizer
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_model(model_path, batch_size=8, max_length=128, device=None, print_samples=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    vocab_size = tokenizer.vocab_size

    # Load model
    model = RadTexModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    test_loader = get_dataloader(mode="test", batch_size=batch_size, shuffle=False)
    
    # Prompt for generation
    prompt = "FINAL REPORT\n\n"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    all_preds, all_labels = [], []
    all_generated_texts, all_reference_texts = [], []

    print("\n📋 Generating reports from test data...\n")

    with torch.no_grad():
        for images, reports, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            class_output, generated_ids = model(
                images,
                text_inputs=prompt_inputs.repeat(images.size(0), 1),
                generate=True,
                max_length=max_length
            )

            predicted_labels = (class_output > 0.5).float().cpu().numpy()
            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())

            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reports)

            if print_samples:
                for i, text in enumerate(generated_texts[:5]):
                    print(f"🧠 Generated Report {i+1}:\n{text}\n---\n💬 Ground Truth:\n{reports[i]}\n---")

    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except:
        auc_roc = "N/A (only one class present)"

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_rouge = {k: sum(d[k].fmeasure for d in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]}

    meteor_scores = [meteor_score([ref.split()], gen.split()) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print("\n✅ Evaluation Results:")
    print(f"🔹 Accuracy: {accuracy:.4f}")
    print(f"🔹 AUC-ROC: {auc_roc}" if isinstance(auc_roc, str) else f"🔹 AUC-ROC: {auc_roc:.4f}")
    print(f"🔹 ROUGE: {avg_rouge}")
    print(f"🔹 METEOR: {avg_meteor:.4f}")

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "rouge": avg_rouge,
        "meteor": avg_meteor
    }
