# src/evaluate.py

import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def evaluate_model(model, test_loader, cfg, run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    prompt = "FINAL REPORT\n\n"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    all_preds, all_labels = [], []
    all_generated_texts, all_reference_texts = [], []

    with torch.no_grad():
        for images, reports, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            class_output, generated_ids = model(
                images,
                text_inputs=prompt_inputs.repeat(images.size(0), 1),
                generate=True,
                max_length=cfg.run.max_length
            )

            preds = (class_output > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            all_generated_texts.extend(decoded)
            all_reference_texts.extend(reports)

            for i, text in enumerate(decoded):
                path = run_dir / "generation" / f"report_{i}.txt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text.strip())
                print(f"✅ Saved generated report: {path}")


    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds) if len(set(x[0] for x in all_labels)) > 1 else -1

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_rouge = {k: sum(d[k].fmeasure for d in rouge_scores) / len(rouge_scores) for k in rouge_scores[0]}

    meteor = [meteor_score([ref.split()], gen.split()) for ref, gen in zip(all_reference_texts, all_generated_texts)]
    avg_meteor = sum(meteor) / len(meteor)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "rouge": avg_rouge,
        "meteor": avg_meteor
    }
