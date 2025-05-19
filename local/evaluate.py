import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from data import get_dataloader
from transformers import AutoTokenizer
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from radtex_model import build_model

def evaluate(model_path, batch_size, device, encoder, decoder,
             setup="local", csv_file="local/Final_AP_url_label_50000.csv",
             prompt="FINAL REPORT\n\n", num_datapoints=None,
             img_size=(224, 224), output_dir=None):

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size

    model = build_model(encoder, decoder, vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = get_dataloader(
        mode="test",
        batch_size=batch_size,
        shuffle=False,
        setup=setup,
        csv_file=csv_file,
        num_samples=num_datapoints,
        img_size=img_size
    )

    # 👇 Read the CSV so we can access the full dataset with paths
    full_data = pd.read_csv(csv_file)

    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    all_preds, all_labels = [], []
    all_generated_texts = []
    all_reference_texts = []

    rows = []

    print("\n📋 Generating Reports from Images...\n")

    start_idx = 0  # to keep track of which samples we're evaluating

    with torch.no_grad():
        for images, reports, labels in tqdm(test_loader):
            batch_size_actual = images.size(0)

            images = images.to(device)
            labels = labels.to(device).long()

            class_output, generated_ids = model(
                images,
                text_inputs=prompt_inputs.repeat(images.size(0), 1),
                generate=True,
                generation_args={
                    "max_length": 128,
                    "repetition_penalty": 1.2,
                    "top_k": 50,
                    "top_p": 0.95
                }
            )

            predicted_labels = class_output.argmax(dim=1).cpu().numpy()
            generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())
            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reports)

            # For each sample in batch, save results + path
            for i in range(batch_size_actual):
                true_label = labels[i].item()
                pred_label = predicted_labels[i]
                gen_text = generated_texts[i]
                ref_text = reports[i]

                # 📝 Get image path from CSV
                img_path = full_data.iloc[start_idx + i]["local_urls"] if setup == "local" else full_data.iloc[start_idx + i]["hpc_urls"]

                rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                rouge_score = rouge.score(ref_text, gen_text)
                meteor = meteor_score([ref_text.split()], gen_text.split())

                rows.append({
                    "Image_Path": img_path,
                    "Predicted_Label": pred_label,
                    "True_Label": true_label,
                    "Generated_Text": gen_text,
                    "Reference_Text": ref_text,
                    "ROUGE1": rouge_score["rouge1"].fmeasure,
                    "ROUGE2": rouge_score["rouge2"].fmeasure,
                    "ROUGEL": rouge_score["rougeL"].fmeasure,
                    "METEOR": meteor
                })

            start_idx += batch_size_actual  # update position

    # Classification metric
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Text metrics
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

    # 🔥 Save results if output_dir given
    if output_dir:
        output_dir = Path(output_dir)
        (output_dir / "evaluations").mkdir(parents=True, exist_ok=True)

        # Per-image
        pd.DataFrame(rows).to_csv(output_dir / "evaluations" / "per_image_results.csv", index=False)

        # Summary
        summary = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "rouge1": avg_rouge["rouge1"],
            "rouge2": avg_rouge["rouge2"],
            "rougeL": avg_rouge["rougeL"],
            "meteor": avg_meteor
        }

        pd.DataFrame([summary]).to_csv(output_dir / "evaluations" / "final_test_metrics_summary.csv", index=False)

        print(f"\n💾 Saved evaluation results to: {output_dir / 'evaluations'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="densenet121")
    parser.add_argument("--decoder", type=str, default="biogpt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--setup", type=str, choices=["local", "hpc"], default="local")
    parser.add_argument("--csv_file", type=str, default="local/Final_AP_url_label_50000.csv")
    parser.add_argument("--prompt", type=str, default="FINAL REPORT\n\n")
    parser.add_argument("--num_datapoints", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        encoder=args.encoder,
        decoder=args.decoder,
        setup=args.setup,
        csv_file=args.csv_file,
        num_datapoints=args.num_datapoints,
        img_size=(args.img_size, args.img_size),
        output_dir=args.output_dir
    )
