# src/visualize.py

import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AdamW

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"



def generate_visualizations(model, test_loader, cfg, run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    images, reports, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device).float().unsqueeze(1)

    prompt = "FINAL REPORT\n\n"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_inputs = prompt_inputs.repeat(images.size(0), 1)

    class_output, generated_ids = model(images, text_inputs=text_inputs, generate=True, max_length=cfg.run.max_length)
    predicted_labels = (class_output > 0.5).float().cpu().numpy()
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {int(predicted_labels[i][0])} | True: {int(labels[i][0])}")
        ax.set_xlabel(f"Gen: {generated_texts[i][:30]}...\nTrue: {reports[i][:30]}...")
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = run_dir / "visualizations" / "predictions.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved visualization: {out_path}")