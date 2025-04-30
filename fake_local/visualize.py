import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data import get_dataloader
from transformers import AutoTokenizer
from radtex_model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_predictions(model_path="radtex_model.pth", batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token

    
    model = build_model("densenet", "biogpt", vocab_size=vocab_size).to(device)   # or load names from saved yaml
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_loader = get_dataloader(mode="test", batch_size=batch_size, shuffle=False)
    images, reports, labels = next(iter(test_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE).long()

    prompt = "FINAL REPORT\n\n"
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    text_inputs = prompt_inputs.repeat(images.size(0), 1)

    with torch.no_grad():
        class_output, generated_ids = model(images, text_inputs=text_inputs, generate=True, max_length=128)

    predicted_labels = class_output.argmax(dim=1).cpu().numpy()
    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predicted_labels[i]}, True: {labels[i].item()}")
        ax.set_xlabel(f"Gen: {generated_texts[i][:40]}...\nRef: {reports[i][:40]}...")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_predictions()
