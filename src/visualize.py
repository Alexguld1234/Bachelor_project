import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data import get_dataloader
from model import RadTexModel
from transformers import GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 30522

# ✅ Load Model
model = RadTexModel(vocab_size=VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("radtex_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

# ✅ Load Test Data
test_loader = get_dataloader(mode="test", batch_size=8, shuffle=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def show_predictions():
    """
    Display images with classification predictions and generated text.
    """
    images, reports, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

    text_inputs = torch.randint(0, VOCAB_SIZE, (images.shape[0], 30)).to(DEVICE)
    class_output, text_output = model(images, text_inputs)

    predicted_labels = (class_output > 0.5).float().cpu().numpy()
    generated_ids = text_output.argmax(dim=-1).cpu().tolist()
    generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {int(predicted_labels[i].item())}, True: {int(labels[i].item())}")
        ax.set_xlabel(f"Generated: {generated_texts[i][:50]}...\nActual: {reports[i][:50]}...")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_predictions()
