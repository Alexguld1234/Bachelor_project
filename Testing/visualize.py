import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data import get_dataloader
from model import RadTexModel
from transformers import GPT2Tokenizer

def show_predictions(model_path, batch_size=8, device=None, num_samples=8, max_length=30):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 30522  # GPT2 base vocab size
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load model
    model = RadTexModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    test_loader = get_dataloader(mode="test", batch_size=batch_size, shuffle=False)

    # Get a batch
    images, reports, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device).float().unsqueeze(1)

    # Generate dummy input for forward pass (random text tokens)
    text_inputs = torch.randint(0, vocab_size, (images.shape[0], max_length)).to(device)

    # Forward pass
    class_output, text_output = model(images, text_inputs)
    predicted_labels = (class_output > 0.5).float().cpu().numpy()
    generated_ids = text_output.argmax(dim=-1).cpu().tolist()
    generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

    # Plot images and predictions
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images) or i >= num_samples:
            break
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {int(predicted_labels[i])}, True: {int(labels[i].item())}")
        ax.set_xlabel(f"Gen: {generated_texts[i][:50]}...\nGT: {reports[i][:50]}...")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
