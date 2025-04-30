import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from transformers import AutoTokenizer
from data import get_dataloader
from radtex_model import build_model
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_predictions(model_path,
                          batch_size=8,
                          encoder="densenet",
                          decoder="biogpt",
                          prompt="FINAL REPORT\n\n",
                          setup="local",
                          csv_file="Final_AP_url_label_50000.csv",
                          num_datapoints=None,
                          img_size=(224, 224)):

    # Load tokenizer and vocab size
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load model
    model = build_model(encoder, decoder, vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load data
    test_loader = get_dataloader(
        mode="test",
        batch_size=batch_size,
        shuffle=False,
        setup=setup,
        csv_file=csv_file,
        num_samples=num_datapoints,
        img_size=img_size
    )

    # Get one batch
    images, reports, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE).long()

    # Prepare prompt input
    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    text_inputs = prompt_inputs.repeat(images.size(0), 1)

    # Generate predictions
    with torch.no_grad():
        class_output, generated_ids = model(
    images,
    text_inputs=text_inputs,
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

    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predicted_labels[i]}, True: {labels[i].item()}")
        ax.set_xlabel(f"Gen: {generated_texts[i][:40]}...\nRef: {reports[i][:40]}...")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

# Save the figure to output path
    output_path = Path("visualizations") / "prediction_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

# Optional: clean up
    plt.close()
    print(f"📸 Saved visualization to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder", type=str, default="densenet")
    parser.add_argument("--decoder", type=str, default="biogpt")
    parser.add_argument("--prompt", type=str, default="FINAL REPORT\n\n")
    parser.add_argument("--setup", choices=["local", "hpc"], default="local")
    parser.add_argument("--csv_file", type=str, default="Final_AP_url_label_50000.csv")
    parser.add_argument("--num_datapoints", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()

    visualize_predictions(
        model_path=args.model_path,
        batch_size=args.batch_size,
        encoder=args.encoder,
        decoder=args.decoder,
        prompt=args.prompt,
        setup=args.setup,
        csv_file=args.csv_file,
        num_datapoints=args.num_datapoints,
        img_size=(args.img_size, args.img_size)
    )

