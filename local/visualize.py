import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from pathlib import Path
import numpy as np

def format_text(gen_text, ref_text):
    gen_words = gen_text.split()
    ref_words = ref_text.split()

    formatted = []
    for word in gen_words:
        if word in ref_words:
            formatted.append(f"**{word}**")
        else:
            formatted.append(f"_\u0332{word}_")  # italic + underline

    return " ".join(formatted)

def plot_single_example(image_path, pred_label, true_label,
                        gen_text, ref_text, metrics_dict, output_path):

    img = Image.open(image_path).convert("L")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12),
                            gridspec_kw={'width_ratios': [1, 1]})

    # Top-left: actual image
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Chest X-ray")

    # Top-right: labels + metrics
    axs[0, 1].axis("off")
    metrics_text = (f"Predicted Label: {pred_label}\n"
                    f"True Label: {true_label}\n")
    for key, value in metrics_dict.items():
        metrics_text += f"{key.upper()}: {value:.3f}\n"
    metrics_text += "\n**Bold** = correct words\n" \
                    "_Italic underlined_ = mismatch"
    axs[0, 1].text(0, 0.5, metrics_text, fontsize=12, wrap=True, va='center')

    # Bottom-left: generated report
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Generated Report", loc="left")
    axs[1, 0].text(0, 0.5, format_text(gen_text, ref_text),
                   fontsize=10, wrap=True, va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey', alpha=0.2))

    # Bottom-right: reference report
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Reference Report", loc="left")
    axs[1, 1].text(0, 0.5, ref_text, fontsize=10, wrap=True, va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgrey', alpha=0.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def line_plot_multi_y(data, output_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('ROUGE / METEOR', color='tab:blue')
    ax1.plot(data["ROUGEL"], 'o-', label="ROUGE-L", color='tab:blue')
    ax1.plot(data["ROUGE1"], 'o--', label="ROUGE-1", color='tab:cyan')
    ax1.plot(data["ROUGE2"], 'o--', label="ROUGE-2", color='tab:purple')
    ax1.plot(data["METEOR"], 's-', label="METEOR", color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Classification Accuracy', color='tab:green')
    acc = (data["Predicted_Label"] == data["True_Label"]).astype(int)
    ax2.plot(acc, 'x-', label="Accuracy", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.legend(loc="upper right")
    plt.title("Metrics per Image")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()

def plot_epoch_losses(run_dir, vis_dir):
    import pandas as pd
    eval_dir = Path(run_dir) / "evaluations"

    cls_file = eval_dir / "classification_epoch_metrics.csv"
    txt_file = eval_dir / "text_gen_epoch_metrics.csv"

    if not cls_file.exists() or not txt_file.exists():
        print("⚠️ Loss CSV files not found. Skipping epoch loss plot.")
        return

    cls = pd.read_csv(cls_file)
    txt = pd.read_csv(txt_file)

    txt["epoch"] += cls["epoch"].max()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(cls["epoch"], cls["class_loss"], label="Classification Loss", marker="o", color="tab:red")
    ax1.plot(txt["epoch"], txt["text_loss"], label="Text Generation Loss", marker="s", color="tab:orange")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # If accuracy is available, plot it on a second axis
    if "accuracy" in cls.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:green')
        ax2.plot(cls["epoch"], cls["accuracy"], label="Classification Accuracy", marker="x", color="tab:green")
        ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.legend(loc="upper right")
    plt.title("Loss and Accuracy per Epoch")
    plt.tight_layout()
    plt.savefig(vis_dir / "metrics_per_epoch.png")
    plt.close()

def visualize_results(results_csv, output_dir):

    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Sort best and worst by ROUGE-L
    best = df.sort_values("ROUGEL", ascending=False).head(10)
    worst = df.sort_values("ROUGEL", ascending=True).head(10)

    for idx, row in best.iterrows():
        outpath = vis_dir / f"best_{idx}.png"
        metrics_dict = {
            "rouge1": row["ROUGE1"],
            "rouge2": row["ROUGE2"],
            "rougel": row["ROUGEL"],
            "meteor": row["METEOR"]
        }
        plot_single_example(row["Image_Path"], row["Predicted_Label"], row["True_Label"],
                            row["Generated_Text"], row["Reference_Text"],
                            metrics_dict, outpath)

    for idx, row in worst.iterrows():
        outpath = vis_dir / f"worst_{idx}.png"
        metrics_dict = {
            "rouge1": row["ROUGE1"],
            "rouge2": row["ROUGE2"],
            "rougel": row["ROUGEL"],
            "meteor": row["METEOR"]
        }
        plot_single_example(row["Image_Path"], row["Predicted_Label"], row["True_Label"],
                            row["Generated_Text"], row["Reference_Text"],
                            metrics_dict, outpath)

    # Multi-metric line plot (multiple y-axis)
    line_plot_multi_y(df, vis_dir / "metrics_per_image.png")

    # Confusion matrix
    plot_confusion(df["True_Label"], df["Predicted_Label"], vis_dir / "confusion_matrix.png")

    # Loss + accuracy plot (epoch-wise)
    plot_epoch_losses(output_dir, vis_dir)

    print(f"✅ Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to per_image_results.csv from evaluate.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Main output directory to save visualizations")

    args = parser.parse_args()

    visualize_results(args.results_csv, args.output_dir)
