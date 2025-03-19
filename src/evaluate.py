import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from data import get_dataloader
from model import RadTexModel
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

# ✅ Load Model & Tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 30522
model = RadTexModel(vocab_size=VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("radtex_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ✅ Load Test Data
test_loader = get_dataloader(mode="test", batch_size=8, shuffle=False)

# ✅ Loss Function
classification_criterion = nn.BCELoss()

def evaluate():
    """
    Evaluate the model on classification and text generation tasks.
    """
    all_preds, all_labels = [], []
    all_generated_texts, all_reference_texts = [], []
    total_class_loss = 0.0

    with torch.no_grad():
        for images, reports, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            # Generate text (dummy token input for inference)
            text_inputs = torch.randint(0, VOCAB_SIZE, (images.shape[0], 30)).to(DEVICE)
            class_output, text_output = model(images, text_inputs)

            # ✅ Classification Evaluation
            class_loss = classification_criterion(class_output, labels)
            total_class_loss += class_loss.item()

            predicted_labels = (class_output > 0.5).float().cpu().numpy()
            all_preds.extend(predicted_labels)
            all_labels.extend(labels.cpu().numpy())

            # ✅ Text Generation Evaluation
            generated_ids = text_output.argmax(dim=-1).cpu().tolist()
            generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reports)  # Assuming `reports` are raw reference texts

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    if len(set(int(label.item()) for label in all_labels)) > 1:
  # Ensure elements are integers before using set()

  # Ensure at least two classes exist
        auc_roc = roc_auc_score(all_labels, all_preds)
    else:
        auc_roc = "N/A (Only one class present)"


    # ✅ BLEU Score (NLTK)
    bleu_score = corpus_bleu([[ref] for ref in all_reference_texts], all_generated_texts)

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

    print(f"🔹 BLEU Score: {bleu_score:.4f}")
    print(f"🔹 ROUGE Scores: {avg_rouge}")
    print(f"🔹 METEOR Score: {avg_meteor_score:.4f}")

if __name__ == "__main__":
    evaluate()
