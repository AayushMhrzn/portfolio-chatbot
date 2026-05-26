"""
train.py
--------
Domain-adapts a pretrained sentence transformer (all-MiniLM-L6-v2)
using LoRA on your portfolio Q&A pairs.

This is the industry-standard approach:
  - Start from a model already trained for embeddings
  - Fine-tune only LoRA adapters for your domain
  - Much more stable than training raw DistilBERT

Run:    python train.py
Output: models/portfolio-distilbert/
"""

import os
import json
import torch
import random
from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# all-MiniLM-L6-v2: already trained for semantic similarity
# Much better base than raw DistilBERT for embedding tasks
BASE_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR  = "models/portfolio-distilbert"
DATA_FILE   = "data/training_pairs.json"
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
MAX_LENGTH  = 128
MARGIN      = 0.5   # how far apart negative pairs should be pushed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_pairs(filepath=DATA_FILE):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    positive, negative = [], []
    for item in data:
        q = item["question"].strip()
        a = item["answer"].strip()
        if item.get("label", "positive") == "negative":
            negative.append((q, a))
        else:
            positive.append((q, a))

    print(f"Positive pairs: {len(positive)} | Negative pairs: {len(negative)}")
    return positive, negative


def build_hard_negatives(positive_pairs, n=200):
    """
    Build HARD negatives: take a question and pair it with an answer
    from a DIFFERENT question that shares some keywords.
    These are harder for the model to distinguish — better training signal.
    """
    hard_negatives = []
    questions = [q for q, _ in positive_pairs]
    answers   = [a for _, a in positive_pairs]

    for i, (q, correct_a) in enumerate(positive_pairs):
        q_words = set(q.lower().split())
        # Find answers from questions that share some words (harder negatives)
        candidates = []
        for j, other_a in enumerate(answers):
            if j == i:
                continue
            other_q_words = set(questions[j].lower().split())
            overlap = len(q_words & other_q_words)
            if overlap >= 2:  # shares at least 2 words — hard negative
                candidates.append(other_a)
        if candidates:
            hard_negatives.append((q, random.choice(candidates)))
        if len(hard_negatives) >= n:
            break

    print(f"Built {len(hard_negatives)} hard negative pairs")
    return hard_negatives


# ─── DATASET ──────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Triplet dataset: (anchor, positive, negative)
    Anchor   = question
    Positive = correct answer
    Negative = wrong answer (hard negative)

    Triplet loss is more stable than cosine embedding loss
    for small datasets — standard in production embedding training.
    """

    def __init__(self, positive_pairs, negative_pairs, tokenizer):
        self.triplets  = []
        self.tokenizer = tokenizer

        # Build triplets: for each positive pair, find a negative
        neg_answers = [a for _, a in negative_pairs]
        random.shuffle(neg_answers)

        for i, (q, pos_a) in enumerate(positive_pairs):
            neg_a = neg_answers[i % len(neg_answers)]
            self.triplets.append((q, pos_a, neg_a))

    def __len__(self):
        return len(self.triplets)

    def tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        a_tok = self.tokenize(anchor)
        p_tok = self.tokenize(positive)
        n_tok = self.tokenize(negative)
        return {
            "a_ids":  a_tok["input_ids"].squeeze(0),
            "a_mask": a_tok["attention_mask"].squeeze(0),
            "p_ids":  p_tok["input_ids"].squeeze(0),
            "p_mask": p_tok["attention_mask"].squeeze(0),
            "n_ids":  n_tok["input_ids"].squeeze(0),
            "n_mask": n_tok["attention_mask"].squeeze(0),
        }


# ─── MODEL ────────────────────────────────────────────────────────────────────

class SentenceEmbedder(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)

        # LoRA config for MiniLM attention layers
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()

    def mean_pool(self, token_embeddings, attention_mask):
        mask        = attention_mask.unsqueeze(-1).float()
        sum_emb     = torch.sum(token_embeddings * mask, dim=1)
        sum_mask    = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.mean_pool(out.last_hidden_state, attention_mask)
        # L2 normalize — standard for cosine similarity
        return nn.functional.normalize(emb, p=2, dim=1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(model, dataloader, tokenizer, epochs=EPOCHS, lr=LR):
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    # TripletMarginLoss: standard for embedding/retrieval models
    # Ensures: dist(anchor, positive) + margin < dist(anchor, negative)
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - nn.functional.cosine_similarity(x, y),
        margin=MARGIN,
    )

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    model.to(device)
    best_loss  = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        progress   = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            a_emb = model(batch["a_ids"].to(device), batch["a_mask"].to(device))
            p_emb = model(batch["p_ids"].to(device), batch["p_mask"].to(device))
            n_emb = model(batch["n_ids"].to(device), batch["n_mask"].to(device))

            loss  = triplet_loss(a_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer)
            print(f"  ✅ Best model saved (loss: {best_loss:.4f})")

    return model


# ─── SAVE ─────────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir=OUTPUT_DIR):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, tokenizer):
    print("\n--- Evaluation ---")
    model.eval()
    model.to(device)

    def embed(text):
        tok = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            return model(tok["input_ids"].to(device), tok["attention_mask"].to(device))

    cos = nn.CosineSimilarity(dim=1)
    test_cases = [
        ("What projects has Aayush built?",
         "Aayush built Audio Driven Facial Animation, Network Intrusion Detection, Interview ChatBot, RAG Chatbot, DIY Home CCTV.",
         "MATCH"),
        ("Where is Aayush located?",
         "Aayush is based in Gwarko, Lalitpur, Nepal.",
         "MATCH"),
        ("What projects has Aayush built?",
         "Aayush is based in Gwarko, Lalitpur, Nepal.",
         "MISMATCH"),
        ("Does Aayush know Docker?",
         "Aayush knows Docker and has containerized production applications.",
         "MATCH"),
        ("Does Aayush know Docker?",
         "Aayush completed the Deep Learning Specialization from Coursera.",
         "MISMATCH"),
        ("What certifications does Aayush have?",
         "Aayush has Deep Learning Specialization, PyTorch Bootcamp, TensorFlow Keras Bootcamp certifications.",
         "MATCH"),
        ("What certifications does Aayush have?",
         "Aayush won the HACK-a-LITE hackathon at Kantipur Engineering College.",
         "MISMATCH"),
    ]

    correct = 0
    print(f"\n{'Question':<45} {'Expected':<10} {'Score':>6} {'Pass'}")
    print("-" * 70)
    for q, a, expected in test_cases:
        score  = cos(embed(q), embed(a)).item()
        passed = (expected == "MATCH" and score > 0.5) or \
                 (expected == "MISMATCH" and score < 0.5)
        icon   = "✅" if passed else "❌"
        if passed:
            correct += 1
        print(f"{q[:44]:<45} {expected:<10} {score:>6.3f} {icon}")

    accuracy = correct / len(test_cases) * 100
    print(f"\nEvaluation accuracy: {correct}/{len(test_cases)} ({accuracy:.0f}%)")
    return accuracy


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Sentence Transformer + LoRA Fine-tuning")
    print("  Base: sentence-transformers/all-MiniLM-L6-v2")
    print("=" * 60)

    # Load data
    print(f"\n[1/5] Loading pairs from {DATA_FILE}...")
    positive, negative = load_pairs()

    # Build hard negatives and combine
    print("\n[2/5] Building hard negatives...")
    hard_negs = build_hard_negatives(positive, n=300)
    all_negatives = negative + hard_negs
    print(f"Total negatives: {len(all_negatives)}")

    # Tokenizer
    print("\n[3/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Dataset
    print("\n[4/5] Building triplet dataset...")
    dataset    = TripletDataset(positive, all_negatives, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Triplets: {len(dataset)} | Batches: {len(dataloader)}")

    # Model
    print("\n[5/5] Loading base model + applying LoRA...")
    model = SentenceEmbedder()

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Estimated time on CPU: ~{EPOCHS * len(dataloader) // 10} minutes\n")
    model = train(model, dataloader, tokenizer)

    # Evaluate
    accuracy = evaluate(model, tokenizer)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Base model:     {BASE_MODEL}")
    print(f"  Training pairs: {len(positive)} positive + {len(all_negatives)} negative")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Eval accuracy:  {accuracy:.0f}%")
    print(f"  Model saved to: {OUTPUT_DIR}/")
    print("\nNext step: run python embed.py")


if __name__ == "__main__":
    main()