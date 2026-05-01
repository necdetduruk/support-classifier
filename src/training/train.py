"""Fine-tune DistilBERT on banking77 for intent classification."""
import os
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# --- Config ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("models/banking77-distilbert")
NUM_LABELS = 77
MAX_LENGTH = 64  # banking77 utterances are short
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    print("Loading dataset...")
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)

    # Save label names for inference later
    label_names = ds["train"].features["label"].names
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"Saved {len(label_names)} label names")

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH
        )

    ds_tok = ds.map(tokenize, batched=True)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",  # silence wandb/tensorboard prompts
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating...")
    final_metrics = trainer.evaluate()
    print(f"Final metrics: {final_metrics}")

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save metrics for the README / model card later
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()