import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import AdamW
import onnx
import onnxruntime
from tqdm import tqdm

#########################
#   PATH CONFIGURATION  #
#########################
# We place checkpoints and final model in the parent directory of this file.
CURRENT_FILE_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_FILE_DIR.parent

CHECKPOINT_DIR = PARENT_DIR / "distilbert_checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "training_checkpoint.json"
FINAL_MODEL_DIR = PARENT_DIR / "distilbert_intent_final"
ONNX_MODEL_PATH = str(PARENT_DIR / "distilbert_intent.onnx")

#########################
# DATASET DIRECTORY     #
#########################
DATA_DIR = Path(
    "/Users/user-home/Downloads/semantic-query-engine/datasets/query_label_datasets/two_M_samples"
)

#########################
# TRAINING PARAMETERS   #
#########################
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 128
LINES_PER_CHUNK = 1_000_000  # Adjust for memory constraints

LABEL2ID: Dict[str, int] = {}
ID2LABEL: Dict[int, str] = {}


# ----------------------------------------------------
# 1) Checkpoint Utilities
# ----------------------------------------------------
def load_checkpoint() -> Dict[str, Any]:
    """
    Load checkpoint if it exists. Returns a dict with keys:
        - "epoch": <int>,
        - "chunk_index": <int>,
        - "label2id": <Dict[str,int]>
    If not found, returns an empty dict.
    """
    if CHECKPOINT_FILE.is_file() and CHECKPOINT_DIR.is_dir():
        with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_checkpoint(epoch: int, chunk_index: int, label2id: Dict[str, int]):
    """
    Save training state to a JSON file for resuming.
    """
    checkpoint_data = {
        "epoch": epoch,
        "chunk_index": chunk_index,
        "label2id": label2id,
    }
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)


# ----------------------------------------------------
# 2) Data I/O - Reading & Chunking
# ----------------------------------------------------
def get_all_labels(data_dir: Path) -> set:
    """
    Scan the data to collect all unique labels.
    """
    labels = set()
    print("\nCollecting all labels from dataset...")
    for jsonl_path in data_dir.glob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    lbl = record.get("label", None)
                    if lbl:
                        labels.add(lbl)
                except json.JSONDecodeError:
                    continue

    print(f"Found {len(labels)} unique labels.")
    return labels


def chunked_line_reader(data_dir: Path, lines_per_chunk: int):
    """
    Generator that yields up to `lines_per_chunk` lines at a time from all *.jsonl files.
    """
    current_chunk = []
    for jsonl_path in data_dir.glob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                current_chunk.append(line)
                if len(current_chunk) >= lines_per_chunk:
                    yield current_chunk
                    current_chunk = []

    # yield any leftover
    if current_chunk:
        yield current_chunk


# ----------------------------------------------------
# 3) PyTorch Dataset for Each Chunk
# ----------------------------------------------------
class IntentDataset(Dataset):
    def __init__(self, lines: List[str], tokenizer: DistilBertTokenizerFast):
        self.samples = []
        self.tokenizer = tokenizer

        for line in lines:
            try:
                data = json.loads(line)
                text = data.get("text", None)
                label = data.get("label", None)
                if not text or not label:
                    continue
                self.samples.append((text, label))
            except json.JSONDecodeError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        label_id = LABEL2ID[label]
        return text, label_id

    def collate_fn(self, batch):
        texts, label_ids = zip(*batch)
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(label_ids, dtype=torch.long)
        return encodings, labels_tensor


# ----------------------------------------------------
# 4) Model Training Function
# ----------------------------------------------------
def train_distilbert_intent():
    global LABEL2ID, ID2LABEL

    # -----------------------------
    # Use MPS if available
    # -----------------------------
    if not torch.backends.mps.is_available():
        raise EnvironmentError(
            "MPS device not found. This script is intended for Apple Silicon GPUs (M1/M2). "
            "Ensure you're running a PyTorch build with MPS >= 1.12."
        )

    device = torch.device("mps")
    print("Using MPS device for Apple Silicon GPU acceleration.")

    # 1) Load or create checkpoint
    resume_data = load_checkpoint()
    resume_epoch = resume_data.get("epoch", 0)
    resume_chunk = resume_data.get("chunk_index", 0)
    saved_label2id = resume_data.get("label2id", {})

    # 2) Label mapping
    if saved_label2id:
        LABEL2ID = saved_label2id
        ID2LABEL = {v: k for k, v in LABEL2ID.items()}
        print(f"Resuming from checkpoint. Found {len(LABEL2ID)} labels.")
    else:
        # Fresh start
        all_labels = sorted(list(get_all_labels(DATA_DIR)))
        LABEL2ID = {label: i for i, label in enumerate(all_labels)}
        ID2LABEL = {i: label for label, i in LABEL2ID.items()}
        print("Created LABEL2ID:", list(LABEL2ID.items())[:10])

    # 3) Load DistilBERT tokenizer
    print("\nLoading DistilBertTokenizerFast...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # 4) Initialize or load DistilBERT model
    model_path = CHECKPOINT_DIR / "pytorch_model.bin"
    config_path = CHECKPOINT_DIR / "config.json"

    if model_path.is_file() and config_path.is_file():
        print(f"Loading DistilBERT model from checkpoint: {CHECKPOINT_DIR} ...")
        model = DistilBertForSequenceClassification.from_pretrained(
            str(CHECKPOINT_DIR), num_labels=len(LABEL2ID)
        )
    else:
        print("No existing checkpoint, initializing new DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(LABEL2ID)
        )

    model.to(device)

    # 5) Set up optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------
    # Main Training Loop
    # -----------------------------
    for epoch in range(resume_epoch, EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")

        chunk_reader = chunked_line_reader(DATA_DIR, LINES_PER_CHUNK)

        for chunk_idx, chunk in enumerate(chunk_reader):
            # If resuming from checkpoint, skip processed chunks
            if (epoch == resume_epoch) and (chunk_idx < resume_chunk):
                print(f"Skipping chunk {chunk_idx} (checkpoint resume).")
                continue

            dataset = IntentDataset(chunk, tokenizer)
            if len(dataset) == 0:
                print(f"Chunk {chunk_idx} has no valid samples, skipping.")
                continue

            data_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=dataset.collate_fn,
            )

            model.train()
            total_loss = 0.0

            for encodings, labels in tqdm(data_loader, desc=f"Chunk {chunk_idx}"):
                # Move to MPS
                encodings = {k: v.to(device) for k, v in encodings.items()}
                labels = labels.to(device)

                outputs = model(**encodings, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"  [Chunk {chunk_idx}] Avg Loss = {avg_loss:.4f}")

            # Save checkpoint after each chunk
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
            save_checkpoint(epoch, chunk_idx + 1, LABEL2ID)
            print(f"[Checkpoint saved - Epoch={epoch}, Chunk={chunk_idx+1}]")

        # End of epoch, save checkpoint for next epoch
        print(f"=== EPOCH {epoch+1} done. Saving epoch checkpoint... ===")
        model.save_pretrained(CHECKPOINT_DIR)
        tokenizer.save_pretrained(CHECKPOINT_DIR)
        save_checkpoint(epoch + 1, 0, LABEL2ID)

    print("\nAll epochs complete!")
    print("Saving final model to disk...")

    FINAL_MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Final model is saved at: {FINAL_MODEL_DIR}")

    # -----------------------------
    # ONNX Export
    # -----------------------------
    export_model_to_onnx(model, tokenizer, ONNX_MODEL_PATH, device)
    print(f"ONNX model exported to: {ONNX_MODEL_PATH}")

    print("\nVerifying ONNX model on CPU (MPS not supported by ONNX Runtime yet)...")
    verify_onnx_model(ONNX_MODEL_PATH)
    print("ONNX verification complete!")


# ----------------------------------------------------
# 5) ONNX Export Utilities
# ----------------------------------------------------
def export_model_to_onnx(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    output_path: str,
    device: torch.device,
):
    """
    Exports the PyTorch DistilBert model to ONNX format.
    Note: ONNX Runtime doesn't support MPS, so this is CPU-based verification only.
    """
    model.eval()
    model.to(device)

    # Dummy input on MPS
    dummy_text = ["Sample input for ONNX export"]
    dummy_encodings = tokenizer(
        dummy_text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_encodings["input_ids"], dummy_encodings["attention_mask"]),
        output_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def verify_onnx_model(onnx_path: str):
    """
    Run a basic CPU-based inference check with ONNX Runtime (MPS is not supported).
    """
    # 1) Structural check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid.")

    # 2) Quick CPU inference
    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(FINAL_MODEL_DIR)
    test_encodings = tokenizer(
        ["This is a test sample for ONNX verification on Apple Silicon."],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    ort_inputs = {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
    }

    outputs = session.run(None, ort_inputs)
    print("ONNX inference output shape (CPU):", outputs[0].shape)


# ----------------------------------------------------
# 6) Main
# ----------------------------------------------------
if __name__ == "__main__":
    train_distilbert_intent()
