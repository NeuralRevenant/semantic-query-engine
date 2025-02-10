import json
import random
from pathlib import Path
from itertools import islice

import spacy
from spacy.training import Example

#########################
#  CHECKPOINT UTILITIES  #
#########################
CHECKPOINT_FILE = Path("training_checkpoint.json")
MODEL_DIR = Path("model_nlu")


def load_checkpoint():
    """
    Load the training checkpoint JSON file if it exists.
    Returns a dict with 'epoch_index' and 'chunk_index' keys.
    If not found, return None.
    """
    if CHECKPOINT_FILE.is_file() and MODEL_DIR.is_dir():
        with CHECKPOINT_FILE.open("r") as f:
            return json.load(f)
    return None


def save_checkpoint(epoch_idx: int, chunk_idx: int):
    """
    Save the current training state to JSON so we can resume later.
    """
    checkpoint_data = {"epoch_index": epoch_idx, "chunk_index": chunk_idx}
    with CHECKPOINT_FILE.open("w") as f:
        json.dump(checkpoint_data, f, indent=2)


def get_labels(train_dir: Path) -> set:
    """
    First pass over the training data to collect all unique labels.
    """
    labels = set()
    for file_path in train_dir.glob("*.jsonl"):
        print(f"Collecting labels from {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    label = data.get("label")
                    if label:
                        labels.add(label)
                except json.JSONDecodeError:
                    continue
    return labels


def chunked_line_reader(train_dir: Path, lines_per_chunk: int = 1_000_000):
    """
    Generator that reads the *.jsonl files in `train_dir` and yields
    lists of raw JSON lines, each list having up to `lines_per_chunk` lines.
    """
    current_chunk = []
    for file_path in train_dir.glob("*.jsonl"):
        print(f"Reading lines from {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                current_chunk.append(line)
                if len(current_chunk) >= lines_per_chunk:
                    yield current_chunk
                    current_chunk = []
                    
    # Yield leftover lines
    if current_chunk:
        yield current_chunk


def parse_lines_to_examples(lines, nlp: spacy.Language, labels: set) -> list[Example]:
    """
    Convert a list of raw JSON lines into spaCy Examples.
    Each line is expected to be JSON: {"text": "...", "label": "..."}
    """
    examples = []
    for line in lines:
        try:
            data = json.loads(line)
            text = data.get("text")
            label = data.get("label")
            if text and label:
                doc = nlp.make_doc(text)
                # Skip examples with fewer than 2 tokens
                if len(doc) < 2:
                    continue
                cats = {lbl: 0.0 for lbl in labels}
                cats[label] = 1.0
                examples.append(Example.from_dict(doc, {"cats": cats}))
        except json.JSONDecodeError:
            continue
    return examples


def batch_generator(examples, batch_size: int):
    """
    Yield minibatches of the given size from a list of examples.
    """
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def train_spacy_textcat_model():
    """
    Train spaCy text classification model incrementally using CPU parallelism
    and prompt the user to continue after each chunk is processed.
    """
    # 1) Check for existing checkpoint & model
    resume_data = load_checkpoint()
    if resume_data:
        epoch_start = resume_data["epoch_index"]
        chunk_start = resume_data["chunk_index"]
        print(f"Found existing checkpoint: epoch={epoch_start}, chunk={chunk_start}.")
        print(f"Loading existing model from {MODEL_DIR} ...")
        nlp = spacy.load(MODEL_DIR)
        print("Loaded existing model on CPU. Multi-core parallelism enabled.")
    else:
        epoch_start = 0
        chunk_start = 0
        print("No checkpoint found. Starting fresh.")

        # 2) Load spaCy base pipeline (CPU)
        print("Loading base 'en_core_web_sm' model on CPU...")
        nlp = spacy.load("en_core_web_sm")

        # 3) Remove existing textcat if present
        if "textcat" in nlp.pipe_names:
            nlp.remove_pipe("textcat")

        # 4) Add a new textcat_multilabel component
        config = {
            "model": {
                "@architectures": "spacy.TextCatEnsemble.v2",
                "linear_model": {
                    "@architectures": "spacy.TextCatBOW.v3",
                    "exclusive_classes": True,
                    "length": 262144,
                    "ngram_size": 1,
                    "no_output_layer": False,
                },
                "tok2vec": {
                    "@architectures": "spacy.Tok2Vec.v2",
                    "embed": {
                        "@architectures": "spacy.MultiHashEmbed.v2",
                        "width": 64,
                        "rows": [2000, 2000, 500, 1000, 500],
                        "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                        "include_static_vectors": False,
                    },
                    "encode": {
                        "@architectures": "spacy.MaxoutWindowEncoder.v2",
                        "width": 64,
                        "window_size": 1,
                        "maxout_pieces": 3,
                        "depth": 2,
                    },
                },
            },
        }
        textcat = nlp.add_pipe("textcat_multilabel", config=config, last=True)

        # 5) Set up data directory
        data_dir = Path(
            "/Users/user-home/Downloads/semantic-query-engine/datasets/query_label_datasets/two_M_samples"
        )

        # 6) Gather all labels
        LABEL_SET = get_labels(data_dir)
        if not LABEL_SET:
            raise ValueError("No training data found or no labels found.")
        print(f"Collected labels: {LABEL_SET}")
        for label in LABEL_SET:
            textcat.add_label(label)

        # 7) Gather ~1000 lines for initialization
        init_lines = []
        lines_collected = 0
        init_line_reader = chunked_line_reader(data_dir, lines_per_chunk=10_000)
        for chunk in init_line_reader:
            for line in chunk:
                init_lines.append(line)
                lines_collected += 1
                if lines_collected >= 1000:
                    break
            if lines_collected >= 1000:
                break

        if not init_lines:
            raise ValueError("No lines available for initialization.")
        print(f"Collected {len(init_lines)} lines for initialization.")

        init_examples = parse_lines_to_examples(init_lines, nlp, LABEL_SET)
        if not init_examples:
            raise ValueError("No valid training examples in initialization.")
        print(f"Using {len(init_examples)} examples for initialization.")

        # 8) Initialize the model with these examples
        print("Initializing the model with sample examples...")
        nlp.initialize(lambda: iter(init_examples))
        print("Initialization complete.")

        # 9) Save initial model & checkpoint
        MODEL_DIR.mkdir(exist_ok=True)
        nlp.to_disk(MODEL_DIR)
        save_checkpoint(epoch_start, chunk_start)
        print(f"Initial model saved to {MODEL_DIR}")

    ##################################################
    # 2) MAIN TRAINING PHASE: EPOCHS + CHUNKS
    ##################################################
    data_dir = Path(
        "/Users/user-home/Downloads/semantic-query-engine/datasets/query_label_datasets/two_M_samples"
    )
    epochs = 3
    lines_per_chunk = 1_000_000
    minibatch_size = 128

    LABEL_SET = get_labels(data_dir)
    optimizer = nlp.create_optimizer()

    for epoch in range(epoch_start, epochs):
        losses = {}
        print(f"\n=== Starting Epoch {epoch+1}/{epochs} ===")

        line_reader = chunked_line_reader(data_dir, lines_per_chunk=lines_per_chunk)

        # Process each chunk
        for chunk_idx, chunk in enumerate(line_reader):
            # If resuming from checkpoint, skip processed chunks
            if (
                resume_data
                and epoch == resume_data["epoch_index"]
                and chunk_idx < resume_data["chunk_index"]
            ):
                print(f"Skipping chunk #{chunk_idx} (checkpoint resume).")
                continue

            examples_chunk = parse_lines_to_examples(chunk, nlp, LABEL_SET)
            if not examples_chunk:
                print(f"  Chunk #{chunk_idx} had no valid examples, skipping.")
                continue

            random.shuffle(examples_chunk)
            for batch in batch_generator(examples_chunk, minibatch_size):
                nlp.update(batch, sgd=optimizer, losses=losses)

            print(
                f"  Epoch {epoch+1}, chunk #{chunk_idx} done. ({len(examples_chunk)} examples)"
            )

            # Save model & checkpoint for each chunk
            nlp.to_disk(MODEL_DIR)
            save_checkpoint(epoch, chunk_idx + 1)
            print(f"  [Checkpoint saved: epoch={epoch}, chunk={chunk_idx+1}]")

            # Prompt user: Continue or Exit?
            user_input = input(
                "Press [Enter] to continue to the next chunk, or type anything else to exit: "
            )
            if user_input != "":
                print("Exiting training as requested. Model is saved so far.")
                return  # Stop the entire training process

        # Finished all chunks in this epoch
        print(f"Epoch {epoch+1} complete. Losses: {losses}")

        # Save checkpoint for next epoch
        save_checkpoint(epoch + 1, 0)
        nlp.to_disk(MODEL_DIR)
        print(f"[Epoch {epoch+1} model checkpoint saved]\n")

    print("All epochs complete!")
    print(f"Final model is in '{MODEL_DIR}'.")


if __name__ == "__main__":
    train_spacy_textcat_model()
