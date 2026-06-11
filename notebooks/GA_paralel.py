# %% [markdown]
# # Parallel Genetic Algorithm Hyperparameter Optimization
# # IndoBERT-LoRA for Stock Sentiment
#
# Prepared for Google Colab / Kaggle (T4 GPU) using PySpark for parallel execution.

# %%
!pip install -q pyspark transformers "datasets>=2.20.0" accelerate evaluate peft "torchao>=0.16.0" scikit-learn pandas numpy matplotlib seaborn tqdm

# %%
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch

# %%
# Setup CUDA + VideoReader patch
try:
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    try:
        import torchvision.io
        if not hasattr(torchvision.io, "VideoReader"):
            torchvision.io.VideoReader = type("VideoReader", (object,), {})
    except ImportError:
        pass
except Exception:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"

print("CUDA available:", CUDA_AVAILABLE)
print("Device:", DEVICE)

# %%
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GA_Parallel") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print("Spark version:", spark.version)

# %%
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 64
OUTPUT_DIR = "outputs/indobert-lora-stock-sentiment-par"
NUM_SLICES = 2   # safe for 1x T4 GPU (Colab) or 2x T4 (Kaggle)

POPULATION_SIZE = 20
GENERATIONS = 10

LABEL_MAP = {"negatif": 0, "netral": 1, "positif": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# %%
# Download & load data
DATA_PATH = Path("data/raw/Dataset-CNBCI-Sentimented.csv")
if not DATA_PATH.exists():
    !mkdir -p data/raw
    !wget -q -O data/raw/Dataset-CNBCI-Sentimented.csv "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"judul": "text", "sentimen": "label", "tanggal": "date"})
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[df["label"].isin(LABEL_MAP)].copy()
df["label_id"] = df["label"].map(LABEL_MAP)

def clean_text(text: str) -> str:
    if text is None or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# %%
# Parallel Data Processing with PySpark
print("--- PARALLEL DATA PROCESSING (PYSPARK) ---")
start_par = time.time()

data_list = df[["text", "label", "label_id"]].to_dict("records")
rdd = sc.parallelize(data_list, numSlices=NUM_SLICES)
rdd_cleaned = rdd.map(lambda row: {
    "text": row["text"],
    "label": row["label"],
    "label_id": row["label_id"],
    "clean_text": clean_text(row["text"])
})
rdd_filtered = rdd_cleaned.filter(lambda row: len(row["clean_text"].split()) >= 3)
rdd_filtered.cache()

processed_data = rdd_filtered.collect()
df = pd.DataFrame(processed_data)
data_time = time.time() - start_par
print(f"Parallel Data Processing Time: {data_time:.2f}s")
print(f"Total samples after cleaning: {len(df)}")

# %%
# Train/Val/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label_id"])

# Convert to plain lists/dicts so they can be broadcast via PySpark
train_records = train_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
val_records   = val_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
test_records  = test_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")

# Broadcast the data so workers can access it without re-sending over network
bc_train = sc.broadcast(train_records)
bc_val   = sc.broadcast(val_records)

print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

# %%
# Pre-warm model cache to avoid HuggingFace deadlock on first worker call
print("Pre-warming model cache (to prevent worker deadlocks)...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_mdl = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3,
    id2label=ID_TO_LABEL, label2id=LABEL_MAP,
    ignore_mismatched_sizes=True
)
del _mdl
print("Model cache ready.")

# %%
def compute_metrics_fn(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1_weighted": f1, "precision_weighted": precision, "recall_weighted": recall}

def worker_train_eval(hyperparameters: dict) -> dict:
    """
    Self-contained training function designed to run inside a PySpark worker.
    All data is loaded from broadcast variables; no shared state assumed.
    """
    import random, time, os
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Silence HF/transformers logs inside worker
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            DataCollatorWithPadding, Trainer, TrainingArguments
        )
        from peft import LoraConfig, TaskType, get_peft_model

        _MODEL_NAME   = "indobenchmark/indobert-base-p1"
        _MAX_LENGTH   = 64
        _OUTPUT_DIR   = "outputs/indobert-lora-stock-sentiment-par"
        _LABEL_MAP    = {"negatif": 0, "netral": 1, "positif": 2}
        _ID_TO_LABEL  = {v: k for k, v in _LABEL_MAP.items()}
        _SEED         = 42

        # Load data from broadcast
        train_data = bc_train.value   # list of {"text": ..., "labels": ...}
        val_data   = bc_val.value

        # Take 30% stratified subset
        def subsample(records, frac=0.3):
            df = pd.DataFrame(records)
            sampled = df.groupby("labels", group_keys=False).apply(
                lambda x: x.sample(frac=frac, random_state=_SEED), include_groups=False
            )
            return sampled.to_dict("records")

        train_sub = subsample(train_data, frac=0.3)
        val_sub   = subsample(val_data,   frac=0.3)

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

        def tokenize(records):
            ds = Dataset.from_list(records)
            ds = ds.map(
                lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=_MAX_LENGTH),
                batched=True
            ).remove_columns(["text"])
            ds.set_format("torch")
            return ds

        train_ds = tokenize(train_sub)
        val_ds   = tokenize(val_sub)

        cuda_ok = False

        model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME, num_labels=len(_LABEL_MAP),
            id2label=_ID_TO_LABEL, label2id=_LABEL_MAP,
            ignore_mismatched_sizes=True
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(hyperparameters["lora_rank"]),
            lora_alpha=int(hyperparameters["lora_alpha"]),
            lora_dropout=float(hyperparameters["lora_dropout"]),
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, lora_cfg)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
            return {"f1_weighted": float(f1), "accuracy": float(accuracy_score(labels, preds))}

        training_args = TrainingArguments(
            output_dir=_OUTPUT_DIR,
            learning_rate=float(hyperparameters["learning_rate"]),
            per_device_train_batch_size=int(hyperparameters["batch_size"]),
            num_train_epochs=1,
            weight_decay=float(hyperparameters["weight_decay"]),
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            fp16=cuda_ok,
            dataloader_num_workers=0,  # IMPORTANT: avoid fork deadlock inside worker
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics
        )

        t0 = time.time()
        trainer.train()
        metrics = trainer.evaluate()
        elapsed = time.time() - t0

        return {
            "individual": hyperparameters,
            "fitness": float(metrics.get("eval_f1_weighted", metrics.get("eval_f1", -1.0))),
            "elapsed": elapsed,
            "metrics": {k: float(v) for k, v in metrics.items()}
        }

    except Exception as e:
        import traceback
        return {
            "individual": hyperparameters,
            "fitness": -1.0,
            "error": traceback.format_exc()
        }

# %%
@dataclass
class SearchSpace:
    learning_rates: tuple = (1e-5, 2e-5, 3e-5, 5e-5)
    batch_sizes: tuple = (8, 16, 32)
    epochs: tuple = (1, 2, 3)
    lora_ranks: tuple = (4, 8, 16)
    lora_alphas: tuple = (8, 16, 32)
    lora_dropouts: tuple = (0.05, 0.1, 0.2)
    weight_decays: tuple = (0.0, 0.01, 0.05)

space = SearchSpace()

def create_pop(size):
    return [{"learning_rate": random.choice(space.learning_rates),
             "batch_size": random.choice(space.batch_sizes),
             "lora_rank": random.choice(space.lora_ranks),
             "lora_alpha": random.choice(space.lora_alphas),
             "lora_dropout": random.choice(space.lora_dropouts),
             "weight_decay": random.choice(space.weight_decays)} for _ in range(size)]

def mutate(ind):
    mut = ind.copy()
    for k in mut:
        if random.random() < 0.2:
            mut[k] = random.choice(getattr(space, k + "s"))
    return mut

def next_generation(evaluated, pop_size):
    parents = [x["individual"] for x in sorted(evaluated, key=lambda x: x["fitness"], reverse=True)[:len(evaluated)//2]]
    next_pop = [max(evaluated, key=lambda x: x["fitness"])["individual"]]
    while len(next_pop) < pop_size:
        p1, p2 = random.choice(parents), random.choice(parents)
        child = {k: (p1[k] if random.random() > 0.5 else p2[k]) for k in p1}
        next_pop.append(mutate(child))
    return next_pop

# %%
population = create_pop(POPULATION_SIZE)
PAR_CACHE = {}

def get_key(ind): return str(sorted(ind.items()))

print("Starting Parallel GA...")
print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | Slices: {NUM_SLICES}")
ga_start = time.time()

for gen in range(GENERATIONS):
    print(f"--- Generation {gen + 1}/{GENERATIONS} ---")

    unique_to_eval = []
    for ind in population:
        key = get_key(ind)
        if key not in PAR_CACHE and ind not in unique_to_eval:
            unique_to_eval.append(ind)

    if unique_to_eval:
        print(f"  Evaluating {len(unique_to_eval)} unique candidates in parallel...")
        pop_rdd = sc.parallelize(unique_to_eval, numSlices=min(NUM_SLICES, len(unique_to_eval)))
        new_results = pop_rdd.map(worker_train_eval).collect()

        for res in new_results:
            key = get_key(res["individual"])
            PAR_CACHE[key] = res
            if res["fitness"] == -1.0:
                print(f"  ⚠ Worker failed! Error: {res.get('error', 'Unknown')[:200]}")
            else:
                print(f"  ✓ Candidate fitness: {res['fitness']:.4f} (elapsed: {res.get('elapsed', 0):.1f}s)")

    evaluated = [PAR_CACHE[get_key(ind)] for ind in population]
    best = max(evaluated, key=lambda x: x["fitness"])
    print(f"Best fitness this gen: {best['fitness']:.4f}")

    if gen < GENERATIONS - 1:
        population = next_generation(evaluated, POPULATION_SIZE)

ga_time = time.time() - ga_start
print(f"\nParallel GA Time: {ga_time:.2f}s ({ga_time/60:.1f} min)")

# %%
# Final Training with best hyperparameters (full data, 3 epochs)
all_eval = sorted(PAR_CACHE.values(), key=lambda x: x["fitness"], reverse=True)
best_hp = all_eval[0]["individual"].copy()
best_hp["epochs"] = 3

print("\n--- Final Training with Best Hyperparameters ---")
print("Best GA HP:", best_hp)

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_full_ds(records):
    ds = Dataset.from_list(records)
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH),
        batched=True
    ).remove_columns(["text"])
    ds.set_format("torch")
    return ds

train_full = build_full_ds(train_records)
val_full   = build_full_ds(val_records)
test_full  = build_full_ds(test_records)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABEL_MAP),
    id2label=ID_TO_LABEL, label2id=LABEL_MAP,
    ignore_mismatched_sizes=True
)
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=int(best_hp["lora_rank"]),
    lora_alpha=int(best_hp["lora_alpha"]),
    lora_dropout=float(best_hp["lora_dropout"]),
    target_modules=["query", "value"]
)
model = get_peft_model(model, lora_cfg)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"f1_weighted": float(f1), "accuracy": float(accuracy_score(labels, preds))}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=float(best_hp["learning_rate"]),
    per_device_train_batch_size=int(best_hp["batch_size"]),
    num_train_epochs=int(best_hp["epochs"]),
    weight_decay=float(best_hp["weight_decay"]),
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=CUDA_AVAILABLE
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_full, eval_dataset=val_full,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
final_metrics = trainer.evaluate(test_full)
final_f1 = float(final_metrics.get("eval_f1_weighted", final_metrics.get("eval_f1", 0)))
print("Final F1-Score (test):", final_f1)

# %%
# Save results
with open("parallel_results.json", "w") as f:
    json.dump({
        "GA_Time_Seconds": ga_time,
        "Data_Processing_Time": data_time,
        "Best_HP": best_hp,
        "Final_F1": final_f1,
        "All_Metrics": {k: float(v) for k, v in final_metrics.items()}
    }, f, indent=4)

print("\n✅ Results saved to parallel_results.json")
print(json.dumps({"GA_Time": f"{ga_time:.1f}s", "Final_F1": f"{final_f1:.4f}"}, indent=2))
