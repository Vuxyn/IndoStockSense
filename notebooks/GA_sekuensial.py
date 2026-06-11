# %% [markdown]
# # Sequential Genetic Algorithm Hyperparameter Optimization
# # IndoBERT-LoRA for Stock Sentiment — Indonesian Stock News
#
# **Baseline untuk perbandingan dengan Surrogate-Assisted Parallel GA.**
# Script ini mengevaluasi SEMUA kandidat secara sekuensial (satu per satu)
# tanpa surrogate, tanpa Spark.
#
# Identik dalam: population_size, generations, subset_fraction, random_seed
# Berbeda dalam: tidak ada paralelisme, tidak ada surrogate screening

# %%
# ─── 1. ENVIRONMENT SETUP ────────────────────────────────────────────────────
!pip uninstall -y torchvision
!pip install -q transformers "datasets>=2.20.0" accelerate evaluate peft \
    "torchao>=0.16.0" scikit-learn pandas numpy tqdm

# %%
import random
import re
import time
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch

# %%
# ─── 2. CUDA CHECK ───────────────────────────────────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE         = "cuda" if CUDA_AVAILABLE else "cpu"
print(f"CUDA available : {CUDA_AVAILABLE}")
print(f"Device         : {DEVICE}")

# %%
# ─── 3. GLOBAL CONFIG ────────────────────────────────────────────────────────
RANDOM_SEED     = 42
MODEL_NAME      = "indobenchmark/indobert-base-p1"
MAX_LENGTH      = 64
OUTPUT_DIR      = "outputs/indobert-lora-stock-sentiment-seq"
DATA_PATH       = Path("data/raw/Dataset-CNBCI-Sentimented.csv")

POPULATION_SIZE = 20          # ← HARUS SAMA dengan GA_paralel_v2.py
GENERATIONS     = 10          # ← HARUS SAMA dengan GA_paralel_v2.py
SUBSET_FRACTION = 0.3         # ← HARUS SAMA dengan GA_paralel_v2.py
EARLY_STOP_PATIENCE = 3       # generasi tanpa perbaikan → hentikan awal

LABEL_MAP    = {"negatif": 0, "netral": 1, "positif": 2}
ID_TO_LABEL  = {v: k for k, v in LABEL_MAP.items()}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Config         : pop={POPULATION_SIZE}, gen={GENERATIONS}, subset={SUBSET_FRACTION}")

# %%
# ─── 4. DATA LOADING & SEQUENTIAL PREPROCESSING ──────────────────────────────
def clean_text(text: str) -> str:
    if text is None or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if not DATA_PATH.exists():
    os.makedirs("data/raw", exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv",
        DATA_PATH
    )

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"judul": "text", "sentimen": "label", "tanggal": "date"})
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[df["label"].isin(LABEL_MAP)].copy()
df["label_id"] = df["label"].map(LABEL_MAP)

print("\n--- SEQUENTIAL DATA PROCESSING (PANDAS) ---")
t_data_start = time.time()

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].apply(lambda x: len(str(x).split()) >= 3)].copy()

# Sequential equivalents of map/reduce operations
total_words_seq   = df["clean_text"].apply(lambda x: len(str(x).split())).sum()   # ← sequential reduce
label_dist_seq    = df["label"].value_counts().to_dict()                           # ← sequential aggregation

t_data_seq = time.time() - t_data_start

print(f"Total records  : {len(df)}")
print(f"Total words    : {total_words_seq:,}")
print(f"Label dist     : {label_dist_seq}")
print(f"[TIME] Data processing (Pandas): {t_data_seq:.4f}s")

# %%
# ─── 5. TRAIN / VAL / TEST SPLIT ─────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label_id"])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# ─── 6. DATASET PREPARATION (TOKENIZE ONCE, GLOBALLY) ───────────────────────
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_hf(frame):
    return Dataset.from_pandas(
        frame[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}),
        preserve_index=False
    )

hf_datasets = DatasetDict({
    "train"     : to_hf(train_df),
    "validation": to_hf(val_df),
    "test"      : to_hf(test_df),
})

def tokenize_all(dataset_dict):
    def tok(batch): return global_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized = dataset_dict.map(tok, batched=True).remove_columns(["text"])
    tokenized.set_format("torch")
    return tokenized

global_tokenized = tokenize_all(hf_datasets)
print("Tokenization complete.")

# %%
# ─── 7. TRAINING & EVALUATION FUNCTION ───────────────────────────────────────

def build_stratified_subset(tokenized_ds, hf_ds, fraction: float):
    """Buat stratified subset dengan proporsi kelas yang terjaga."""
    labels = list(hf_ds["labels"])
    df_idx = pd.DataFrame({"idx": range(len(labels)), "label": labels})
    # Loop eksplisit — tidak pakai groupby.apply agar idx selalu ada
    sampled_parts = []
    for lbl in sorted(df_idx["label"].unique()):
        grp = df_idx[df_idx["label"] == lbl]
        n   = max(1, int(len(grp) * fraction))
        sampled_parts.append(grp.sample(n=n, random_state=RANDOM_SEED))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return tokenized_ds.select(sampled["idx"].tolist())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "f1_weighted"       : float(f1),
        "accuracy"          : float(accuracy_score(labels, preds)),
        "precision_weighted": float(prec),
        "recall_weighted"   : float(rec),
    }

def train_and_evaluate(hyperparameters: dict, use_subset: bool = False,
                        early_stop: bool = False) -> dict:
    """
    Evaluasi satu kandidat hyperparameter secara sekuensial.

    Args:
        hyperparameters : dict hyperparameter kandidat GA
        use_subset      : True → gunakan 30% data (GA evaluation)
                          False → gunakan 100% data (final training)
        early_stop      : True → aktifkan early stopping (patience=3 epoch evaluasi)
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              EarlyStoppingCallback)

    if use_subset:
        train_dataset = build_stratified_subset(global_tokenized["train"], hf_datasets["train"], SUBSET_FRACTION)
        val_dataset   = build_stratified_subset(global_tokenized["validation"], hf_datasets["validation"], SUBSET_FRACTION)
        num_epochs    = 1   # proxy evaluation: 1 epoch cukup
    else:
        train_dataset = global_tokenized["train"]
        val_dataset   = global_tokenized["validation"]
        num_epochs    = int(hyperparameters.get("epochs", 3))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL, label2id=LABEL_MAP,
        ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(hyperparameters["lora_rank"]),
        lora_alpha=int(hyperparameters["lora_alpha"]),
        lora_dropout=float(hyperparameters["lora_dropout"]),
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)

    callbacks = []
    if early_stop and num_epochs > 1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=float(hyperparameters["learning_rate"]),
        per_device_train_batch_size=int(hyperparameters["batch_size"]),
        num_train_epochs=num_epochs,
        weight_decay=float(hyperparameters["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=CUDA_AVAILABLE,
        load_best_model_at_end=early_stop and num_epochs > 1,
        metric_for_best_model="f1_weighted" if early_stop else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=global_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    metrics = trainer.evaluate(val_dataset)
    elapsed = time.time() - t0

    return {
        "individual": hyperparameters,
        "fitness"   : float(metrics.get("eval_f1_weighted", -1.0)),
        "elapsed"   : elapsed,
        "metrics"   : {k: float(v) for k, v in metrics.items()},
    }

# %%
# ─── 8. GENETIC ALGORITHM SEARCH SPACE ───────────────────────────────────────

@dataclass
class SearchSpace:
    learning_rates: tuple = (1e-5, 2e-5, 3e-5, 5e-5)
    batch_sizes   : tuple = (8, 16, 32)
    epochs        : tuple = (1, 2, 3)
    lora_ranks    : tuple = (4, 8, 16)
    lora_alphas   : tuple = (8, 16, 32)
    lora_dropouts : tuple = (0.05, 0.1, 0.2)
    weight_decays : tuple = (0.0, 0.01, 0.05)

space = SearchSpace()

def create_pop(size: int) -> list:
    return [{
        "learning_rate": random.choice(space.learning_rates),
        "batch_size"   : random.choice(space.batch_sizes),
        "lora_rank"    : random.choice(space.lora_ranks),
        "lora_alpha"   : random.choice(space.lora_alphas),
        "lora_dropout" : random.choice(space.lora_dropouts),
        "weight_decay" : random.choice(space.weight_decays),
    } for _ in range(size)]

def mutate(ind: dict, rate: float = 0.2) -> dict:
    mut = ind.copy()
    for k in mut:
        if random.random() < rate:
            mut[k] = random.choice(getattr(space, k + "s"))
    return mut

def crossover(p1: dict, p2: dict) -> dict:
    return {k: (p1[k] if random.random() > 0.5 else p2[k]) for k in p1}

def next_generation(evaluated: list, pop_size: int) -> list:
    parents  = [x["individual"] for x in sorted(evaluated, key=lambda x: x["fitness"], reverse=True)[:len(evaluated)//2]]
    next_pop = [max(evaluated, key=lambda x: x["fitness"])["individual"]]  # elitism
    while len(next_pop) < pop_size:
        child = crossover(random.choice(parents), random.choice(parents))
        next_pop.append(mutate(child))
    return next_pop

def get_key(ind: dict) -> str:
    return str(sorted(ind.items()))

# %%
# ─── 9. SEQUENTIAL GA MAIN LOOP ──────────────────────────────────────────────

population  = create_pop(POPULATION_SIZE)
SEQ_CACHE   = {}

# Timing collectors — detail per generasi
t_per_gen           = []    # waktu per generasi
n_evals_per_gen     = []    # jumlah evaluasi baru per generasi
cache_hits_per_gen  = []    # jumlah cache hit per generasi
best_per_gen        = []    # fitness terbaik per generasi

# Early stopping state
no_improve_count    = 0
global_best_fitness = -1.0

print("=" * 65)
print("  SEQUENTIAL GENETIC ALGORITHM (BASELINE)")
print(f"  Population={POPULATION_SIZE} | Generations={GENERATIONS}")
print(f"  Subset={SUBSET_FRACTION*100:.0f}% | Early stopping patience={EARLY_STOP_PATIENCE}")
print("=" * 65)

ga_start = time.time()

for gen in range(GENERATIONS):
    print(f"\n{'─'*65}")
    print(f"  Gen {gen+1}/{GENERATIONS}")
    print(f"{'─'*65}")
    gen_start  = time.time()
    evaluated  = []
    cache_hits = 0
    new_evals  = 0

    for ind in population:
        key = get_key(ind)
        if key in SEQ_CACHE:
            evaluated.append(SEQ_CACHE[key])
            cache_hits += 1
        else:
            try:
                res = train_and_evaluate(ind, use_subset=True)
            except Exception as e:
                import traceback
                res = {"individual": ind, "fitness": -1.0, "elapsed": 0.0, "error": traceback.format_exc()}
            SEQ_CACHE[key] = res
            evaluated.append(res)
            new_evals += 1
            if res["fitness"] >= 0:
                print(f"    fitness={res['fitness']:.4f} | elapsed={res.get('elapsed',0):.1f}s")
            else:
                err_lines = [l.strip() for l in str(res.get('error','')).strip().splitlines() if l.strip()]
                print(f"    FAILED | HP={res['individual']}")
                for el in err_lines[-3:]:
                    print(f"      >> {el}")

    gen_elapsed = time.time() - gen_start

    valid    = [e for e in evaluated if e["fitness"] >= 0]
    best_gen = max(valid, key=lambda x: x["fitness"]) if valid else None
    best_f   = best_gen["fitness"] if best_gen else -1.0

    t_per_gen.append(gen_elapsed)
    n_evals_per_gen.append(new_evals)
    cache_hits_per_gen.append(cache_hits)
    best_per_gen.append(best_f)

    print(f"  Cache hits: {cache_hits}/{POPULATION_SIZE} ({100*cache_hits/POPULATION_SIZE:.0f}%)")
    print(f"  New evals : {new_evals} | Gen time: {gen_elapsed:.1f}s")
    print(f"  ✦ Best this gen: {best_f:.4f}")

    # ─── Early Stopping (hanya hitung jika ada evaluasi yang berhasil) ────
    if best_f >= 0:   # jangan hitung generasi yang semua gagal
        if best_f > global_best_fitness:
            global_best_fitness = best_f
            no_improve_count    = 0
        else:
            no_improve_count += 1
            print(f"  [Early Stop] No improvement ({no_improve_count}/{EARLY_STOP_PATIENCE})")
            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"  [Early Stop] Triggered at Gen {gen+1}. Stopping GA.")
                GENERATIONS = gen + 1
                break
    else:
        print(f"  [Early Stop] Gen {gen+1} all failed — not counted toward patience.")

    if gen < GENERATIONS - 1:
        population = next_generation(evaluated, POPULATION_SIZE)

ga_time = time.time() - ga_start

# %%
# ─── 10. FINAL TRAINING ───────────────────────────────────────────────────────
all_candidates = sorted(
    [r for r in SEQ_CACHE.values() if r["fitness"] >= 0],
    key=lambda x: x["fitness"],
    reverse=True
)

# Guard: jika semua evaluasi gagal, pakai default HP
if not all_candidates:
    print("[WARN] Tidak ada evaluasi yang berhasil. Menggunakan default hyperparameters.")
    best_hp = {
        "learning_rate": 2e-5, "batch_size": 16,
        "lora_rank": 8, "lora_alpha": 16,
        "lora_dropout": 0.1, "weight_decay": 0.01,
        "epochs": 3,
    }
else:
    best_hp = all_candidates[0]["individual"].copy()
    best_hp["epochs"] = 3

print(f"\n{'='*65}")
print("  FINAL TRAINING — Full Data, 3 Epochs")
print(f"  Best HP: {best_hp}")
print(f"{'='*65}")

t_final_start = time.time()
final_res     = train_and_evaluate(best_hp, use_subset=False)
t_final       = time.time() - t_final_start

final_f1   = float(final_res["fitness"])
final_acc  = float(final_res["metrics"].get("eval_accuracy", 0))
final_prec = float(final_res["metrics"].get("eval_precision_weighted", 0))
final_rec  = float(final_res["metrics"].get("eval_recall_weighted", 0))

# %%
# ─── 11. SUMMARY TABLES ──────────────────────────────────────────────────────

avg_eval_time  = np.mean([r.get("elapsed", 0) for r in SEQ_CACHE.values() if r.get("elapsed", 0) > 0])
total_gpu_eval = sum(n_evals_per_gen)
avg_t_per_gen  = np.mean(t_per_gen)
actual_gens    = len(t_per_gen)

W = 65
def hr(char="─"): return char * W
def row(label, val): print(f"  {label:<40} {val}")

print()
print("═" * W)
print("  HASIL PENELITIAN — SEQUENTIAL GENETIC ALGORITHM")
print("  IndoBERT-LoRA Hyperparameter Optimization (Baseline)")
print("═" * W)

# ── Tabel 1: Per-Generasi Breakdown ──
print()
print(hr())
print("  [TABLE 1] Per-Generation Timing Detail")
print(hr())
print(f"  {'Gen':<6} {'New Evals':<12} {'Cache Hits':<12} {'Best F1':<10} {'Time (s)'}")
print(hr("·"))
for i in range(actual_gens):
    print(f"  {i+1:<6} {n_evals_per_gen[i]:<12} {cache_hits_per_gen[i]:<12} {best_per_gen[i]:<10.4f} {t_per_gen[i]:.1f}")
print(hr("·"))
print(f"  {'TOTAL':<6} {sum(n_evals_per_gen):<12} {sum(cache_hits_per_gen):<12} {max(best_per_gen):<10.4f} {ga_time:.1f}")

# ── Tabel 2: Sequential vs Parallel (untuk diisi dari hasil paralel) ──
print()
print(hr())
print("  [TABLE 2] Comparison: Sequential GA vs Surrogate-Assisted Parallel GA")
print("  (Kolom Parallel diisi manual dari hasil parallel_surrogate_results.json)")
print(hr())
print(f"  {'Metric':<40} {'Sequential':>12}  {'Parallel':>12}")
print(hr("·"))
print(f"  {'Total GA Time (s)':<40} {ga_time:>12.1f}  {'[run parallel]':>12}")
print(f"  {'Avg time per generation (s)':<40} {avg_t_per_gen:>12.1f}  {'[run parallel]':>12}")
print(f"  {'Total GPU evaluations':<40} {total_gpu_eval:>12}  {'[run parallel]':>12}")
print(f"  {'Unique HP configs evaluated':<40} {len(SEQ_CACHE):>12}  {'[run parallel]':>12}")
print(f"  {'Cache hit rate (%)':<40} {100*sum(cache_hits_per_gen)/max(POPULATION_SIZE*actual_gens,1):>11.1f}%  {'[run parallel]':>12}")
print(f"  {'Speedup (x)':<40} {'1.00x':>12}  {'[run parallel]':>12}")
print(f"  {'Best proxy F1':<40} {max(best_per_gen):>12.4f}  {'[run parallel]':>12}")
print(f"  {'Final F1 (full data)':<40} {final_f1:>12.4f}  {'[run parallel]':>12}")

# ── Tabel 3: Timing Breakdown ──
print()
print(hr())
print("  [TABLE 3] Timing Breakdown (seconds)")
print(hr())
row("Data preprocessing (Pandas):", f"{t_data_seq:.4f}s")
row("GA sequential total:", f"{ga_time:.2f}s")
row("Avg time per candidate (GPU):", f"{avg_eval_time:.1f}s")
row("Avg time per generation:", f"{avg_t_per_gen:.1f}s")
row("Actual generations ran:", str(actual_gens))
row("Final training (3 epochs, full data):", f"{t_final:.2f}s")
print(hr("·"))
row("TOTAL wall-clock time:", f"{t_data_seq + ga_time + t_final:.2f}s  ({(t_data_seq + ga_time + t_final)/60:.1f} min)")

# ── Tabel 4: Model Performance ──
print()
print(hr())
print("  [TABLE 4] Final Model Performance (Validation Set)")
print(hr())
row("Best Hyperparameters:", "")
for k, v in best_hp.items():
    row(f"  {k}:", str(v))
print(hr("·"))
row("F1-Score (weighted):", f"{final_f1:.4f}")
row("Accuracy:", f"{final_acc:.4f}")
row("Precision (weighted):", f"{final_prec:.4f}")
row("Recall (weighted):", f"{final_rec:.4f}")

# ── Tabel 5: Search Summary ──
print()
print(hr())
print("  [TABLE 5] GA Search Summary")
print(hr())
row("Total candidates in search space:", str(4*3*3*3*3*3))   # search space size
row("Total HP configs explored:", str(len(SEQ_CACHE)))
row("Coverage (%):", f"{100*len(SEQ_CACHE)/(4*3*3*3*3*3):.1f}%")
row("Best proxy fitness (30% subset, 1ep):", f"{max(best_per_gen):.4f}")
row("Final F1 (100% data, 3ep):", f"{final_f1:.4f}")
row("Early stopping triggered:", "Yes" if no_improve_count >= EARLY_STOP_PATIENCE else "No")

print()
print("═" * W)
print("  SEQUENTIAL BASELINE SELESAI")
print("═" * W)
print()

# %%
# ─── 12. SAVE RESULTS ────────────────────────────────────────────────────────
results_data = {
    "experiment"    : "Sequential GA (Baseline)",
    "config"        : {
        "population_size"    : POPULATION_SIZE,
        "generations_planned": GENERATIONS,
        "generations_actual" : actual_gens,
        "subset_fraction"    : SUBSET_FRACTION,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stopped"      : no_improve_count >= EARLY_STOP_PATIENCE,
    },
    "timing"        : {
        "data_preprocessing_s"  : round(t_data_seq, 4),
        "ga_total_s"            : round(ga_time, 2),
        "avg_per_gen_s"         : round(avg_t_per_gen, 2),
        "avg_per_candidate_s"   : round(avg_eval_time, 2),
        "final_training_s"      : round(t_final, 2),
        "total_wall_clock_s"    : round(t_data_seq + ga_time + t_final, 2),
    },
    "per_gen"       : [
        {
            "gen"        : i + 1,
            "new_evals"  : n_evals_per_gen[i],
            "cache_hits" : cache_hits_per_gen[i],
            "best_f1"    : round(best_per_gen[i], 4),
            "elapsed_s"  : round(t_per_gen[i], 2),
        }
        for i in range(actual_gens)
    ],
    "search"        : {
        "unique_configs_evaluated": len(SEQ_CACHE),
        "total_gpu_evaluations"   : total_gpu_eval,
        "cache_hit_rate"          : round(100*sum(cache_hits_per_gen)/max(POPULATION_SIZE*actual_gens, 1), 1),
        "search_space_size"       : 4*3*3*3*3*3,
    },
    "model"         : {
        "best_hp"              : best_hp,
        "best_proxy_f1"        : round(max(best_per_gen), 4),
        "final_f1_weighted"    : round(final_f1, 4),
        "final_accuracy"       : round(final_acc, 4),
        "final_precision"      : round(final_prec, 4),
        "final_recall"         : round(final_rec, 4),
    },
    "all_eval"      : [
        {
            "hp"     : r["individual"],
            "fitness": round(r["fitness"], 4),
            "elapsed": round(r.get("elapsed", 0), 1)
        } for r in sorted(SEQ_CACHE.values(), key=lambda x: x["fitness"], reverse=True)
        if r["fitness"] >= 0
    ]
}

with open("sequential_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("Results saved to sequential_results.json")
