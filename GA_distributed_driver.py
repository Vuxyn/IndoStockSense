# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED PGA DRIVER — Pure Parallel Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════════
# Cara pakai:
#   1. Pastikan Spark Master sudah berjalan (start_master.ps1)
#   2. Pastikan kedua Colab Worker sudah terhubung
#   3. Buka PowerShell baru → conda activate spark312
#   4. python GA_distributed_driver.py
# ═══════════════════════════════════════════════════════════════════════════════

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

# ─── 1. CONFIGURATION ───────────────────────────────────────────────────────
MASTER_IP      = "100.70.125.49"                    # IP Tailscale laptop
RANDOM_SEED    = 42
MODEL_NAME     = "indobenchmark/indobert-base-p1"
MAX_LENGTH     = 64
OUTPUT_DIR     = "outputs/indobert-lora-ga-distributed"
DATA_URL       = "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv"
DATA_PATH      = Path("data/raw/Dataset-CNBCI-Sentimented.csv")

POPULATION_SIZE    = 20       # kandidat per generasi
GENERATIONS        = 10        # total generasi
NUM_SLICES         = 2        # 2 Worker = 2 partisi (1 per Worker)

LABEL_MAP    = {"negatif": 0, "netral": 1, "positif": 2}
ID_TO_LABEL  = {v: k for k, v in LABEL_MAP.items()}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  DISTRIBUTED PYSPARK — Pure Parallel Genetic Algorithm")
print(f"  Master: spark://{MASTER_IP}:7077")
print(f"  Config: pop={POPULATION_SIZE}, gen={GENERATIONS}, slices={NUM_SLICES}")
print("=" * 65)

# ─── 2. SPARK SESSION ───────────────────────────────────────────────────────
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("GA_Distributed_IndoBERT")
    .master(f"spark://{MASTER_IP}:7077")
    .config("spark.driver.host", MASTER_IP)
    .config("spark.driver.bindAddress", MASTER_IP)
    .config("spark.executor.memory", "10g")
    .config("spark.executor.cores", "2")
    .config("spark.task.cpus", "1")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.rpc.message.maxSize", "256")
    .config("spark.network.timeout", "800s")
    .config("spark.executor.heartbeatInterval", "120s")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("WARN")
print(f"\nSpark version  : {spark.version}")
print(f"Default parallelism: {sc.defaultParallelism}")

# ─── 3. DATA LOADING (di Driver, untuk statistik & final training) ────────
def clean_text(text: str) -> str:
    if text is None or pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if not DATA_PATH.exists():
    os.makedirs("data/raw", exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.rename(columns={"judul": "text", "sentimen": "label", "tanggal": "date"})
df_raw["label"] = df_raw["label"].astype(str).str.lower().str.strip()
df_raw = df_raw[df_raw["label"].isin(LABEL_MAP)].copy()
df_raw["label_id"] = df_raw["label"].map(LABEL_MAP)
df_raw["clean_text"] = df_raw["text"].apply(clean_text)
df_raw = df_raw[df_raw["clean_text"].apply(lambda x: len(x.split()) >= 3)]

print(f"\nTotal records  : {len(df_raw)}")
print(f"Label dist     : {dict(df_raw['label'].value_counts())}")

# Spark parallel preprocessing (untuk demonstrasi MapReduce)
print("\n--- PARALLEL DATA PROCESSING (PYSPARK) ---")
t_data_start = time.time()

data_list = df_raw[["text", "label", "label_id"]].to_dict("records")
rdd_raw   = sc.parallelize(data_list, numSlices=NUM_SLICES)

rdd_cleaned = rdd_raw.map(lambda row: {                            # ← MAP
    "text"      : row["text"],
    "label"     : row["label"],
    "label_id"  : row["label_id"],
    "clean_text": clean_text(row["text"])
})

rdd_filtered = rdd_cleaned.filter(lambda r: len(r["clean_text"].split()) >= 3)
rdd_filtered.cache()

# REDUCE: hitung total kata
total_words = rdd_filtered.map(lambda r: len(r["clean_text"].split())).reduce(lambda a, b: a + b)

# REDUCE: distribusi label
label_dist  = rdd_filtered.map(lambda r: (r["label"], 1)).reduceByKey(lambda a, b: a + b).collect()

processed_data = rdd_filtered.collect()
df = pd.DataFrame(processed_data)
t_data_par = time.time() - t_data_start

print(f"Total words    : {total_words:,}")
print(f"Label dist     : {dict(label_dist)}")
print(f"[TIME] Data processing (Spark): {t_data_par:.2f}s")

# ─── 4. TRAIN / VAL / TEST SPLIT ────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label_id"])

train_records = train_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
val_records   = val_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
test_records  = test_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")

print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

# ─── 5. WORKER FUNCTION — DISTRIBUTED mapPartitions ─────────────────────────

def evaluate_partition(hyperparams_iter):
    """
    Dipanggil SEKALI per Spark partition (= per Worker/Colab GPU).
    Setiap Worker:
    1. Load dataset dari disk/internet SENDIRI
    2. Load tokenizer & model SENDIRI
    3. Training IndoBERT-LoRA dengan hyperparameter yang dikirim Master
    4. Return fitness score ke Master
    """
    import os, time, traceback, re
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split

    os.environ["TRANSFORMERS_VERBOSITY"]  = "error"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"

    _MODEL_NAME  = "indobenchmark/indobert-base-p1"
    _MAX_LENGTH  = 64
    _OUTPUT_DIR  = "outputs/indobert-lora-ga-distributed"
    _LABEL_MAP   = {"negatif": 0, "netral": 1, "positif": 2}
    _ID_TO_LABEL = {v: k for k, v in _LABEL_MAP.items()}
    _SEED        = 42
    _DATA_URL    = "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv"
    _DATA_PATH   = "data/raw/Dataset-CNBCI-Sentimented.csv"

    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from peft import LoraConfig, TaskType, get_peft_model

    # ── Worker download dataset sendiri jika belum ada ──
    if not os.path.exists(_DATA_PATH):
        os.makedirs("data/raw", exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(_DATA_URL, _DATA_PATH)

    # ── Load & preprocess data di Worker ──
    def clean_text(text):
        if text is None or pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df_raw = pd.read_csv(_DATA_PATH)
    df_raw = df_raw.rename(columns={"judul": "text", "sentimen": "label", "tanggal": "date"})
    df_raw["label"] = df_raw["label"].astype(str).str.lower().str.strip()
    df_raw = df_raw[df_raw["label"].isin(_LABEL_MAP)].copy()
    df_raw["label_id"] = df_raw["label"].map(_LABEL_MAP)
    df_raw["clean_text"] = df_raw["text"].apply(clean_text)
    df_raw = df_raw[df_raw["clean_text"].apply(lambda x: len(x.split()) >= 3)]

    train_df, temp_df = train_test_split(df_raw, test_size=0.2, random_state=_SEED, stratify=df_raw["label_id"])
    val_df, _         = train_test_split(temp_df, test_size=0.5, random_state=_SEED, stratify=temp_df["label_id"])

    # ── Load tokenizer SEKALI per partisi ──
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    # ── Tokenize subset SEKALI per partisi ──
    def tokenize_df(df, frac=0.3):
        sampled_parts = []
        for lbl in sorted(df["label_id"].unique()):
            grp = df[df["label_id"] == lbl]
            n   = max(1, int(len(grp) * frac))
            sampled_parts.append(grp.sample(n=n, random_state=_SEED))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        records = sampled[["clean_text", "label_id"]].rename(
            columns={"clean_text": "text", "label_id": "labels"}
        ).to_dict("records")
        ds = Dataset.from_list(records)
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True,
                                padding="max_length", max_length=_MAX_LENGTH),
            batched=True
        ).remove_columns(["text"])
        ds.set_format("torch")
        return ds

    train_ds = tokenize_df(train_df, frac=0.3)
    val_ds   = tokenize_df(val_df,   frac=0.3)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0)
        return {"f1_weighted": float(f1),
                "accuracy"   : float(accuracy_score(labels, preds))}

    # ── Iterasi setiap kandidat dalam partisi ──
    for hp in hyperparams_iter:
        t0 = time.time()
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME, num_labels=len(_LABEL_MAP),
                id2label=_ID_TO_LABEL, label2id=_LABEL_MAP,
                ignore_mismatched_sizes=True
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(hp["lora_rank"]),
                lora_alpha=int(hp["lora_alpha"]),
                lora_dropout=float(hp["lora_dropout"]),
                target_modules=["query", "value"]
            )
            model = get_peft_model(model, lora_cfg)

            args = TrainingArguments(
                output_dir=_OUTPUT_DIR,
                learning_rate=float(hp["learning_rate"]),
                per_device_train_batch_size=int(hp["batch_size"]),
                num_train_epochs=1,
                weight_decay=float(hp["weight_decay"]),
                eval_strategy="epoch",
                save_strategy="no",
                report_to="none",
                fp16=(torch.cuda.is_available()),
                dataloader_num_workers=0,
            )

            trainer = Trainer(
                model=model, args=args,
                train_dataset=train_ds, eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics
            )
            trainer.train()
            metrics = trainer.evaluate()
            elapsed = time.time() - t0

            # Bersihkan GPU memory
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield {
                "individual": hp,
                "fitness"   : float(metrics.get("eval_f1_weighted", -1.0)),
                "elapsed"   : elapsed,
                "metrics"   : {k: float(v) for k, v in metrics.items()}
            }

        except Exception:
            yield {
                "individual": hp,
                "fitness"   : -1.0,
                "elapsed"   : time.time() - t0,
                "error"     : traceback.format_exc()
            }

# ─── 6. GENETIC ALGORITHM UTILITIES ─────────────────────────────────────────

@dataclass
class SearchSpace:
    learning_rates : tuple = (1e-5, 2e-5, 3e-5, 5e-5)
    batch_sizes    : tuple = (8, 16, 32)
    epochs         : tuple = (1, 2, 3)
    lora_ranks     : tuple = (4, 8, 16)
    lora_alphas    : tuple = (8, 16, 32)
    lora_dropouts  : tuple = (0.05, 0.1, 0.2)
    weight_decays  : tuple = (0.0, 0.01, 0.05)

space = SearchSpace()

def create_individual() -> dict:
    return {
        "learning_rate": random.choice(space.learning_rates),
        "batch_size"   : random.choice(space.batch_sizes),
        "lora_rank"    : random.choice(space.lora_ranks),
        "lora_alpha"   : random.choice(space.lora_alphas),
        "lora_dropout" : random.choice(space.lora_dropouts),
        "weight_decay" : random.choice(space.weight_decays),
    }

def create_pop(size: int) -> list:
    return [create_individual() for _ in range(size)]

def mutate(ind: dict, rate: float = 0.2) -> dict:
    mut = ind.copy()
    for k in mut:
        if random.random() < rate:
            mut[k] = random.choice(getattr(space, k + "s"))
    return mut

def crossover(p1: dict, p2: dict) -> dict:
    return {k: (p1[k] if random.random() > 0.5 else p2[k]) for k in p1}

def next_generation(evaluated: list, pop_size: int) -> list:
    parents = [x["individual"] for x in sorted(evaluated, key=lambda x: x["fitness"], reverse=True)[:len(evaluated)//2]]
    next_pop = [max(evaluated, key=lambda x: x["fitness"])["individual"]]  # elitism
    while len(next_pop) < pop_size:
        child = crossover(random.choice(parents), random.choice(parents))
        next_pop.append(mutate(child))
    return next_pop

def get_key(ind: dict) -> str:
    return str(sorted(ind.items()))

def run_parallel_eval(candidates: list, label: str = "") -> list:
    """
    ═══════════════════════════════════════════════════════════
    ▶ DISTRIBUTED SPARK MAPREDUCE — Full IndoBERT Evaluation
    ═══════════════════════════════════════════════════════════
    MAP    : Spark distributes candidates to Colab Workers.
             Each Worker (partition) trains IndoBERT-LoRA
             independently on its own GPU.

    REDUCE : Collect all results back to driver (laptop),
             then reduce (find global best).
    ═══════════════════════════════════════════════════════════
    """
    if not candidates:
        return []

    n_slices = min(NUM_SLICES, len(candidates))
    rdd = sc.parallelize(candidates, numSlices=n_slices)

    # ▶ MAP: mapPartitions → setiap Worker load data & train sendiri
    results_rdd = rdd.mapPartitions(evaluate_partition)                    # ← MAP

    # ▶ REDUCE: kumpulkan semua hasil ke driver (laptop)
    results = results_rdd.collect()                                        # ← COLLECT

    # ▶ REDUCE: cari fitness terbaik dari semua Worker
    valid = [r for r in results if r["fitness"] >= 0]
    if valid:
        best_result = sc.parallelize(valid).reduce(                       # ← REDUCE
            lambda a, b: a if a["fitness"] > b["fitness"] else b
        )
        if label:
            print(f"  [{label}] Best fitness from reduce: {best_result['fitness']:.4f}")

    return results

# ─── 7. MAIN LOOP — PURE PARALLEL GENETIC ALGORITHM ─────────────────────────

population    = create_pop(POPULATION_SIZE)
EVAL_CACHE    = {}
all_results   = []
t_per_gen     = []
n_gpu_evals   = 0

print("\n" + "=" * 65)
print("  PURE PARALLEL GENETIC ALGORITHM (DISTRIBUTED)")
print(f"  Population={POPULATION_SIZE} | Generations={GENERATIONS}")
print(f"  Workers={NUM_SLICES} (Colab GPUs)")
print("=" * 65)

ga_start = time.time()

for gen in range(GENERATIONS):
    gen_label = f"Gen {gen+1}/{GENERATIONS}"
    print(f"\n{'─'*65}")
    print(f"  {gen_label}")
    print(f"{'─'*65}")
    gen_start = time.time()

    # Filter kandidat yang belum pernah dievaluasi (cache hit)
    to_eval = [ind for ind in population if get_key(ind) not in EVAL_CACHE]
    cached  = POPULATION_SIZE - len(to_eval)
    print(f"  Candidates to evaluate: {len(to_eval)} (cache hits: {cached})")

    if to_eval:
        # ▶ MAP + REDUCE: distribusikan ke Colab Workers
        results = run_parallel_eval(to_eval, label=gen_label)
        for r in results:
            k = get_key(r["individual"])
            EVAL_CACHE[k] = r
            if r["fitness"] >= 0:
                all_results.append(r)
                n_gpu_evals += 1
                print(f"    fitness={r['fitness']:.4f} | elapsed={r.get('elapsed',0):.1f}s")
            else:
                err = r.get("error", "Unknown error")
                print(f"    FAILED | HP={r['individual']}")
                err_lines = [l.strip() for l in err.strip().splitlines() if l.strip()]
                for el in err_lines[-3:]:
                    print(f"      >> {el}")

    gen_elapsed = time.time() - gen_start
    t_per_gen.append(gen_elapsed)

    # Kumpulkan hasil evaluasi untuk populasi saat ini
    evaluated = [EVAL_CACHE[get_key(ind)] for ind in population if get_key(ind) in EVAL_CACHE]
    valid_eval = [e for e in evaluated if e["fitness"] >= 0]
    best_gen  = max(valid_eval, key=lambda x: x["fitness"]) if valid_eval else None
    if best_gen:
        print(f"  ✦ Best this gen: {best_gen['fitness']:.4f} | Gen time: {gen_elapsed:.1f}s")

    # Buat generasi berikutnya (kecuali generasi terakhir)
    if gen < GENERATIONS - 1:
        if valid_eval:
            population = next_generation(evaluated, POPULATION_SIZE)
        else:
            population = create_pop(POPULATION_SIZE)
            print("  [WARN] All candidates failed. Creating fresh random population.")

ga_total = time.time() - ga_start

# ─── 8. BEST HYPERPARAMETERS ─────────────────────────────────────────────────

best_candidates = sorted(
    [r for r in EVAL_CACHE.values() if r["fitness"] >= 0],
    key=lambda x: x["fitness"],
    reverse=True
)

if not best_candidates:
    print("[WARN] Tidak ada evaluasi yang berhasil. Menggunakan default hyperparameters.")
    best_hp = {
        "learning_rate": 2e-5, "batch_size": 16,
        "lora_rank": 8, "lora_alpha": 16,
        "lora_dropout": 0.1, "weight_decay": 0.01,
    }
else:
    best_hp = best_candidates[0]["individual"].copy()

# ─── 9. SUMMARY TABLES ───────────────────────────────────────────────────────

avg_per_gen = np.mean(t_per_gen) if t_per_gen else 0

W = 65

def hr(char="─"): return char * W
def row(label, val): print(f"  {label:<40} {val}")

print()
print("═" * W)
print("  HASIL — DISTRIBUTED PURE PARALLEL GA")
print("  IndoBERT-LoRA Hyperparameter Optimization")
print("═" * W)

# ── Tabel 1: MapReduce Operations ──
print()
print(hr())
print("  [TABLE 1] Distributed MapReduce Operations Summary")
print(hr())
print(f"  {'Operation':<30} {'Phase':<14} {'Unit':<8} {'Time/op'}")
print(hr("·"))
print(f"  {'rdd.map(clean_text)':<30} {'Preprocessing':<14} {'Spark':<8} {t_data_par:.2f}s total")
print(f"  {'rdd.reduce(word_count)':<30} {'Preprocessing':<14} {'Spark':<8} (included above)")
print(f"  {'rdd.reduceByKey(label)':<30} {'Preprocessing':<14} {'Spark':<8} (included above)")
print(f"  {'rdd.mapPartitions(IndoBERT)':<30} {'GA Eval':<14} {'GPU×2':<8} {avg_per_gen:.1f}s/gen")
print(f"  {'rdd.reduce(best_fitness)':<30} {'GA Eval':<14} {'CPU':<8} (included above)")

# ── Tabel 2: Timing ──
print()
print(hr())
print("  [TABLE 2] Timing Breakdown (seconds)")
print(hr())
row("Data preprocessing (Spark parallel):", f"{t_data_par:.2f}s")
row("GA total time:", f"{ga_total:.2f}s  ({ga_total/60:.1f} min)")
row("Average time per generation:", f"{avg_per_gen:.2f}s")
row("Total GPU evaluations:", str(n_gpu_evals))
row("Workers (Colab GPUs):", str(NUM_SLICES))

for i, t in enumerate(t_per_gen):
    row(f"  Gen {i+1}:", f"{t:.2f}s")

# ── Tabel 3: Best HP ──
print()
print(hr())
print("  [TABLE 3] Best Hyperparameters Found")
print(hr())
for k, v in best_hp.items():
    row(f"  {k}:", str(v))
if best_candidates:
    row("Best fitness (proxy, 30% subset):", f"{best_candidates[0]['fitness']:.4f}")

print()
print("═" * W)
print("  ✅ DISTRIBUTED EKSPERIMEN SELESAI")
print("═" * W)

# ─── 10. SAVE RESULTS ────────────────────────────────────────────────────────
results_data = {
    "experiment"  : "Distributed Pure Parallel GA",
    "architecture": f"Master (Laptop {MASTER_IP}) + {NUM_SLICES} Workers (Colab GPU)",
    "config"      : {
        "population_size"   : POPULATION_SIZE,
        "generations"       : GENERATIONS,
        "num_slices"        : NUM_SLICES,
    },
    "timing"      : {
        "data_preprocessing_s"      : round(t_data_par, 2),
        "ga_total_s"                : round(ga_total, 2),
        "avg_per_gen_s"             : round(avg_per_gen, 2),
        "per_gen_s"                 : [round(t, 2) for t in t_per_gen],
    },
    "gpu_eval"    : {
        "total_gpu_evals"           : n_gpu_evals,
    },
    "model"       : {
        "best_hp"                   : best_hp,
        "best_proxy_fitness"        : round(best_candidates[0]["fitness"], 4) if best_candidates else None,
    },
    "all_eval"    : [
        {
            "hp"     : r["individual"],
            "fitness": round(r["fitness"], 4),
            "elapsed": round(r.get("elapsed", 0), 1)
        } for r in sorted(EVAL_CACHE.values(), key=lambda x: x["fitness"], reverse=True)
        if r["fitness"] >= 0
    ]
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/distributed_ga_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print(f"\n✅ Results saved to outputs/distributed_ga_results.json")

spark.stop()
print("✅ Spark session stopped.")
