# %% [markdown]
# # Surrogate-Assisted Parallel Genetic Algorithm
# # IndoBERT-LoRA Hyperparameter Optimization — Indonesian Stock Sentiment
#
# **Arsitektur:**
# - Phase 1 (Gen 1): Pure PGA warm-up — evaluasi penuh IndoBERT, kumpulkan data surrogate
# - Phase 2 (Gen 2-N): Surrogate-Assisted PGA:
#   - **MAP**  : Spark mendistribusikan prediksi surrogate ke worker (CPU, cepat)
#   - **FILTER**: Ambil top-K kandidat hasil surrogate screening
#   - **MAP**  : Spark mendistribusikan evaluasi IndoBERT ke worker (GPU)
#   - **REDUCE**: Kumpulkan fitness terbaik dari semua worker
#
# Comparison Table: Pure PGA (extrapolated) vs Surrogate-Assisted PGA (actual)

# %%
# ─── 1. ENVIRONMENT SETUP ───────────────────────────────────────────────────
# torchvision di-uninstall dulu karena bentrok dengan torchao di Colab
!pip uninstall -y torchvision
!pip install -q pyspark transformers "datasets>=2.20.0" accelerate evaluate peft \
    "torchao>=0.16.0" scikit-learn pandas numpy matplotlib seaborn tqdm

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import torch

# %%
# ─── 2. CUDA & SPARK CHECK ──────────────────────────────────────────────────
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

print(f"CUDA available : {CUDA_AVAILABLE}")
print(f"Device         : {DEVICE}")

# %%
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("GA_Surrogate_Parallel")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "2g")
    .config("spark.python.worker.memory", "1g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print(f"Spark version  : {spark.version}")

# %%
# ─── 3. GLOBAL CONFIG ───────────────────────────────────────────────────────
RANDOM_SEED    = 42
MODEL_NAME     = "indobenchmark/indobert-base-p1"
MAX_LENGTH     = 64
OUTPUT_DIR     = "outputs/indobert-lora-ga-surrogate"
DATA_PATH      = Path("data/raw/Dataset-CNBCI-Sentimented.csv")

POPULATION_SIZE    = 20       # kandidat per generasi
GENERATIONS        = 10       # total generasi
WARMUP_GENS        = 2        # generasi awal sebagai "pure PGA" → lebih banyak data surrogate
SURROGATE_POOL     = 50       # kandidat yang di-screen oleh surrogate per generasi
SURROGATE_TOPK     = 5        # top-K kandidat surrogate yang dievaluasi penuh IndoBERT
MIN_SURROGATE_DATA = 5        # minimum data untuk melatih surrogate (turunkan threshold)
NUM_SLICES         = 2        # Spark partitions (2 lebih aman untuk Colab T4 single GPU)

LABEL_MAP    = {"negatif": 0, "netral": 1, "positif": 2}
ID_TO_LABEL  = {v: k for k, v in LABEL_MAP.items()}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Config         : pop={POPULATION_SIZE}, gen={GENERATIONS}, warmup={WARMUP_GENS}")
print(f"Surrogate      : pool={SURROGATE_POOL}, top-k={SURROGATE_TOPK}")

# %%
# ─── 4. DATA LOADING & PARALLEL PREPROCESSING ───────────────────────────────
# MAP: setiap record dibersihkan secara paralel
# REDUCE: filter + count di seluruh partisi

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

df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.rename(columns={"judul": "text", "sentimen": "label", "tanggal": "date"})
df_raw["label"] = df_raw["label"].astype(str).str.lower().str.strip()
df_raw = df_raw[df_raw["label"].isin(LABEL_MAP)].copy()
df_raw["label_id"] = df_raw["label"].map(LABEL_MAP)

print("\n--- PARALLEL DATA PROCESSING (PYSPARK) ---")
t_data_start = time.time()

# MAP: distribusikan clean_text ke seluruh partisi
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

# REDUCE: hitung total kata di semua dokumen
total_words = rdd_filtered.map(lambda r: len(r["clean_text"].split())).reduce(lambda a, b: a + b)  # ← REDUCE

# REDUCE: distribusi label
label_dist  = rdd_filtered.map(lambda r: (r["label"], 1)).reduceByKey(lambda a, b: a + b).collect()  # ← REDUCE

processed_data = rdd_filtered.collect()
df = pd.DataFrame(processed_data)
t_data_par = time.time() - t_data_start

print(f"Total records  : {len(df)}")
print(f"Total words    : {total_words:,}")
print(f"Label dist     : {dict(label_dist)}")
print(f"[TIME] Data processing (Spark): {t_data_par:.2f}s")

# %%
# ─── 5. TRAIN / VAL / TEST SPLIT + BROADCAST ────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label_id"])

train_records = train_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
val_records   = val_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")
test_records  = test_df[["clean_text", "label_id"]].rename(columns={"clean_text": "text", "label_id": "labels"}).to_dict("records")

# Broadcast data ke semua worker — hindari re-transmisi setiap task
bc_train = sc.broadcast(train_records)
bc_val   = sc.broadcast(val_records)

print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

# %%
# ─── 6. PRE-WARM MODEL CACHE ─────────────────────────────────────────────────
print("Pre-warming model cache...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_mdl = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
    ignore_mismatched_sizes=True
)
del _mdl
print("Model cache ready.")

# %%
# ─── 7. WORKER FUNCTION — mapPartitions ─────────────────────────────────────
# Menggunakan mapPartitions agar tokenizer & data hanya di-load SEKALI per partisi
# bukan setiap elemen (yang terjadi jika menggunakan map biasa)

def evaluate_partition(hyperparams_iter):
    """
    Dipanggil SEKALI per Spark partition.
    Semua kandidat dalam partisi berbagi tokenizer dan dataset yang sama.
    Ini menghemat N × model-loading overhead vs rdd.map().
    """
    import os, time, traceback
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    os.environ["TRANSFORMERS_VERBOSITY"]  = "error"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"

    _MODEL_NAME  = "indobenchmark/indobert-base-p1"
    _MAX_LENGTH  = 64
    _OUTPUT_DIR  = "outputs/indobert-lora-ga-surrogate"
    _LABEL_MAP   = {"negatif": 0, "netral": 1, "positif": 2}
    _ID_TO_LABEL = {v: k for k, v in _LABEL_MAP.items()}
    _SEED        = 42

    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from peft import LoraConfig, TaskType, get_peft_model

    # ── Load tokenizer SEKALI per partisi ──
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    # ── Tokenize subset SEKALI per partisi ──
    def tokenize_list(records, frac=0.3):
        df = pd.DataFrame(records)

        sampled_parts = []
        for lbl in sorted(df["labels"].unique()):
            grp = df[df["labels"] == lbl]
            n   = max(1, int(len(grp) * frac))
            sampled_parts.append(grp.sample(n=n, random_state=_SEED))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        # Guard: pastikan 'labels' tidak hilang
        if "labels" not in sampled.columns:
            raise RuntimeError(f"'labels' column missing! Columns: {sampled.columns.tolist()}")
        ds = Dataset.from_list(sampled.to_dict("records"))
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True,
                                padding="max_length", max_length=_MAX_LENGTH),
            batched=True
        ).remove_columns(["text"])
        ds.set_format("torch")
        return ds

    train_ds = tokenize_list(bc_train.value, frac=0.3)
    val_ds   = tokenize_list(bc_val.value,   frac=0.3)

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

# %%
# ─── 8. SURROGATE MODEL UTILITIES ───────────────────────────────────────────

LR_VALS      = [1e-5, 2e-5, 3e-5, 5e-5]
BS_VALS      = [8, 16, 32]
RANK_VALS    = [4, 8, 16]
ALPHA_VALS   = [8, 16, 32]
DROPOUT_VALS = [0.05, 0.1, 0.2]
WD_VALS      = [0.0, 0.01, 0.05]

def hp_to_features(hp: dict) -> list:
    """Encode hyperparameter dict → numeric feature vector untuk surrogate model."""
    def idx(val, lst): return lst.index(val) / max(len(lst) - 1, 1)
    return [
        idx(hp["learning_rate"], LR_VALS),
        idx(hp["batch_size"],    BS_VALS),
        idx(hp["lora_rank"],     RANK_VALS),
        idx(hp["lora_alpha"],    ALPHA_VALS),
        idx(hp["lora_dropout"],  DROPOUT_VALS),
        idx(hp["weight_decay"],  WD_VALS),
        np.log10(hp["learning_rate"]),
        float(hp["lora_rank"]),
        float(hp["lora_alpha"]),
        hp["lora_alpha"] / max(hp["lora_rank"], 1),   # alpha/rank ratio (best practice LoRA)
    ]

surrogate_model  = None
surrogate_scaler = StandardScaler()
surrogate_data   = []   # list of (features, fitness)

def train_surrogate(data: list) -> GradientBoostingRegressor:
    """Latih surrogate model dari data evaluasi IndoBERT yang sudah terkumpul."""
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    X_scaled = surrogate_scaler.fit_transform(X)
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=RANDOM_SEED
    )
    model.fit(X_scaled, y)
    return model

def surrogate_predict_worker(candidate: dict, bc_surrogate, bc_scaler) -> dict:
    """
    Worker untuk MAP phase surrogate screening.
    Ringan (CPU only) — tidak perlu GPU, tidak perlu load model besar.
    """
    import numpy as np
    import pickle, io

    # De-serialize surrogate dari broadcast
    sur_model  = pickle.loads(bc_surrogate.value)
    sur_scaler = pickle.loads(bc_scaler.value)

    lr_vals      = [1e-5, 2e-5, 3e-5, 5e-5]
    bs_vals      = [8, 16, 32]
    rank_vals    = [4, 8, 16]
    alpha_vals   = [8, 16, 32]
    dropout_vals = [0.05, 0.1, 0.2]
    wd_vals      = [0.0, 0.01, 0.05]

    def idx(val, lst): return lst.index(val) / max(len(lst) - 1, 1)

    hp  = candidate
    feat = [
        idx(hp["learning_rate"], lr_vals),
        idx(hp["batch_size"],    bs_vals),
        idx(hp["lora_rank"],     rank_vals),
        idx(hp["lora_alpha"],    alpha_vals),
        idx(hp["lora_dropout"],  dropout_vals),
        idx(hp["weight_decay"],  wd_vals),
        np.log10(hp["learning_rate"]),
        float(hp["lora_rank"]),
        float(hp["lora_alpha"]),
        hp["lora_alpha"] / max(hp["lora_rank"], 1),
    ]
    X = sur_scaler.transform([feat])
    predicted_fitness = float(sur_model.predict(X)[0])
    return {"individual": hp, "predicted_fitness": predicted_fitness}

# %%
# ─── 9. GENETIC ALGORITHM UTILITIES ─────────────────────────────────────────

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
    ▶ SPARK MAPREDUCE PIPELINE — Full IndoBERT Evaluation
    ═══════════════════════════════════════════════════════════
    MAP    : Spark distributes candidates to workers.
             Each worker (partition) trains IndoBERT-LoRA
             independently on its assigned candidates.

    REDUCE : Collect all results back to driver, then reduce
             (find global best) using Python reduce + max.
    ═══════════════════════════════════════════════════════════
    """
    if not candidates:
        return []

    n_slices = min(NUM_SLICES, len(candidates))
    rdd = sc.parallelize(candidates, numSlices=n_slices)

    # ▶ MAP: mapPartitions → load model & tokenizer sekali per partisi
    results_rdd = rdd.mapPartitions(evaluate_partition)                    # ← MAP

    # ▶ REDUCE: kumpulkan semua hasil ke driver
    results = results_rdd.collect()                                        # ← COLLECT (implicit Reduce)

    # ▶ REDUCE: cari fitness terbaik dari semua worker
    valid = [r for r in results if r["fitness"] >= 0]
    if valid:
        best = valid[0]
        best_result = sc.parallelize(valid).reduce(                       # ← REDUCE
            lambda a, b: a if a["fitness"] > b["fitness"] else b
        )
        if label:
            print(f"  [{label}] Best fitness from reduce: {best_result['fitness']:.4f}")

    return results

# %%
# ─── 10. MAIN LOOP — SURROGATE-ASSISTED PGA ──────────────────────────────────

population    = create_pop(POPULATION_SIZE)
EVAL_CACHE    = {}
all_results   = []   # akumulasi semua hasil evaluasi nyata

# Timing collectors
t_pure_per_gen    = []   # waktu evaluasi per gen di fase warm-up (untuk extrapolasi)
t_surrogate_screen = []  # waktu surrogate screening per gen
t_topk_eval       = []   # waktu evaluasi top-K per gen
n_gpu_evals_pure  = 0    # jumlah evaluasi GPU di pure PGA
n_gpu_evals_sur   = 0    # jumlah evaluasi GPU di surrogate PGA

print("=" * 65)
print("  SURROGATE-ASSISTED PARALLEL GENETIC ALGORITHM")
print(f"  Population={POPULATION_SIZE} | Generations={GENERATIONS}")
print(f"  Warm-up={WARMUP_GENS} gen | Surrogate pool={SURROGATE_POOL} | Top-K={SURROGATE_TOPK}")
print("=" * 65)

ga_start = time.time()

for gen in range(GENERATIONS):
    gen_label = f"Gen {gen+1}/{GENERATIONS}"
    print(f"\n{'─'*65}")
    print(f"  {gen_label}")
    print(f"{'─'*65}")
    gen_start = time.time()

    # ─── FASE WARM-UP (PURE PGA) ───────────────────────────────────────────
    if gen < WARMUP_GENS:
        print(f"  Mode: PURE PGA (warm-up, Gen {gen+1})")

        to_eval = [ind for ind in population if get_key(ind) not in EVAL_CACHE]
        print(f"  Candidates to evaluate: {len(to_eval)} (cache hits: {POPULATION_SIZE - len(to_eval)})")

        if to_eval:
            # ▶ MAP + REDUCE (full IndoBERT, lihat fungsi run_parallel_eval)
            results = run_parallel_eval(to_eval, label=gen_label)
            for r in results:
                k = get_key(r["individual"])
                EVAL_CACHE[k] = r
                if r["fitness"] >= 0:
                    all_results.append(r)
                    feat = hp_to_features(r["individual"])
                    surrogate_data.append((feat, r["fitness"]))
                    n_gpu_evals_pure += 1
                    print(f"    fitness={r['fitness']:.4f} | elapsed={r.get('elapsed',0):.1f}s")
                else:
                    # Tampilkan full error untuk debugging
                    err = r.get('error', 'Unknown error')
                    print(f"    FAILED | HP={r['individual']}")
                    # Cari baris error utama (bukan traceback header)
                    err_lines = [l.strip() for l in err.strip().splitlines() if l.strip()]
                    # Tampilkan 3 baris terakhir error (paling informatif)
                    for el in err_lines[-3:]:
                        print(f"      >> {el}")

        gen_elapsed = time.time() - gen_start
        t_pure_per_gen.append(gen_elapsed)

        evaluated = [EVAL_CACHE[get_key(ind)] for ind in population if get_key(ind) in EVAL_CACHE]
        best_gen  = max(evaluated, key=lambda x: x["fitness"]) if evaluated else None
        if best_gen:
            print(f"  ✦ Best this gen: {best_gen['fitness']:.4f} | Gen time: {gen_elapsed:.1f}s")

    # ─── FASE SURROGATE-ASSISTED ───────────────────────────────────────────
    else:
        print(f"  Mode: SURROGATE-ASSISTED PGA (Gen {gen+1})")

        # ── Step 1: (Re-)train surrogate ──────────────────────────────────
        t_sur_train_start = time.time()
        if len(surrogate_data) >= MIN_SURROGATE_DATA:
            surrogate_model = train_surrogate(surrogate_data)
            t_sur_train = time.time() - t_sur_train_start
            print(f"  Surrogate trained on {len(surrogate_data)} samples ({t_sur_train:.2f}s)")
        else:
            # Tidak cukup data → fallback ke pure eval
            print(f"  [WARN] Surrogate data insufficient ({len(surrogate_data)} < {MIN_SURROGATE_DATA}), fallback to pure eval")
            surrogate_model = None

        if surrogate_model is None:
            # Fallback: evaluate semua (seperti pure PGA)
            to_eval = [ind for ind in population if get_key(ind) not in EVAL_CACHE]
            if to_eval:
                results = run_parallel_eval(to_eval, label=gen_label)
                for r in results:
                    k = get_key(r["individual"])
                    EVAL_CACHE[k] = r
                    if r["fitness"] >= 0:
                        all_results.append(r)
                        surrogate_data.append((hp_to_features(r["individual"]), r["fitness"]))
                        n_gpu_evals_sur += 1
            gen_elapsed = time.time() - gen_start
            evaluated = [EVAL_CACHE[get_key(ind)] for ind in population if get_key(ind) in EVAL_CACHE]
        else:
            import pickle

            bc_surrogate = sc.broadcast(pickle.dumps(surrogate_model))
            bc_scaler    = sc.broadcast(pickle.dumps(surrogate_scaler))

            # ── Step 2: Generate large candidate pool ──────────────────────
            candidate_pool = [create_individual() for _ in range(SURROGATE_POOL)]

            # ── Step 3: MAP — Surrogate screening (CPU, sangat cepat) ──────
            # ═══════════════════════════════════════════════════════════════
            # ▶ MAP: Spark distributes surrogate prediction to all workers.
            #        CPU-only operation — no GPU contention.
            #        This is where TRUE parallelism happens.
            # ═══════════════════════════════════════════════════════════════
            t_screen_start = time.time()
            pool_rdd = sc.parallelize(candidate_pool, numSlices=NUM_SLICES)
            screened = pool_rdd.map(                                      # ← MAP (CPU, parallel)
                lambda hp: surrogate_predict_worker(hp, bc_surrogate, bc_scaler)
            ).collect()
            t_screen = time.time() - t_screen_start
            t_surrogate_screen.append(t_screen)
            print(f"  [MAP] Surrogate screening: {len(screened)} candidates in {t_screen:.2f}s")

            # ── Step 4: REDUCE — Ambil top-K dari hasil screening ──────────
            # ═══════════════════════════════════════════════════════════════
            # ▶ REDUCE: Aggregate predicted fitness → select top-K
            # ═══════════════════════════════════════════════════════════════
            screened_sorted = sorted(screened, key=lambda x: x["predicted_fitness"], reverse=True)
            topk_candidates = [s["individual"] for s in screened_sorted[:SURROGATE_TOPK]]
            # Filter yang sudah di-cache
            topk_to_eval = [hp for hp in topk_candidates if get_key(hp) not in EVAL_CACHE]
            print(f"  [REDUCE] Top-{SURROGATE_TOPK} by surrogate → {len(topk_to_eval)} need full eval")

            # ── Step 5: MAP — Full IndoBERT eval pada top-K ────────────────
            t_topk_start = time.time()
            if topk_to_eval:
                results_topk = run_parallel_eval(topk_to_eval, label=f"{gen_label} TopK")
                for r in results_topk:
                    k = get_key(r["individual"])
                    EVAL_CACHE[k] = r
                    if r["fitness"] >= 0:
                        all_results.append(r)
                        surrogate_data.append((hp_to_features(r["individual"]), r["fitness"]))
                        n_gpu_evals_sur += 1
                        print(f"    ✓ fitness={r['fitness']:.4f} (surrogate predicted {screened_sorted[[get_key(c['individual']) for c in screened_sorted].index(get_key(r['individual']))]['predicted_fitness']:.4f}) | elapsed={r.get('elapsed',0):.1f}s")
            t_topk = time.time() - t_topk_start
            t_topk_eval.append(t_topk)

            # Gabungkan top-K hasil nyata dengan kandidat populasi yang sudah di-cache
            eval_from_population = []
            for ind in population:
                k = get_key(ind)
                if k in EVAL_CACHE:
                    eval_from_population.append(EVAL_CACHE[k])

            # Tambahkan top-K hasil baru
            new_topk_results = [EVAL_CACHE[get_key(hp)] for hp in topk_candidates if get_key(hp) in EVAL_CACHE]
            evaluated = eval_from_population + [r for r in new_topk_results if r not in eval_from_population]
            if not evaluated:
                evaluated = list(EVAL_CACHE.values())

            gen_elapsed = time.time() - gen_start

            bc_surrogate.unpersist()
            bc_scaler.unpersist()

            valid_eval = [e for e in evaluated if e["fitness"] >= 0]
            best_gen   = max(valid_eval, key=lambda x: x["fitness"]) if valid_eval else None
            if best_gen:
                print(f"  ✦ Best this gen: {best_gen['fitness']:.4f} | Gen time: {gen_elapsed:.1f}s")

    valid_eval = [e for e in EVAL_CACHE.values() if e["fitness"] >= 0]
    if gen < GENERATIONS - 1:
        if valid_eval:
            population = next_generation(list(EVAL_CACHE.values()), POPULATION_SIZE)
        else:
            # Semua gagal — jangan re-use populasi yang sama, buat yang baru
            population = create_pop(POPULATION_SIZE)
            print("  [WARN] All candidates failed. Creating fresh random population.")

ga_total = time.time() - ga_start

# %%
# ─── 11. FINAL TRAINING ───────────────────────────────────────────────────────
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model

best_candidates = sorted(
    [r for r in EVAL_CACHE.values() if r["fitness"] >= 0],
    key=lambda x: x["fitness"],
    reverse=True
)

# Guard: jika tidak ada evaluasi yang berhasil, pakai default HP
if not best_candidates:
    print("[WARN] Tidak ada evaluasi yang berhasil. Menggunakan default hyperparameters.")
    best_hp = {
        "learning_rate": 2e-5, "batch_size": 16,
        "lora_rank": 8, "lora_alpha": 16,
        "lora_dropout": 0.1, "weight_decay": 0.01,
    }
else:
    best_hp = best_candidates[0]["individual"].copy()

print(f"\n{'='*65}")
print("  FINAL TRAINING — Full Data, 3 Epochs")
print(f"  Best HP: {best_hp}")
print(f"{'='*65}")

t_final_start  = time.time()
tokenizer      = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_ds(records):
    ds = Dataset.from_list(records)
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True,
                            padding="max_length", max_length=MAX_LENGTH),
        batched=True
    ).remove_columns(["text"])
    ds.set_format("torch")
    return ds

train_full = build_ds(train_records)
val_full   = build_ds(val_records)
test_full  = build_ds(test_records)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
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

def compute_metrics_final(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "f1_weighted"       : float(f1),
        "accuracy"          : float(accuracy_score(labels, preds)),
        "precision_weighted": float(prec),
        "recall_weighted"   : float(rec),
    }

final_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=float(best_hp["learning_rate"]),
    per_device_train_batch_size=int(best_hp["batch_size"]),
    num_train_epochs=3,
    weight_decay=float(best_hp["weight_decay"]),
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=CUDA_AVAILABLE,
)
trainer = Trainer(
    model=model, args=final_args,
    train_dataset=train_full, eval_dataset=val_full,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics_final
)
trainer.train()
final_metrics = trainer.evaluate(test_full)
t_final = time.time() - t_final_start

final_f1       = float(final_metrics.get("eval_f1_weighted", 0))
final_acc      = float(final_metrics.get("eval_accuracy", 0))
final_prec     = float(final_metrics.get("eval_precision_weighted", 0))
final_rec      = float(final_metrics.get("eval_recall_weighted", 0))

# %%
# ─── 12. COMPARISON & SUMMARY TABLES ─────────────────────────────────────────

# ── Kalkulasi Pure PGA (extrapolasi) ──
avg_pure_per_gen       = np.mean(t_pure_per_gen) if t_pure_per_gen else 0
estimated_pure_ga_time = avg_pure_per_gen * GENERATIONS   # extrapolasi jika semua gen = pure

# ── Kalkulasi Surrogate PGA (aktual) ──
t_warmup        = sum(t_pure_per_gen)
t_surrogate_ops = sum(t_surrogate_screen) + sum(t_topk_eval)
actual_sur_time = ga_total   # termasuk semua overhead

speedup_ga      = estimated_pure_ga_time / max(actual_sur_time, 1)
gpu_eval_ratio  = n_gpu_evals_pure / max(n_gpu_evals_sur, 1)

# ── MapReduce Phase Summary ──
sur_screen_avg = np.mean(t_surrogate_screen) if t_surrogate_screen else 0
topk_eval_avg  = np.mean(t_topk_eval)        if t_topk_eval        else 0

W = 65

def hr(char="─"): return char * W
def row(label, val): print(f"  {label:<40} {val}")

print()
print("═" * W)
print("  HASIL PENELITIAN — SURROGATE-ASSISTED PARALLEL GA")
print("  IndoBERT-LoRA Hyperparameter Optimization")
print("═" * W)

# ── Tabel 1: MapReduce Operation Summary ──
print()
print(hr())
print("  [TABLE 1] MapReduce Operations Summary")
print(hr())
print(f"  {'Operation':<28} {'Phase':<14} {'Unit':<8} {'Time/op'}")
print(hr("·"))
print(f"  {'rdd.map(clean_text)':<28} {'Preprocessing':<14} {'Spark':<8} {t_data_par:.2f}s total")
print(f"  {'rdd.reduce(word_count)':<28} {'Preprocessing':<14} {'Spark':<8} (included above)")
print(f"  {'rdd.reduceByKey(label)':<28} {'Preprocessing':<14} {'Spark':<8} (included above)")
print(f"  {'rdd.mapPartitions(IndoBERT)':<28} {'Warm-up Eval':<14} {'GPU':<8} {avg_pure_per_gen:.1f}s/gen")
print(f"  {'rdd.map(surrogate_predict)':<28} {'Sur. Screening':<14} {'CPU':<8} {sur_screen_avg:.2f}s/gen")
print(f"  {'rdd.reduce(best_fitness)':<28} {'Sur. Screening':<14} {'CPU':<8} (included above)")
print(f"  {'rdd.mapPartitions(IndoBERT)':<28} {'Top-K Eval':<14} {'GPU':<8} {topk_eval_avg:.1f}s/gen")

# ── Tabel 2: Perbandingan Pure PGA vs Surrogate PGA ──
print()
print(hr())
print("  [TABLE 2] Pure PGA vs Surrogate-Assisted PGA Comparison")
print(hr())
print(f"  {'Metric':<40} {'Pure PGA':>10}  {'Surrogate':>10}")
print(hr("·"))
print(f"  {'GA Total Time (s)':<40} {estimated_pure_ga_time:>10.1f}  {actual_sur_time:>10.1f}")
print(f"  {'Speedup (x)':<40} {'1.00x':>10}  {speedup_ga:>9.2f}x")
print(f"  {'GPU Evaluations (estimated/actual)':<40} {n_gpu_evals_pure*GENERATIONS:>10}  {n_gpu_evals_sur:>10}")
print(f"  {'Avg surrogate screening time/gen (s)':<40} {'N/A':>10}  {sur_screen_avg:>10.2f}")
print(f"  {'Avg top-K IndoBERT eval time/gen (s)':<40} {avg_pure_per_gen:>10.1f}  {topk_eval_avg:>10.1f}")
print(f"  {'Candidates screened per gen':<40} {POPULATION_SIZE:>10}  {SURROGATE_POOL:>10}")
print(f"  {'Candidates fully evaluated per gen':<40} {POPULATION_SIZE:>10}  {SURROGATE_TOPK:>10}")

# ── Tabel 3: Timing Breakdown ──
print()
print(hr())
print("  [TABLE 3] Timing Breakdown (seconds)")
print(hr())
row("Data preprocessing (Spark parallel):", f"{t_data_par:.2f}s")
row("GA warm-up phase (pure eval, GPU):", f"{t_warmup:.2f}s")
row("Surrogate training total:", f"{sum([t_pure_per_gen[0] if t_pure_per_gen else 0]):.2f}s (first gen data)")
row("Surrogate screening total (Spark MAP):", f"{sum(t_surrogate_screen):.2f}s ({len(t_surrogate_screen)} gens)")
row("Top-K IndoBERT eval total (Spark MAP):", f"{sum(t_topk_eval):.2f}s ({len(t_topk_eval)} gens)")
row("Final training (3 epochs, full data):", f"{t_final:.2f}s")
row("─" * 43 + "─────────────", "")
row("TOTAL wall-clock time:", f"{t_data_par + ga_total + t_final:.2f}s  ({(t_data_par + ga_total + t_final)/60:.1f} min)")

# ── Tabel 4: Model Performance ──
print()
print(hr())
print("  [TABLE 4] Final Model Performance (Test Set)")
print(hr())
row("Best Hyperparameters:", "")
for k, v in best_hp.items():
    row(f"  {k}:", str(v))
print(hr("·"))
row("F1-Score (weighted):", f"{final_f1:.4f}")
row("Accuracy:", f"{final_acc:.4f}")
row("Precision (weighted):", f"{final_prec:.4f}")
row("Recall (weighted):", f"{final_rec:.4f}")

# ── Tabel 5: GA Surrogate Data Quality ──
print()
print(hr())
print("  [TABLE 5] Surrogate Model & Search Summary")
print(hr())
row("Total surrogate training samples:", str(len(surrogate_data)))
row("Unique HP configs evaluated (GPU):", str(len(EVAL_CACHE)))
row("Total candidates screened (surrogate):", str(SURROGATE_POOL * len(t_surrogate_screen)))
row("GPU evaluation reduction:", f"{100*(1 - n_gpu_evals_sur/max(n_gpu_evals_pure*GENERATIONS,1)):.1f}% fewer GPU calls")
row("Best fitness (proxy, 30% subset):", f"{best_candidates[0]['fitness']:.4f}")
row("Final F1 (full data, 3 epochs):", f"{final_f1:.4f}")

print()
print("═" * W)
print("  ✅ EKSPERIMEN SELESAI")
print("═" * W)
print()

# %%
# ─── 13. SAVE RESULTS ────────────────────────────────────────────────────────
results_data = {
    "experiment"  : "Surrogate-Assisted Parallel GA",
    "config"      : {
        "population_size"   : POPULATION_SIZE,
        "generations"       : GENERATIONS,
        "warmup_gens"       : WARMUP_GENS,
        "surrogate_pool"    : SURROGATE_POOL,
        "surrogate_topk"    : SURROGATE_TOPK,
        "num_slices"        : NUM_SLICES,
    },
    "timing"      : {
        "data_preprocessing_s"      : round(t_data_par, 2),
        "ga_total_s"                : round(ga_total, 2),
        "estimated_pure_ga_s"       : round(estimated_pure_ga_time, 2),
        "surrogate_speedup_x"       : round(speedup_ga, 3),
        "final_training_s"          : round(t_final, 2),
        "total_wall_clock_s"        : round(t_data_par + ga_total + t_final, 2),
    },
    "mapreduce"   : {
        "data_map_reduce_s"         : round(t_data_par, 2),
        "surrogate_map_per_gen_s"   : round(sur_screen_avg, 3),
        "indobert_map_warmup_s"     : round(avg_pure_per_gen, 2),
        "indobert_map_topk_s"       : round(topk_eval_avg, 2),
    },
    "gpu_eval"    : {
        "warmup_gpu_evals"          : n_gpu_evals_pure,
        "surrogate_gpu_evals"       : n_gpu_evals_sur,
        "total_gpu_evals"           : n_gpu_evals_pure + n_gpu_evals_sur,
    },
    "model"       : {
        "best_hp"                   : best_hp,
        "best_proxy_fitness"        : round(best_candidates[0]["fitness"], 4),
        "final_f1_weighted"         : round(final_f1, 4),
        "final_accuracy"            : round(final_acc, 4),
        "final_precision_weighted"  : round(final_prec, 4),
        "final_recall_weighted"     : round(final_rec, 4),
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

with open("parallel_surrogate_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("✅ Results saved to parallel_surrogate_results.json")
