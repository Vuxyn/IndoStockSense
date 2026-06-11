# %% [markdown]
# # Parallel Genetic Algorithm-Based Hyperparameter Optimization for IndoBERT-LoRA
#
# Indonesian Stock News Sentiment Analysis
#
# This notebook is designed for the Parallel Processing course demo. It keeps
# the full code visible for presentation while using modular functions that can
# later be moved into a Python package for dashboard and Telegram deployment.

# %% [markdown]
# ## 1. Environment Setup
#
# Run this cell in Google Colab.

# %%
!pip uninstall -y torchvision
!pip install -q pyspark transformers "datasets>=2.20.0" accelerate evaluate peft "torchao>=0.16.0" scikit-learn pandas numpy matplotlib seaborn tqdm

# %%
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 2. CUDA and Spark Check

# %%
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"

print("CUDA available:", CUDA_AVAILABLE)
print("Device:", DEVICE)

# %%
try:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("IndoStockSense-Pempar")
        .master("local[2]")
        .getOrCreate()
    )
    sc = spark.sparkContext
    print("Spark version:", spark.version)
except Exception as exc:
    spark = None
    sc = None
    print("Spark is not ready:", exc)

# %% [markdown]
# ## 3. Global Configuration

# %%
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 64
OUTPUT_DIR = "outputs/indobert-lora-stock-sentiment"

POPULATION_SIZE = 20
GENERATIONS = 5
NUM_SLICES = 2

LABEL_MAP = {
    "negatif": 0,
    "netral": 1,
    "positif": 2,
}

ID_TO_LABEL = {value: key for key, value in LABEL_MAP.items()}

# %% [markdown]
# ## 4. Dataset Loading
#
# Dataset utama:
#
# - `data/raw/Dataset-CNBCI-Sentimented.csv`
#
# Expected columns:
#
# - `judul`
# - `tanggal`
# - `sentimen`

# %%
DATA_PATH = Path("data/raw/Dataset-CNBCI-Sentimented.csv")

if not DATA_PATH.exists():
    !mkdir -p data/raw
    !wget -O data/raw/Dataset-CNBCI-Sentimented.csv \
      "https://raw.githubusercontent.com/Vuxyn/IndoStockSense/main/data/raw/Dataset-CNBCI-Sentimented.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# %%
df = df.rename(
    columns={
        "judul": "text",
        "sentimen": "label",
        "tanggal": "date",
    }
)

df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[df["label"].isin(LABEL_MAP)].copy()
df["label_id"] = df["label"].map(LABEL_MAP)

print("Clean dataset shape:", df.shape)
print(df["label"].value_counts())
df.head()

# %% [markdown]
# ## 5. Data Processing Benchmark (Sequential vs PySpark)
#
# Di bagian ini, kita melakukan proses pembersihan teks, filter data, 
# reduksi (menghitung total kata), dan query (distribusi sentimen).
# Kita akan membandingkan waktu eksekusinya antara Pandas (Sequential) dan PySpark (Parallel).

# %%
def clean_text(text: str) -> str:
    """Clean Indonesian stock news text before tokenization."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# %%
print("--- PARALLEL DATA PROCESSING (PYSPARK) ---")
time_par = None

if sc is not None:
    start_par = time.time()
    
    data_list = df[["text", "label", "label_id"]].to_dict("records")
    
    rdd = sc.parallelize(data_list, numSlices=NUM_SLICES)
    
    rdd_mapped = rdd.map(lambda row: {
        "text": row["text"], 
        "label": row["label"], 
        "label_id": row["label_id"],
        "clean_text": clean_text(row["text"])
    })
    
    rdd_filtered = rdd_mapped.filter(lambda row: len(row["clean_text"].split()) >= 3)
    
    rdd_filtered.cache()
    
    total_words_par = rdd_filtered.map(lambda row: len(row["clean_text"].split())).reduce(lambda a, b: a + b)
    
    spark_df = spark.createDataFrame(rdd_filtered)
    spark_df.createOrReplaceTempView("stock_news")
    
    query_result = spark.sql("""
        SELECT label, COUNT(*) as count 
        FROM stock_news 
        GROUP BY label 
        ORDER BY count DESC
    """)
    sentiment_dist_par = {row["label"]: row["count"] for row in query_result.collect()}
    
    processed_df = spark_df.toPandas()
    
    time_par = time.time() - start_par
    
    print(f"Total Words: {total_words_par}")
    print(f"Sentiment Distribution: {sentiment_dist_par}")
    print(f"Parallel Processing Time: {time_par:.4f} seconds")
    
    df = processed_df
else:
    print("SparkContext is not available.")

df[["text", "clean_text", "label"]].head()

# %% [markdown]
# ## 6. Train/Validation/Test Split

# %%
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df["label_id"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=temp_df["label_id"],
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)
print("Train label distribution:")
print(train_df["label"].value_counts())

# %% [markdown]
# ## 7. IndoBERT-LoRA Dataset Preparation

# %%
def build_hf_datasets(train_df, val_df, test_df):
    from datasets import Dataset, DatasetDict

    def to_hf_dataset(frame):
        return Dataset.from_pandas(
            frame[["clean_text", "label_id"]].rename(
                columns={"clean_text": "text", "label_id": "labels"}
            ),
            preserve_index=False,
        )

    return DatasetDict(
        {
            "train": to_hf_dataset(train_df),
            "validation": to_hf_dataset(val_df),
            "test": to_hf_dataset(test_df),
        }
    )


def tokenize_datasets(dataset_dict, tokenizer, max_length: int = 128):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset_dict.map(tokenize_batch, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


hf_datasets = build_hf_datasets(train_df, val_df, test_df)
print(hf_datasets)

# %% [markdown]
# ## 8. IndoBERT-LoRA Training Function

# %%
GA_FIXED_EPOCHS = 1

from transformers import AutoTokenizer
global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
global_dataset_dict = build_hf_datasets(train_df, val_df, test_df)
global_tokenized = tokenize_datasets(global_dataset_dict, global_tokenizer, MAX_LENGTH)

def build_stratified_subset(tokenized_dataset, hf_dataset_with_labels, fraction: float):
    """Create a stratified subset preserving class distribution."""
    import pandas as pd
    labels = hf_dataset_with_labels["labels"] 
    df_idx = pd.DataFrame({"idx": range(len(labels)), "label": labels})
    sampled = df_idx.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=RANDOM_SEED)
    )
    return tokenized_dataset.select(sampled["idx"].tolist())


def train_and_evaluate_indobert_lora(hyperparameters: dict, use_subset: bool = False):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if use_subset:
   
        train_dataset = build_stratified_subset(
            global_tokenized["train"], global_dataset_dict["train"], fraction=0.3
        )
        val_dataset = build_stratified_subset(
            global_tokenized["validation"], global_dataset_dict["validation"], fraction=0.3
        )
        num_epochs = GA_FIXED_EPOCHS
    else:
        train_dataset = global_tokenized["train"]
        val_dataset = global_tokenized["validation"]
        num_epochs = int(hyperparameters["epochs"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(hyperparameters["lora_rank"]),
        lora_alpha=int(hyperparameters["lora_alpha"]),
        lora_dropout=float(hyperparameters["lora_dropout"]),
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=float(hyperparameters["learning_rate"]),
        per_device_train_batch_size=int(hyperparameters["batch_size"]),
        per_device_eval_batch_size=int(hyperparameters["batch_size"]),
        num_train_epochs=num_epochs,
        weight_decay=float(hyperparameters["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_SEED,
        fp16=CUDA_AVAILABLE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=global_tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    metrics = trainer.evaluate(global_tokenized["validation"])
    elapsed = time.time() - start_time

    model.save_pretrained(OUTPUT_DIR)
    global_tokenizer.save_pretrained(OUTPUT_DIR)

    return {
        "individual": hyperparameters,
        "fitness": float(metrics["eval_f1_weighted"]),
        "elapsed": elapsed,
        "metrics": metrics,
    }


print("IndoBERT-LoRA training function ready.")

# %% [markdown]
# ## 9. Genetic Algorithm Search Space

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


def create_individual(space: SearchSpace) -> dict:
    return {
        "learning_rate": random.choice(space.learning_rates),
        "batch_size": random.choice(space.batch_sizes),
        "lora_rank": random.choice(space.lora_ranks),
        "lora_alpha": random.choice(space.lora_alphas),
        "lora_dropout": random.choice(space.lora_dropouts),
        "weight_decay": random.choice(space.weight_decays),
    }


def create_population(size: int, space: SearchSpace) -> list[dict]:
    return [create_individual(space) for _ in range(size)]


def crossover(parent1: dict, parent2: dict) -> dict:
    """Combine genes from two parents."""
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child


def mutate(individual: dict, space: SearchSpace, mutation_rate: float = 0.2) -> dict:
    """Randomly change some genes to maintain diversity."""
    mutated = individual.copy()
    for key in mutated:
        if random.random() < mutation_rate:
            if key == "learning_rate": mutated[key] = random.choice(space.learning_rates)
            elif key == "batch_size": mutated[key] = random.choice(space.batch_sizes)
            elif key == "lora_rank": mutated[key] = random.choice(space.lora_ranks)
            elif key == "lora_alpha": mutated[key] = random.choice(space.lora_alphas)
            elif key == "lora_dropout": mutated[key] = random.choice(space.lora_dropouts)
            elif key == "weight_decay": mutated[key] = random.choice(space.weight_decays)
    return mutated


def select_parents(evaluated_population: list[dict]) -> list[dict]:
    """Select the top 50% best performing individuals."""
    sorted_pop = sorted(evaluated_population, key=lambda x: x["fitness"], reverse=True)
    parents = [x["individual"] for x in sorted_pop[:len(sorted_pop)//2]]
    return parents


def next_generation(evaluated_population: list[dict], space: SearchSpace, pop_size: int) -> list[dict]:
    """Create the next generation using selection, crossover, and mutation."""
    parents = select_parents(evaluated_population)
    next_pop = []
    
    best_individual = max(evaluated_population, key=lambda x: x["fitness"])["individual"]
    next_pop.append(best_individual)
    
    while len(next_pop) < pop_size:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        child = crossover(p1, p2)
        child = mutate(child, space)
        next_pop.append(child)
        
    return next_pop


search_space = SearchSpace()
population = create_population(POPULATION_SIZE, search_space)
population

# %% [markdown]
# ## 10. Fitness Function & Caching
#
# Two separate caches are used:
# - SEQ_CACHE: for sequential GA evaluation
# - PAR_CACHE: for parallel GA evaluation
# This ensures speedup measurement is fair and unaffected by cross-contamination.

# %%
SEQ_CACHE = {}
PAR_CACHE = {}

def get_cache_key(individual: dict) -> str:
    return str(sorted(individual.items()))

def evaluate_individual(individual: dict, use_subset: bool = True) -> dict:
    """Evaluate one chromosome and return its fitness score using IndoBERT-LoRA."""
    try:
        return train_and_evaluate_indobert_lora(individual, use_subset=use_subset)
    except Exception as e:
        import traceback
        return {
            "individual": individual, 
            "fitness": -1.0, 
            "elapsed": 0.0, 
            "metrics": {},
            "error": traceback.format_exc()
        }

# %% [markdown]
# ## 11. Parallel GA Evaluation with PySpark

# %%
def evaluate_population_parallel(sc, population: list[dict], num_slices: int = 4) -> list[dict]:
    unique_to_evaluate = []

    for ind in population:
        key = get_cache_key(ind)
        if key not in PAR_CACHE and ind not in unique_to_evaluate:
            unique_to_evaluate.append(ind)

    if unique_to_evaluate:
        pop_rdd = sc.parallelize(unique_to_evaluate, numSlices=num_slices)
        new_results = pop_rdd.map(lambda ind: evaluate_individual(ind, use_subset=True)).collect()

        for res in new_results:
            if "error" in res:
                print(f"\n[PYSPARK WORKER ERROR for {res['individual']}]:\n{res['error']}\n")
            PAR_CACHE[get_cache_key(res["individual"])] = res

    cache_hits = len(population) - len(unique_to_evaluate)
    print(f"  [Cache] {cache_hits}/{len(population)} cache hits "
          f"({100*cache_hits/len(population):.0f}% saved) | "
          f"PAR_CACHE size: {len(PAR_CACHE)}")

    final_results = []
    for ind in population:
        key = get_cache_key(ind)
        cached_result = PAR_CACHE[key].copy()
        if ind not in unique_to_evaluate:
            cached_result["elapsed"] = 0.0
        final_results.append(cached_result)

    return final_results


if sc is not None:
    start = time.time()
    current_population_par = population.copy()
    best_parallel = None
    
    print("Starting Parallel GA...")
    for gen in range(GENERATIONS):
        print(f"--- Generation {gen + 1}/{GENERATIONS} ---")
        
        evaluated = evaluate_population_parallel(sc, current_population_par, NUM_SLICES)
        
        gen_best = max(evaluated, key=lambda item: item["fitness"])
        print(f"Best fitness this gen: {gen_best['fitness']:.4f}")
        if best_parallel is None or gen_best["fitness"] > best_parallel["fitness"]:
            best_parallel = gen_best
            
        if gen < GENERATIONS - 1:
            current_population_par = next_generation(evaluated, search_space, POPULATION_SIZE)

    parallel_time = time.time() - start

    print(f"Parallel total time: {parallel_time:.2f}s")
    print("Overall best parallel result:", best_parallel)
else:
    parallel_time = None
    best_parallel = None
    print("SparkContext is not available.")

# %% [markdown]
# ## 12. Final IndoBERT-LoRA Training & Top-3 Validation
#
# Top-3 hyperparameters from GA are trained on full data to validate
# that proxy fitness (30% subset, 1 epoch) rankings hold on full data.
# This mitigates the proxy fitness inconsistency risk.

# %%
best_evaluated = best_parallel

active_cache = PAR_CACHE
all_evaluated = sorted(active_cache.values(), key=lambda x: x["fitness"], reverse=True)

top3_candidates = []
seen_keys = set()
for res in all_evaluated:
    k = get_cache_key(res["individual"])
    if k not in seen_keys:
        top3_candidates.append(res)
        seen_keys.add(k)
    if len(top3_candidates) == 3:
        break

print("=" * 60)
print("TOP-3 VALIDATION: Training on 100% data (epochs=3)")
print("=" * 60)

FINAL_EPOCHS = 3

top3_full_results = []
for rank, candidate in enumerate(top3_candidates, start=1):
    hp = candidate["individual"].copy()
    hp["epochs"] = FINAL_EPOCHS
    print(f"\n[Rank #{rank} by proxy fitness={candidate['fitness']:.4f}] Training on full data...")
    result = train_and_evaluate_indobert_lora(hp, use_subset=False)
    result["ga_proxy_rank"] = rank
    top3_full_results.append(result)
    print(f"  Full-data F1: {result['fitness']:.4f}")

top3_full_results_sorted = sorted(top3_full_results, key=lambda x: x["fitness"], reverse=True)
best_hyperparameters = top3_full_results_sorted[0]["individual"]
final_training_result = top3_full_results_sorted[0]

print("\n--- Proxy Rank vs Full-Data Rank ---")
for full_rank, res in enumerate(top3_full_results_sorted, start=1):
    print(f"  Full Rank #{full_rank} | Proxy Rank #{res['ga_proxy_rank']} "
          f"| Full F1={res['fitness']:.4f}")

if top3_full_results_sorted[0]["ga_proxy_rank"] == 1:
    print("\n✅ Proxy fitness ranking VALIDATED: Rank #1 from GA is also best on full data.")
else:
    print("\n⚠️  Proxy fitness ranking SHIFTED: A different candidate won on full data.")

print("\nBest hyperparameters (after top-3 validation):", best_hyperparameters)

default_hyperparameters = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": FINAL_EPOCHS,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "weight_decay": 0.01,
}
print("\nBaseline hyperparameters (Default):", default_hyperparameters)
baseline_training_result = train_and_evaluate_indobert_lora(default_hyperparameters, use_subset=False)
print("Baseline training result:", baseline_training_result)

# %% [markdown]
# ## 13. Evaluation Metrics

# %%
evaluation_summary = {
    "accuracy": final_training_result["metrics"].get("eval_accuracy", 0.0),
    "precision_weighted": final_training_result["metrics"].get("eval_precision_weighted", 0.0),
    "recall_weighted": final_training_result["metrics"].get("eval_recall_weighted", 0.0),
    "f1_weighted": final_training_result["fitness"],
    "baseline_f1_weighted": baseline_training_result["fitness"],
    "parallel_time": parallel_time,
}

print("Evaluation Summary:")
print(evaluation_summary)

# %% [markdown]
# ## 14. Sentiment Inference Demo

# %%
def predict_sentiment(text: str) -> dict:
    """Placeholder sentiment inference function for demo flow."""
    cleaned = clean_text(text)
    keywords_positive = {"menguat", "naik", "laba", "positif", "tumbuh"}
    keywords_negative = {"melemah", "turun", "rugi", "tekanan", "negatif"}

    tokens = set(cleaned.split())
    if tokens & keywords_positive:
        label = "positif"
        confidence = 0.82
    elif tokens & keywords_negative:
        label = "negatif"
        confidence = 0.78
    else:
        label = "netral"
        confidence = 0.65

    return {
        "text": text,
        "clean_text": cleaned,
        "label": label,
        "confidence": confidence,
    }


predict_sentiment("Saham BBCA menguat setelah laporan laba bersih tumbuh positif.")

# %% [markdown]
# ## 16. Future Deployment Simulation

# %%
def format_telegram_message(prediction: dict) -> str:
    return (
        "IndoStockSense Alert\n"
        f"Sentiment: {prediction['label']}\n"
        f"Confidence: {prediction['confidence']:.2%}\n"
        f"News: {prediction['text']}"
    )


telegram_preview = format_telegram_message(
    predict_sentiment("IHSG melemah karena tekanan jual investor asing.")
)

print(telegram_preview)




# %% [markdown]
# ## 17. Stock Intelligence Features
#
# These features show how the sentiment model can become a small stock news
# intelligence system after the course project: ticker detection, daily sentiment
# aggregation, and briefing generation.

# %%
IDX_TICKERS = {
    "BBCA": "Bank Central Asia",
    "BBRI": "Bank Rakyat Indonesia",
    "BMRI": "Bank Mandiri",
    "TLKM": "Telkom Indonesia",
    "ASII": "Astra International",
    "GOTO": "GoTo Gojek Tokopedia",
    "UNVR": "Unilever Indonesia",
    "ANTM": "Aneka Tambang",
    "PGAS": "Perusahaan Gas Negara",
    "ADRO": "Adaro Energy",
    "MDKA": "Merdeka Copper Gold",
    "BRIS": "Bank Syariah Indonesia",
}


def detect_tickers(text: str, ticker_map: dict = IDX_TICKERS) -> list[str]:
    upper_text = str(text).upper()
    found = []
    for ticker in ticker_map:
        if re.search(rf"\b{ticker}\b", upper_text):
            found.append(ticker)
    return found


def sentiment_to_signal(label: str, confidence: float) -> str:
    if label == "positif" and confidence >= 0.75:
        return "WATCH"
    if label == "negatif" and confidence >= 0.75:
        return "RISK ALERT"
    return "MONITOR"


sample_news = [
    "Saham BBCA menguat setelah laba bersih tumbuh positif sepanjang kuartal ini.",
    "BBRI dan BMRI bergerak netral menunggu keputusan suku bunga BI.",
    "GOTO melemah karena tekanan jual investor asing meningkat.",
]

feature_rows = []
for news in sample_news:
    prediction = predict_sentiment(news)
    tickers = detect_tickers(news)
    feature_rows.append(
        {
            "news": news,
            "tickers": tickers,
            "sentiment": prediction["label"],
            "confidence": prediction["confidence"],
            "signal": sentiment_to_signal(prediction["label"], prediction["confidence"]),
        }
    )

features_df = pd.DataFrame(feature_rows)
features_df

# %% [markdown]
# ## 18. Daily Market Briefing Simulation

# %%
def build_market_briefing(news_items: list[str]) -> str:
    rows = []
    for news in news_items:
        prediction = predict_sentiment(news)
        tickers = detect_tickers(news)
        rows.append(
            {
                "news": news,
                "tickers": tickers,
                "sentiment": prediction["label"],
                "confidence": prediction["confidence"],
                "signal": sentiment_to_signal(prediction["label"], prediction["confidence"]),
            }
        )

    briefing_df = pd.DataFrame(rows)
    sentiment_counts = briefing_df["sentiment"].value_counts().to_dict()

    ticker_summary = {}
    for _, row in briefing_df.iterrows():
        for ticker in row["tickers"]:
            ticker_summary.setdefault(ticker, []).append(row["sentiment"])

    lines = [
        "IndoStockSense Daily Briefing",
        f"Total news analyzed: {len(news_items)}",
        "",
        "Market sentiment:",
    ]

    for label in ["positif", "netral", "negatif"]:
        lines.append(f"- {label}: {sentiment_counts.get(label, 0)}")

    lines.extend(["", "Ticker highlights:"])
    if ticker_summary:
        for ticker, sentiments in ticker_summary.items():
            dominant = pd.Series(sentiments).mode().iloc[0]
            lines.append(f"- {ticker}: dominant sentiment {dominant}")
    else:
        lines.append("- No ticker detected")

    return "\n".join(lines)


print(build_market_briefing(sample_news))

# %% [markdown]
# ## 19. Telegram Message Preview

# %%
print("Telegram preview executed successfully.")

# %% [markdown]
# ## 20. Paper Data Summary
#
# Seluruh data yang diekstrak untuk penulisan paper ilmiah Anda:

# %%
import json

paper_data = {
    "1. Hardware & Environment": {
        "CUDA Available": CUDA_AVAILABLE,
        "Device": DEVICE,
        "PySpark Version": spark.version if spark is not None else "N/A"
    },
    "2. Dataset Information": {
        "Total Cleaned Rows": int(df.shape[0]),
        "Train Size": int(train_df.shape[0]),
        "Validation Size": int(val_df.shape[0]),
        "Test Size": int(test_df.shape[0]),
        "Label Distribution (Full)": df["label"].value_counts().to_dict(),
    },
    "3. Data Processing Performance": {
        "Sequential Time (s)": round(time_seq, 4) if 'time_seq' in locals() else None,
        "Parallel Time (s)": round(time_par, 4) if 'time_par' in locals() and time_par is not None else None,
        "Speedup (x)": round(data_speedup, 2) if 'data_speedup' in locals() else None,
    },
    "4. Genetic Algorithm Config": {
        "Population Size": POPULATION_SIZE,
        "Generations": GENERATIONS,
    },
    "5. Genetic Algorithm Speedup": {
        "Sequential GA Time (s)": round(sequential_time, 2) if 'sequential_time' in locals() else None,
        "Parallel GA Time (s)": round(parallel_time, 2) if 'parallel_time' in locals() and parallel_time is not None else None,
        "Speedup (x)": round(speedup, 2) if 'speedup' in locals() and speedup is not None else None,
    },
    "6. Model Performance": {
        "Best Hyperparameters (GA)": best_hyperparameters,
        "Baseline F1-Score (Default)": round(baseline_training_result["fitness"], 4),
        "Optimized F1-Score (GA)": round(final_training_result["fitness"], 4),
        "F1-Score Improvement": round(final_training_result["fitness"] - baseline_training_result["fitness"], 4),
    }
}

print("="*60)
print("DATA SUMMARY UNTUK PAPER ILMIAH")
print("="*60)
print(json.dumps(paper_data, indent=4))

# %%
def format_stock_alert(news: str) -> str:
    prediction = predict_sentiment(news)
    tickers = detect_tickers(news)
    signal = sentiment_to_signal(prediction["label"], prediction["confidence"])
    ticker_text = ", ".join(tickers) if tickers else "Not detected"

    return (
        "IndoStockSense Alert\n"
        f"Tickers: {ticker_text}\n"
        f"Sentiment: {prediction['label']}\n"
        f"Confidence: {prediction['confidence']:.2%}\n"
        f"Experimental Signal: {signal}\n"
        f"News: {news}"
    )


print(format_stock_alert("Saham BBCA menguat setelah laporan laba bersih tumbuh positif."))



