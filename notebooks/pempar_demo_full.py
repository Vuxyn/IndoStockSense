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
# !pip install -q pyspark transformers datasets accelerate evaluate peft scikit-learn pandas numpy matplotlib seaborn tqdm

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
        .master("local[*]")
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
MAX_LENGTH = 128
OUTPUT_DIR = "outputs/indobert-lora-stock-sentiment"

POPULATION_SIZE = 20
GENERATIONS = 10
NUM_SLICES = 4

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
# ## 5. Text Preprocessing

# %%
def clean_text(text: str) -> str:
    """Clean Indonesian stock news text before tokenization."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)
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
def train_and_evaluate_indobert_lora(hyperparameters: dict):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset_dict = build_hf_datasets(train_df, val_df, test_df)

    tokenized = tokenize_datasets(dataset_dict, tokenizer, MAX_LENGTH)

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
        num_train_epochs=int(hyperparameters["epochs"]),
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
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    metrics = trainer.evaluate(tokenized["validation"])
    elapsed = time.time() - start_time

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

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
        "epochs": random.choice(space.epochs),
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
        # 50% chance to inherit from parent1 or parent2
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child


def mutate(individual: dict, space: SearchSpace, mutation_rate: float = 0.2) -> dict:
    """Randomly change some genes to maintain diversity."""
    mutated = individual.copy()
    for key in mutated:
        if random.random() < mutation_rate:
            # Pick a new random value from the search space for this specific key
            if key == "learning_rate": mutated[key] = random.choice(space.learning_rates)
            elif key == "batch_size": mutated[key] = random.choice(space.batch_sizes)
            elif key == "epochs": mutated[key] = random.choice(space.epochs)
            elif key == "lora_rank": mutated[key] = random.choice(space.lora_ranks)
            elif key == "lora_alpha": mutated[key] = random.choice(space.lora_alphas)
            elif key == "lora_dropout": mutated[key] = random.choice(space.lora_dropouts)
            elif key == "weight_decay": mutated[key] = random.choice(space.weight_decays)
    return mutated


def select_parents(evaluated_population: list[dict]) -> list[dict]:
    """Select the top 50% best performing individuals."""
    # Sort by fitness descending
    sorted_pop = sorted(evaluated_population, key=lambda x: x["fitness"], reverse=True)
    # Keep the top half
    parents = [x["individual"] for x in sorted_pop[:len(sorted_pop)//2]]
    return parents


def next_generation(evaluated_population: list[dict], space: SearchSpace, pop_size: int) -> list[dict]:
    """Create the next generation using selection, crossover, and mutation."""
    parents = select_parents(evaluated_population)
    next_pop = []
    
    # Elitism: Automatically keep the absolute best individual
    best_individual = max(evaluated_population, key=lambda x: x["fitness"])["individual"]
    next_pop.append(best_individual)
    
    # Fill the rest of the population with offspring
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
# ## 10. Fitness Function

# %%
def evaluate_individual(individual: dict) -> dict:
    """Evaluate one chromosome and return its fitness score using IndoBERT-LoRA."""
    return train_and_evaluate_indobert_lora(individual)


evaluate_individual(population[0])

# %% [markdown]
# ## 11. Sequential GA Evaluation

# %%
def evaluate_population_sequential(population: list[dict]) -> list[dict]:
    return [evaluate_individual(individual) for individual in population]


start = time.time()
current_population = population.copy()
best_sequential = None

print("Starting Sequential GA...")
for gen in range(GENERATIONS):
    print(f"--- Generation {gen + 1}/{GENERATIONS} ---")
    
    # 1. Evaluate current population
    evaluated = evaluate_population_sequential(current_population)
    
    # 2. Track best individual
    gen_best = max(evaluated, key=lambda item: item["fitness"])
    print(f"Best fitness this gen: {gen_best['fitness']:.4f}")
    if best_sequential is None or gen_best["fitness"] > best_sequential["fitness"]:
        best_sequential = gen_best
        
    # 3. Evolve to next generation
    if gen < GENERATIONS - 1:
        current_population = next_generation(evaluated, search_space, POPULATION_SIZE)

sequential_time = time.time() - start

print(f"Sequential total time: {sequential_time:.2f}s")
print("Overall best sequential result:", best_sequential)

# %% [markdown]
# ## 12. Parallel GA Evaluation with PySpark

# %%
def evaluate_population_parallel(sc, population: list[dict], num_slices: int = 4) -> list[dict]:
    pop_rdd = sc.parallelize(population, numSlices=num_slices)
    fitness_rdd = pop_rdd.map(lambda individual: evaluate_individual(individual))
    return fitness_rdd.collect()


if sc is not None:
    start = time.time()
    current_population_par = population.copy()
    best_parallel = None
    
    print("Starting Parallel GA...")
    for gen in range(GENERATIONS):
        print(f"--- Generation {gen + 1}/{GENERATIONS} ---")
        
        # 1. Evaluate current population (Parallel in PySpark)
        evaluated = evaluate_population_parallel(sc, current_population_par, NUM_SLICES)
        
        # 2. Track best individual
        gen_best = max(evaluated, key=lambda item: item["fitness"])
        print(f"Best fitness this gen: {gen_best['fitness']:.4f}")
        if best_parallel is None or gen_best["fitness"] > best_parallel["fitness"]:
            best_parallel = gen_best
            
        # 3. Evolve to next generation (Sequential evolution, Parallel evaluation)
        if gen < GENERATIONS - 1:
            current_population_par = next_generation(evaluated, search_space, POPULATION_SIZE)

    parallel_time = time.time() - start
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"Parallel total time: {parallel_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print("Overall best parallel result:", best_parallel)
else:
    parallel_time = None
    speedup = None
    best_parallel = None
    print("SparkContext is not available.")

# %% [markdown]
# ## 13. Final IndoBERT-LoRA Training
#
# We take the best hyperparameters found by the GA and train the final model.

# %%
best_hyperparameters = best_parallel["individual"] if sc is not None else best_sequential["individual"]
print("Best hyperparameters:", best_hyperparameters)

final_training_result = train_and_evaluate_indobert_lora(best_hyperparameters)
print("Final training result:", final_training_result)

# %% [markdown]
# ## 14. Evaluation Metrics

# %%
evaluation_summary = {
    "accuracy": 0.0,
    "precision_weighted": 0.0,
    "recall_weighted": 0.0,
    "f1_weighted": final_training_result["fitness"],
    "sequential_time": sequential_time,
    "parallel_time": parallel_time,
    "speedup": speedup,
}

evaluation_summary

# %% [markdown]
# ## 15. Sentiment Inference Demo

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



