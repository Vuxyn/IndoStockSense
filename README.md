# IndoStockSense

Parallel Genetic Algorithm-Based Hyperparameter Optimization for IndoBERT-LoRA in Indonesian Stock News Sentiment Analysis.

## Project Focus

This project is built first for a Parallel Processing course assignment using Google Colab. The core experiment compares sequential and PySpark-based parallel Genetic Algorithm evaluation for tuning IndoBERT-LoRA hyperparameters.

After the academic version is complete, the same codebase can be extended into a stock news sentiment dashboard and Telegram market briefing bot.

## Main Pipeline

```text
Indonesian stock news dataset
        |
Text preprocessing
        |
Genetic Algorithm hyperparameter search
        |-- Sequential evaluation
        |-- Parallel evaluation with PySpark
        |
IndoBERT-LoRA fine-tuning with CUDA/GPU
        |
Evaluation and benchmark
        |
Sentiment inference demo
```

## Course Assignment Scope

- Indonesian stock news sentiment classification
- IndoBERT-LoRA fine-tuning
- Genetic Algorithm hyperparameter optimization
- Sequential vs parallel GA benchmark
- PySpark `parallelize()`, `map()`, and `collect()`
- CUDA/GPU acceleration in Google Colab
- Accuracy, precision, recall, weighted F1-score, runtime, and speedup evaluation

## Future Development

- Automated web scraping for Indonesian stock news
- FastAPI backend
- Web dashboard
- Telegram bot
- Scheduled daily market briefing
- Experimental buy/sell/hold signal engine

## Planned Repository Structure

```text
notebooks/
  pempar_demo_full.ipynb

src/
  indostocksense/
    preprocessing.py
    ga.py
    spark_parallel.py
    train_lora.py
    inference.py
    scraper.py
    api.py
    telegram_bot.py

requirements-colab.txt
requirements.txt
```
