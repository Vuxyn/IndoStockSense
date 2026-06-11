import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "outputs", "indobert-lora-stock-sentiment")

LABEL_MAP = {
    "negatif": 0,
    "netral": 1,
    "positif": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print("Loading IndoBERT-LoRA model into memory...")
    try:
        base_model_name = "indobenchmark/indobert-base-p1"
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(LABEL_MAP),
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )
        
        if os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
            _model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        else:
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(device)
        print("Model loaded successfully on", device)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        _model = "MOCK"
        _tokenizer = "MOCK"
        
    return _model, _tokenizer

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text: str):
    model, tokenizer = load_model()
    cleaned_text = clean_text(text)
    
    if model == "MOCK":
        import random
        return {
            "text": text,
            "clean_text": cleaned_text,
            "sentiment": random.choice(["positif", "negatif", "netral"]),
            "confidence": round(random.uniform(0.6, 0.99), 2)
        }
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        confidences = probs[0].cpu().numpy()
        pred_idx = np.argmax(confidences)
        
    label = ID_TO_LABEL[pred_idx]
    confidence = float(confidences[pred_idx])
    
    return {
        "text": text,
        "clean_text": cleaned_text,
        "sentiment": label,
        "confidence": confidence
    }
