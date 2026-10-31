from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"
LOCAL_PATH = "./finbert_model"

print(f"Downloading {MODEL_NAME} to {LOCAL_PATH}...")

# 1. Download tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 2. Save directly to your local folder
tokenizer.save_pretrained(LOCAL_PATH)
model.save_pretrained(LOCAL_PATH)

print("Download complete!")