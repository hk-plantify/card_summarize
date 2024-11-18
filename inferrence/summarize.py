import torch
from transformers import T5Tokenizer
from models.KET5 import T5ConditionalGeneration
from summarize.utils import generate_summary

model_wrapper = T5ConditionalGeneration() 
model_wrapper.load_state_dict(torch.load('.tmp/best_model.pt'))
model_wrapper.eval()

tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-small-ko")

sample_text = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

summary = generate_summary(model_wrapper, tokenizer, sample_text, device, max_len=1024)
print("생성된 요약문:", summary)