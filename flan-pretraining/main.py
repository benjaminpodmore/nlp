from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer(
    "A step by step guide on how to make cacio e pepe:", return_tensors="pt"
)
outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tkens=True))
