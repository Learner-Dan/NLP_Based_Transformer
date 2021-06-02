from transformers import pipeline

model_path = "model/checkpoint-67000"
wikitext_model = pipeline("fill-mask",model=model_path)

test_str = "next [MASK]"
test_res = wikitext_model(test_str)
print(test_res)