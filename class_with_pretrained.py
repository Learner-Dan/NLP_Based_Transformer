from datasets import load_dataset
import transformers
from transformers import DataCollatorForLanguageModeling,AutoModelForMaskedLM,Trainer
from compete import tianchi_tokenizer
import torch

# 映射词典
vocab_to_id = {}
vocab_to_str = {}
for i in range(1001):
    vocab_to_id[str(i)] = i
    vocab_to_str[i] = str(i)
# vocab_to_id["<mask>"] = 1001
# vocab_to_str[1001] = "<mask>"

#　分词器
t_tokenizer = tianchi_tokenizer.TTokenizer(vocab_to_id,vocab_to_str)

print(t_tokenizer.get_vocab())
# # 加载dataset
# t_dataset = load_dataset("text",data_files={"train":"./data/train.txt",
#                                         "test":"./data/test.txt",
#                                         "val":"./data/val.txt"},split="train")
# # dataset映射函数
# def d_map(example):
#     example = t_tokenizer(example["text"],padding="max_length",truncation=True,max_length=107,return_special_tokens_mask=True)
#     return example
# t_dataset = t_dataset.map(d_map,remove_columns=["text"],batched=True)


# albert_class = transformers.AlbertForSequenceClassification.from_pretrained("/home/hedan/tools/Github/NLP_Based_Transformer/checkpoint-1000")
# input_test = t_dataset[0:2]
# x = albert_class(torch.tensor(input_test["input_ids"]))
# print(albert_class)
# print("ok")