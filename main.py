#
# main.py
# @author DanHe
# @description 
# @created 2021-05-08T16:50:40.264Z+08:00
# @last-modified 2021-05-11T15:12:46.713Z+08:00
#

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
vocab_to_id["<mask>"] = 1001
vocab_to_str[1001] = "<mask>"

#　分词器
t_tokenizer = tianchi_tokenizer.TTokenizer(vocab_to_id,vocab_to_str,pad_token="0",mask_token="<mask>")

# 加载dataset
t_dataset = load_dataset("text",data_files={"train":"./data/train.txt",
                                        "test":"./data/test.txt",
                                        "val":"./data/val.txt"},split="train")
# dataset映射函数
def d_map(example):
    example = t_tokenizer(example["text"],padding="max_length",truncation=True,max_length=107,return_special_tokens_mask=True)
    return example
t_dataset = t_dataset.map(d_map,remove_columns=["text"],batched=True)

# 创建针对语言模型的DataCollator
t_DataCollator = DataCollatorForLanguageModeling(t_tokenizer,mlm=True,mlm_probability=0.15)

# 创建Albert模型配置
albert_config = transformers.AlbertConfig(vocab_size=len(t_tokenizer),embedding_size=64,num_hidden_layers=2,num_attention_heads=2,hidden_size=512)

# 创建Albert语言模型
albert_model = AutoModelForMaskedLM.from_config(albert_config)

# 配置训练参数
train_args = transformers.TrainingArguments(output_dir="./",do_train=True,learning_rate=0.01)


# t = t_DataCollator(h["input_ids"])
# x = albert_model(torch.tensor(h["input_ids"]))
# 训练
trainer = Trainer(
        model=albert_model,
        args=train_args,
        train_dataset=t_dataset,
        tokenizer=t_tokenizer,
        data_collator=t_DataCollator,
    )

trainer.train()
print("ok")