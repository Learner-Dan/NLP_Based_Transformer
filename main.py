#
# main.py
# @author DanHe
# @description 
# @created 2021-05-08T16:50:40.264Z+08:00
# @last-modified 2021-05-12T19:11:19.687Z+08:00
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


#　分词器
t_tokenizer = tianchi_tokenizer.TTokenizer(vocab_to_id,vocab_to_str)

# 加载dataset
t_dataset = load_dataset("text",data_files={"train":["./data/train.txt","./data/test.txt","./data/val.txt"],
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
albert_config = transformers.AlbertConfig(vocab_size=len(t_tokenizer),embedding_size=128,
        num_hidden_layers=2,num_attention_heads=4,hidden_size=128,intermediate_size=256,
        pad_token_id=t_tokenizer.pad_token_id,
        bos_token_id=t_tokenizer.bos_token_id,
        eos_token_id=t_tokenizer.eos_token,
        sep_token_id=t_tokenizer.sep_token_id)

# 创建Albert语言模型
albert_model = AutoModelForMaskedLM.from_config(albert_config)

# 配置训练参数
train_args = transformers.TrainingArguments(output_dir="./model",do_train=True,
                                            learning_rate=0.001,num_train_epochs=30,
                                            per_device_train_batch_size=32,lr_scheduler_type="cosine_with_restarts")


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