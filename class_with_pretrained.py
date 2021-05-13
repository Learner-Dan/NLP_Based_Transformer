from datasets import load_dataset
import transformers
from transformers import DataCollatorForLanguageModeling,AutoModelForMaskedLM,Trainer,default_data_collator
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
t_dataset = load_dataset("text",data_files={"train":"./data/train.txt",
                                        "test":"./data/test.txt",
                                        "val":"./data/val.txt"},split="train")
t_val_dataset = load_dataset("text",data_files={"train":"./data/train.txt",
                                        "test":"./data/test.txt",
                                        "val":"./data/val.txt"},split="val")
# dataset映射函数
def d_map(example):
    res = t_tokenizer(example["text"],padding="max_length",truncation=True,max_length=107,return_special_tokens_mask=True)
    total_text = example["text"]
    res["labels"] = []
    for text in total_text:
        text = text.split(";")[1]
        text = text.strip()
        split_text = text.split(" ")
        res["labels"].append([float(k) for k in split_text])
    return res
t_dataset = t_dataset.map(d_map,remove_columns=["text"],batched=True)
t_val_dataset =t_val_dataset.map(d_map,remove_columns=["text"],batched=True)

albert_class_model = transformers.AlbertForSequenceClassification.from_pretrained("/home/hedan/tools/Github/NLP_Based_Transformer/model/checkpoint-24000",num_labels=17)

# 配置训练参数
train_args = transformers.TrainingArguments(output_dir="./modelclass",do_train=True,logging_steps=50,do_eval=True,
                                            learning_rate=0.0002,num_train_epochs=60,save_steps=2000,eval_steps=500,
                                            per_device_train_batch_size=32,lr_scheduler_type="polynomial",evaluation_strategy="steps")


# t = t_DataCollator(h["input_ids"])
# x = albert_model(torch.tensor(h["input_ids"]))
# 训练
trainer = Trainer(
        model=albert_class_model,
        args=train_args,
        train_dataset=t_dataset,
        eval_dataset=t_val_dataset,
        
        tokenizer=t_tokenizer,
        data_collator=None,
    )
trainer.train()