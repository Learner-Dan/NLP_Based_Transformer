import transformers
import os
from datasets import load_dataset
import transformers
from transformers import DataCollatorForLanguageModeling,AutoModelForMaskedLM,Trainer,AlbertModel,AlbertForMaskedLM,AlbertTokenizer
from transformers import BertModel,BertConfig
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# 词典路径
dict_path = "./MaskedLanguageModel/VocabModel/vocab.model"

#　分词器
t_tokenizer = AlbertTokenizer.from_pretrained(dict_path)

# 加载dataset
t_dataset = load_dataset("text",data_files={"train":"./data/wikitext-2/train.txt"},split="train")

# dataset映射函数
def d_map(example):
    example = t_tokenizer(example["text"],padding="max_length",truncation=True,max_length=500,return_special_tokens_mask=True)
    return example
t_dataset = t_dataset.map(d_map,remove_columns=["text"],batched=True)

# 创建针对语言模型的DataCollator
t_DataCollator = DataCollatorForLanguageModeling(t_tokenizer,mlm=True,mlm_probability=0.3)


# 创建Albert模型配置
albert_config = transformers.AlbertConfig(vocab_size=len(t_tokenizer),embedding_size=128,
        num_hidden_layers=2,num_attention_heads=4,hidden_size=256,intermediate_size=512,
        pad_token_id=t_tokenizer.pad_token_id,
        bos_token_id=t_tokenizer.bos_token_id,
        eos_token_id=t_tokenizer.eos_token,
        sep_token_id=t_tokenizer.sep_token_id)

# 创建Albert语言模型
albert_model = AutoModelForMaskedLM.from_config(albert_config)
# albert_model = AlbertForMaskedLM.from_pretrained("/home/hedan/tools/Github/NLP_Based_Transformer/model/checkpoint-5000")
# albert_model.resize_token_embeddings(len(t_tokenizer))

# 配置训练参数
train_args = transformers.TrainingArguments(output_dir="./model",do_train=True,logging_steps=50,
                                            learning_rate=0.001,num_train_epochs=30,save_steps=1000,
                                            per_device_train_batch_size=32,lr_scheduler_type="polynomial",dataloader_num_workers=4)


# t = t_DataCollator(h["input_ids"])
# x = albert_model(torch.tensor(h["input_ids"]))
# 训练
trainer = Trainer(
        model=albert_model,
        args=train_args,
        train_dataset=t_dataset,
        tokenizer=t_tokenizer,
        data_collator=t_DataCollator 
    )

trainer.train()
print("ok")