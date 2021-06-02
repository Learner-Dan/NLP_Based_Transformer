import os
import torch
import pathlib
import transformers
import numpy as np
from transformers import (AlbertForSequenceClassification,
                            AlbertTokenizer,
                            AlbertConfig,
                            Trainer,
                            TrainingArguments,
                            EvalPrediction
                            )
from sklearn.model_selection import train_test_split
import metrics
def gain_accuracy(res : EvalPrediction):
    true_label = res.label_ids
    pre_label = np.argmax(res.predictions,axis=1)
    metric_res = metrics.compute_metric(true_label,pre_label)
    return metric_res

data_path = "./SequenceClassification/aclImdb/"
model_path = "model/checkpoint-67000"
tokenizer_path = "albert-base-v2"

def read_data(data_path):
    if(os.path.exists(data_path) == False):
        raise ValueError("Please check the data path: %s"%data_path)
    path_obj = pathlib.Path(data_path)
    data = []
    labels = []
    for p in path_obj.rglob("*.txt"):
        if p.parts[-2] == "pos":
            labels.append(1)
        elif p.parts[-2] == "neg":
            labels.append(0)
        else:
            continue
        data.append(p.read_text())
    return data,labels

class IMdbDataSet(torch.utils.data.Dataset):
    def __init__(self,data_encode,labels):
        self.data_encode = data_encode
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data_encode.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
train_data,train_label = read_data(os.path.join(data_path,"train"))
# test_data,test_label = read_data(os.path.join(data_path,"test"))
train_data,val_data,train_label,val_label = train_test_split(train_data,train_label,test_size=0.2)

tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)
val_encoding = tokenizer(val_data,padding=True,truncation=True,max_length=500)
val_dataset = IMdbDataSet(val_encoding,val_label)
train_encoding = tokenizer(train_data,padding=True,truncation=True,max_length=500)
train_dataset = IMdbDataSet(train_encoding,train_label)

# test_encoding = tokenizer(test_data,padding=True,truncation=True)
# test_dataset = IMdbDataSet(test_encoding,test_label)

# config = AlbertConfig.from_pretrained(model_path)
# config.num_labels = 2
# model = AlbertForSequenceClassification(config)
model = AlbertForSequenceClassification.from_pretrained(tokenizer_path,num_labels=2)
# model.train()


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    learning_rate=5e-4,
    do_train=True,
    do_eval=True,
    eval_steps=5000,
    evaluation_strategy="steps"
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=gain_accuracy
)


trainer.train()