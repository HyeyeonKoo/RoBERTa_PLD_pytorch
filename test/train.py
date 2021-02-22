#-*-coding:utf-8-*-

import os
import sys
import json
sys.path.append(os.path.abspath(""))
from roberta import RoBERTaWithPLD


with open(os.path.join(os.path.abspath(""), "test/vocab/word_index.json"), "r") as f:
    vocab = json.load(f)


# BERT-Base architecture
roberta = RoBERTaWithPLD(
    vocab, max_len=512, num_workers=5,
    embedding_dropout=0.1,
    hidden_layers=12, hidden_size=768, hidden_dropout=0.1, attention_heads=12,
    feed_forward_size=3072,
    learning_rate=5e-5, warmup_step=500,
    adam_ep=6e-4, adam_beta1=0.9, adam_beta2=0.98, weight_decay=0.01,
    batch_size=2, epochs=10000, pld=True, layer_keep_prob=0.5,
    train_verbose_step=500,
    save_epoch=1000, output_path="output"
)

roberta.train(
    train_data_path=os.path.join(os.path.abspath(""), "test/data/train.txt"),
    dev_data_path=os.path.join(os.path.abspath(""), "test/data/dev.txt")
)

