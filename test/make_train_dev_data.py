#-*-coding:utf-8-*-

"""
Make train and dev data using vocab, morphemed corpus file.
"""


import os
from tokenizer import SentencePieceTokenizer


def tokenizing(file, save_path, tokenizer_, mode="train"):
    docs = []
    with open(os.path.join(os.path.abspath(""), file), "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")

    with open(os.path.join(os.path.abspath(""), save_path, mode +".txt"), "w", encoding="utf-8") as f:
        for doc in docs:
            sentences = doc.strip().split("\t")
            
            tokenized_sents = []
            for sent in sentences:
                tokenized_sents.append(" ".join(tokenizer_.tokenize(sent)))

            f.write("\t".join(tokenized_sents))
            f.write("\n\n")


def main(train_file, dev_file, save_path, vocab_path):
    tokenizer_ = SentencePieceTokenizer(
        model_prefix=os.path.join(os.path.abspath(""), vocab_path, "vocab")
    )
    tokenizer_.load()

    tokenizing(train_file, save_path, tokenizer_, mode="train")   
    tokenizing(dev_file, save_path, tokenizer_, mode="dev")   


if __name__=="__main__":
    train_file = "test/data/train_morph.txt"
    dev_file = "test/data/dev_morph.txt"
    save_path = "test/data/"
    vocab_path = "test/vocab"
    main(train_file, dev_file, save_path, vocab_path)
