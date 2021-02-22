#-*-coding:utf-8-*-

"""
Make vocab using Google's sentencepiece lib.
Also make word_index, index_word json file for encoding and recovering data. 
"""


import os
import json
from tokenizer import SentencePieceTokenizer


def save_word_index(save_path):
    vocab_path = os.path.join(os.path.abspath(""), save_path, "vocab.vocab")

    word_index, index_word = {}, {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        index = 0
        for line in f:
            word = line[:-1].split("\t")[0]
            if word == "<s>" or word == "</s>" or word == "<unk>":
                continue
            
            word_index[word] = index
            index_word[index] = word
            index += 1

    with open(os.path.join(os.path.abspath(""), save_path, "word_index.json"), "w") as f:
        json.dump(word_index, f, ensure_ascii=False)
    with open(os.path.join(os.path.abspath(""), save_path, "index_word.json"), "w") as f:
        json.dump(index_word, f, ensure_ascii=False)


def main(input_file, save_path):
    tokenizer_ = SentencePieceTokenizer(
        input_f=os.path.join(os.path.abspath(""), input_file),
        model_prefix=os.path.join(os.path.abspath(""), save_path, "vocab"),
        vocab_size=50000,
        user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[BOD],[EOD],[MASK]"
    )
    tokenizer_.train()
    save_word_index(save_path)


if __name__=="__main__":
    input_file = "test/data/train_morph.txt"
    save_path = "test/vocab"
    main(input_file, save_path)