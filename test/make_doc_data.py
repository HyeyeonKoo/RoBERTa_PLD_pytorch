#-*-coding:utf-8-*-

"""
KorQuAD_v1.0 is used for test RoBERTa.
Save each file per each context.
Separate each sentence with "\t" and each morpheme with " "
"""


import json
import os
import kss
from konlpy.tag import Mecab
from datetime import datetime
from tqdm import tqdm


m = Mecab()


def get_json(json_file):
    with open(json_file) as f:
        result = json.loads(f.read())
    return result


def get_text(document):
    split_sentences = kss.split_sentences(document)
    
    result_string = ""
    for sent in split_sentences:
        result_string += " ".join(m.morphs(sent.strip().replace("\t", " "))) + "\t"

    return result_string


def make_data(data_path, file_name, mode="train_morph"):
    json_dict = get_json(os.path.join(data_path, file_name))
    save_path = os.path.join(os.path.abspath(""), data_path, mode + ".txt")

    with open(save_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(json_dict["data"])), desc=mode):
            for j in range(len(json_dict["data"][i]["paragraphs"])):
                f.write(get_text(json_dict["data"][i]["paragraphs"][j]["context"]))
                f.write("\n\n")

def main(data_path, train_file, dev_file):
    start = datetime.now()

    make_data(data_path, train_file, mode="train_morph")
    make_data(data_path, dev_file, mode="dev_morph")

    end = datetime.now()
    print("Done : " + str(end-start))
        

if __name__=="__main__":
    data_path = "test/data"
    train_file = "KorQuAD_v1.0_train.json"
    dev_file = "KorQuAD_v1.0_dev.json"
    main(data_path, train_file, dev_file)
