#-*-coding:utf-8-*-

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
import math


"""
RoBERTa Dataset class.
According to RoBERTa paper, There are some rules for line by line data.
    1. Use full-sentences. It means that link several sentences with [SEP] token, not just two sentences.
        If document is ended, link with other special token. In this case, use [BOD], [EOD].
    2. No NSP. So, there are no NLP label.
    3. Dynamic Masking. Masked data will be changed every epoch.

* make_each_line(data_path, tqdm_desc)
    Read corpus and tokenizing and make line by line data for training by rule number 1.
    - data path : Corpus's path. Corpus has to be tokenized using vocab. 
        Tokens are seperated by " " and documents are seperated by "\n\n".
    - tqdm_desc : For checking the process, you can input progress bar's description.

*  masking_data(mlm, mask, replace, unchange, tqdm_desc)
    - mlm : Effecting ratio.
    - mask : Masking ratio.
    - replace : Replacing ratio.
    - unchange : Unchanging ratio.

* len
    - Return whole data's length.

* __getitem__(index)
    - Return training data according to input index. 
        Result is comprised by encoded masked data, encoded MLM label, segment label, position label.
"""
class RoBERTaDataset(Dataset):

    def __init__(self, data_path, vocab, max_len, tqdm_desc="dataset"):
        self.vocab = vocab
        self.max_len = max_len

        self.data = self.make_each_line(data_path, tqdm_desc)
        self.masked_data = None
        self.mlm_correct = None


    def make_each_line(self, data_path, tqdm_desc):

        def read_corpus(data_path):
            corpus = []

            with open(data_path, "r", encoding="utf-8") as f:
                docs = f.read().split("\n\n")
                for doc in docs:
                    sentences = doc.strip().split("\t")
                    corpus.append([sent.split(" ") for sent in sentences])

            return corpus

        def groupping_lines(tokenized_data, tqdm_desc):
            input_lines = []

            i = 0
            one_line = []
            bar = tqdm(total=len(tokenized_data), desc=tqdm_desc)
            while i < len(tokenized_data):
                j = 0
                while j < len(tokenized_data[i]):
                    if j == 0:
                        one_line.append("[BOD]")
                    
                    # [CLS], [SEP], [EOD]
                    if len(one_line) + len(tokenized_data[i][j]) + 3 < self.max_len:
                        one_line.append("[CLS]")
                        one_line += tokenized_data[i][j]
                        one_line.append("[SEP]")
                        j += 1
                    
                    else:
                        # 0: [], 1: ["[BOD]"]
                        if len(one_line) < 2:
                            one_line += (["[CLS]"] + tokenized_data[i][j])
                            one_line = one_line[:self.max_len]
                            j += 1
                        else:
                            if one_line[-1] == "[BOD]":
                                one_line[-1] = "[PAD]"
                            one_line += ["[PAD]"] * (self.max_len - len(one_line))
                            
                        input_lines.append(one_line)
                        one_line = []

                if one_line:
                    one_line.append("[EOD]")
                
                i += 1
                bar.update(1)                    

            return input_lines
            
        tokenized_data = read_corpus(data_path)
        input_data = groupping_lines(tokenized_data, tqdm_desc)

        return input_data


    def masking_data(self, mlm=0.15, mask=0.8, replace=0.1, unchange=0.1, tqdm_desc="masking"):
        assert mask + replace + unchange == 1, "sum(mask, replace, unchange) must be 1.0."

        masked_data = []
        for line in tqdm(self.data, desc=tqdm_desc):

            new_line = []
            whole_masking_token_jump = 0
            for token in line:

                if token in ["[CLS]", "[SEP]", "[BOD]", "[EOD]", "[PAD]"]:
                    new_line.append(token)
                    continue

                mlm_ = random.random()
                if mlm_ < mlm:

                    mask_ = random.random()
                    if mask_ < mask:
                        new_line.append("[MASK]")

                    elif mask_ < mask + replace:
                        new_line.append(list(self.vocab.keys())[random.randrange(0, len(self.vocab))])

                    else:
                        new_line.append(token)

                else:
                    new_line.append(token)

            masked_data.append(new_line)

        self.masked_data = masked_data


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, index):
        input_, label, segment = [], [], []
        seg = 1
        for i in range(self.max_len):
            try:
                input_.append(self.vocab[self.masked_data[index][i]])
            except KeyError:
                input_.append(self.vocab["[UNK]"])
            
            try:
                label.append(self.vocab[self.data[index][i]])
            except:
                label.append(self.vocab["[UNK]"])

            if self.masked_data[index][i] == "[PAD]":
                segment.append(self.vocab["[PAD]"])
            else:
                segment.append(seg)
            if self.masked_data[index][i] == "[SEP]":
                seg = 2 if seg==1 else 1

        return {
            "input": torch.tensor(input_), 
            "label": torch.tensor(label),
            "segment": torch.tensor(segment)
        }
