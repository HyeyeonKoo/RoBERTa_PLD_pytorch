#-*-coding:utf-8-*-

"""
For tokenizing, use google's sentencepiece lib.
"""


import sentencepiece


class SentencePieceTokenizer:
    
    def __init__(self, input_f="input.txt", model_prefix="vocab", vocab_size=32000,
        character_coverage=1.0, model_type="bpe", user_defined_symbols="[CLS],[SEP],[UNK],[PAD]"):
        self.train_param = {
            "input": input_f,
            "model_prefix" : model_prefix,
            "vocab_size": vocab_size,
            "character_coverage": character_coverage,
            "model_type": model_type,
            "user_defined_symbols": user_defined_symbols
        }

        self.model = None


    def train(self):
        command = ("--input=%s --model_prefix=%s --vocab_size=%s --character_coverage=%s --model_type=%s --user_defined_symbols=%s"
            %(self.train_param["input"], self.train_param["model_prefix"], self.train_param["vocab_size"],
            self.train_param["character_coverage"], self.train_param["model_type"], self.train_param["user_defined_symbols"]))

        sentencepiece.SentencePieceTrainer.Train(command)


    def load(self):
        self.model = sentencepiece.SentencePieceProcessor()
        self.model.Load(self.train_param["model_prefix"] + ".model")      

    
    def tokenize(self, sentence):
        return self.model.EncodeAsPieces(sentence)


    def restore(self, segmented_sentence):
        return self.model.DecodePieces(segmented_sentence)


    def encode(self, sentence):
        return self.model.EncodeAsIds(sentence)


    def decode(self, encoded_sentence):
        return self.model.DecodeIds(encoded_sentence)