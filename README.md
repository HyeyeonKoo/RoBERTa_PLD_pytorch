# RoBERTa with PLD 

- I reviewed "Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping" paper at the [Jiphyeonjeon](https://github.com/jiphyeonjeon/nlp-review)
- PLD paper review video : [YouTube](https://www.youtube.com/watch?v=mLyq5JFr-kE&t=1s)
- Paper : [RoBERTa](https://arxiv.org/abs/1907.11692), [PLD](https://arxiv.org/abs/2010.13369) 

*I could not test at diverse environments. And, There can be something wrong in the code. Later, I try to improve these things. If you have some opinions about this code. Please share me.*



### Requirements

- pytorch ==1.7.1

※ I test this code at pytorch 1.7.1. It could possible that this code run on other pytorch version.



### Run

```python
from roberta import RoBERTaWithPLD

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
    train_data_path=os.path.join(os.path.abspath(""), "train.txt"),
    dev_data_path=os.path.join(os.path.abspath(""), "dev.txt")
)
```

- You can get more detail in the test directory.



### Test

- I want to test with paragraph data. So I use [KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html) dataset. The license does not allow to distributing changed this dataset. So I do not open the train, dev dataset I use. However, If you want, you can use these codes.
- make_doc_data.py : From KorQuAD1.0 extract paragraph data.
- make_train_dev_data.py : From paragraph data, make train, dev data.
- make_vocab.py : Make vocab model and dictionary using [sentencepiece](https://github.com/google/sentencepiece) and rebuild own word_index dictionary for RoBERTa train.

