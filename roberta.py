#-*-coding:utf-8-*-

from data import RoBERTaDataset
from model import RoBERTaWithPLDModel
from trainer import RoBERTaWithPLDTrainer
from torch.utils.data import DataLoader
from datetime import datetime
import math


"""
RoBERTa main class.

    __init__() Arguments(Default is like BERT Base)
        - vocab : Vocalbulary file for encoding data.
            It has to be dictionary. {word: index}
            It has to comprise 6 tokens([PAD], [UNK], [CLS], [SEP], [BOD], [EOD], [MASK])
        - max_len : Maximum length of input tokens. 
        - embedding_dropout : Dropout probability for embedding.
        - hidden_layers : Number of transformer blocks to stack.
        - hidden_size : Dimmension of each token.
        - hidden_dropout : Dropout probability for transformer blocks.
        - attention_heads : Number of attention heads for transformer blocks.
        - feed_forward_size : Hidden size for linear calculation in transformer block.
        - batch_size : Batch size for training and test.
        - epochs : How many times that the model train whole dataset
        - num_workers : How many subprocess to use for data loading. 
            (Quote from pytorch documentation)(for pytorch Dataloader)
        - pld : Whether to apply progressive layer dropping.
            If you set this False, pld will not be applied.
        - layer_keep_prob : Ratio to keep layers. If you want to apply pld, it is needed.
            Paper recommended that the ratio is good between 0.5 and 0.9.
        - train_verbose_step : Verbosity is remained every step which is setted.
        - output_path : Path which model will be saved.

    train() Arguments
        - train_data_path : Data path for training. 
                Sentences are splitted by "\t" and Documents are splittend by "\n\n"
        - dev_data_path : If it is not None, test would be proceeded during training
                Default is None.
"""
class RoBERTaWithPLD():        
    def __init__(
        self,
        vocab, max_len=512, num_workers=5,
        embedding_dropout=0.1,
        hidden_layers=12, hidden_size=768, hidden_dropout=0.1, attention_heads=12,
        feed_forward_size=3072,
        learning_rate=5e-5, warmup_step=24000,
        adam_ep=6e-4, adam_beta1=0.9, adam_beta2=0.98, weight_decay=0.01,
        batch_size=8, epochs=10, pld=True, layer_keep_prob=0.5,
        train_verbose_step=100,
        save_epoch=100, output_path="output"
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.num_workers = num_workers

        self.embedding_dropout = embedding_dropout
        
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.attention_heads = attention_heads

        self.feed_forward_size = feed_forward_size

        self.learning_rate = learning_rate
        self.warmup_step = warmup_step
        self.adam_ep = adam_ep
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.epochs = epochs

        self.pld = pld
        if pld:
            self.layer_keep_prob = layer_keep_prob
        else:
            self.layer_keep_prob = 1

        self.train_verbose_step = train_verbose_step

        self.output_path = output_path


    def train(self, train_data_path, dev_data_path=None):
        """
        Load dataset

        train_data_path is necessary.
        if dev_data_path exist, load dev_dataset.
        """
        start = datetime.now()

        train_dataset = RoBERTaDataset(
            train_data_path, 
            self.vocab, self.max_len, 
            tqdm_desc="train line Dataset"
        )
        train_dataset.masking_data(tqdm_desc="train data masking")

        dev_dataset = None
        if dev_data_path:
            dev_dataset = RoBERTaDataset(
                dev_data_path, 
                self.vocab, self.max_len, 
                tqdm_desc="test line Dataset"
            )     
            dev_dataset.masking_data(tqdm_desc="dev data masking")

        end = datetime.now()
        print("Load Dataset : " + str(end - start))

        """
        Create dataloader
        """
        start = datetime.now()

        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

        dev_data_loader = None
        if dev_dataset:
            dev_data_loader = DataLoader(
                dev_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )

        end = datetime.now()
        print("Create DataLoader : " + str(end-start))

        """
        Initialize RoBERTa model
        """
        start = datetime.now()

        model = RoBERTaWithPLDModel(
            vocab_size=len(self.vocab), max_len=self.max_len, padding_index=self.vocab["[PAD]"],
            embedding_dropout=self.embedding_dropout,
            hidden_layers=self.hidden_layers, hidden_size=self.hidden_size, 
            hidden_dropout=self.hidden_dropout, attention_heads=self.attention_heads, 
            feed_forward_size=self.feed_forward_size,
        )

        end = datetime.now()
        print("Initialize Model : " + str(end-start))
        print(model)

        """
        Initialize RoBERTa Trainer
        """
        start = datetime.now()

        trainer = RoBERTaWithPLDTrainer(
            model=model,
            learning_rate=self.learning_rate,
            warmup_step=self.warmup_step,
            adam_ep=self.adam_ep, 
            adam_beta1=self.adam_beta1, adam_beta2=self.adam_beta2,
            weight_decay=self.weight_decay,
            train_data_loader=train_data_loader,
            dev_data_loader=dev_data_loader
        )

        end = datetime.now()
        print("Initialize Trainer : " +  str(end-start))

        """
        Train the RoBERTa model with PLD
        """
        gamma = 100/(self.batch_size * self.epochs)

        for epoch in range(self.epochs):
            theta_t = (1 - self.layer_keep_prob) * math.exp(-1 * gamma * epoch) + self.layer_keep_prob
            pld_step = (1 - theta_t) / self.hidden_layers
            
            trainer.train(
                epoch, pld_step, trainable=True,
                train_verbose_step=self.train_verbose_step
            )
            trainer.save(epoch, self.output_path)

            if dev_data_loader:
                trainer.train(
                    epoch, pld_step=0, trainable=False, 
                    train_verbose_step=self.train_verbose_step
                )

            if epoch % self.save_epoch == 0:
                trainer.save(self.output_path)

            """
            RoBERTa make new masked dataset every epoch.
            """
            train_dataset.masking_data(tqdm_desc="train data masking")
            if dev_dataset:
                dev_dataset.masking_data(tqdm_desc="dev data masking")

