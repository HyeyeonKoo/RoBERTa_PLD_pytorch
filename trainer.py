#-*-coding:utf-8-*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


class RoBERTaWithPLDTrainer():
    def __init__(
        self, model,
        learning_rate, warmup_step,
        adam_ep, adam_beta1, adam_beta2, weight_decay,
        train_data_loader, dev_data_loader
    ):
        self.device = self.get_device()
        self.model = self.model_parallel_or_not(self.device, model)
        print("Total Parameters :", sum(
            [layer.nelement() for layer in self.model.parameters()]
        ))

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_ep,
            weight_decay=weight_decay
        )

        self.schedule = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda lr_step: (lr_step / warmup_step) * learning_rate \
                if lr_step < warmup_step else learning_rate
        )

        self.loss_fn = nn.NLLLoss(ignore_index=0)

        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader

        
    def get_device(self):
        device = None

        if torch.cuda.is_available():
            device = "cuda:0"
            print("Device : cuda")
        else:
            device = "cpu"
            print("Device : cpu")

        return device


    def model_parallel_or_not(self, device, model):
        if device == "cuda:0":
            model_ = model.to(device)
        else:
            model_ = model
        
        if torch.cuda.device_count() > 1:
            return nn.parallel.DistributedDataParallel(model_)
        else:
            return model_


    def train(self, epoch, pld_step, trainable=True, train_verbose_step=100):
        data_loader = None
        if trainable:
            data_loader = self.train_data_loader
        else:
            data_loader = self.dev_data_loader

        for i , data in enumerate(data_loader):
            data = {k: v.to(self.device) for k, v in data.items()}
            current_lr = self.schedule.get_last_lr()[0]
            output = self.model.forward(data["input"], data["segment"], pld_step, trainable)
            loss = self.loss_fn(output.transpose(1, 2), data["label"])

            if trainable:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()

            if i == 0 or i % train_verbose_step == 0 or i == len(data_loader) - 1:
                print({
                    "epoch": epoch, 
                    "step": i,
                    "lr" : current_lr,
                    "loss" : loss.item(),
                })

        
    def save(self, epoch, path):
        torch.save(self.model.cpu(), path)
        self.model.to(self.device)
        print("Epoch :", epoch, "Save :", path)
