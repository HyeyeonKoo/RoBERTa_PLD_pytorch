#-*-coding:utf-8-*-

import torch.nn as nn
import torch
import random


"""
RoBERTa model
    RoBERTa architecture is like BERT.

    __init__() Arguments
        - vocab_size : Whole vocab size for mapping each token to vector.
        - max_len : Maximum length of input tokens. 
        - padding_index : For segment embedding, it is needed.
            Segment labels are composed of one sentence index, another sentence index, [PAD] token index.
            So, segment dictionary keys are 0, 1, 2.
            Sentence indices are 1, 2. So, [PAD] index must be 0.
        - embedding_dropout : Dropout probability for embedding.
        - hidden_layers : Number of transformer blocks to stack.
        - hidden_size : Dimmension of each token.
        - hidden_dropout : Dropout probability for transformer blocks.
        - attention_heads : Number of attention heads for transformer blocks.
        - feed_forward_size : Hidden size for linear calculation in transformer block.

Progressive Layer Dropping
    Paper :https://arxiv.org/pdf/2010.13369.pdf
    Drop layer by probability which is decided by each epoch, depth of block and keep probability.
"""
class RoBERTaWithPLDModel(nn.Module):
    def __init__(
            self,
            vocab_size, max_len, padding_index,
            embedding_dropout,
            hidden_layers, hidden_size, hidden_dropout, attention_heads,
            feed_forward_size
    ):
        super().__init__()

        self.embedding = Embedding(
            vocab_size, max_len, 
            embedding_dropout, 
            hidden_size, padding_index
        )

        self.transformer_block = nn.ModuleList(
            [TransformerBlock(
                max_len,
                hidden_size, hidden_dropout, attention_heads,
                feed_forward_size
            ) for _ in range(hidden_layers)]
        )

        self.mlm_loss = MaskedLanguageModel(
            vocab_size, hidden_size
        )


    def forward(self, x, segment, pld_step, trainable):
        x_ = self.embedding.forward(x, segment)

        p = 1
        if trainable:
            for tf in self.transformer_block:
                action = random.randrange(0, 2)
                
                if action == 0:
                    pass
                else:
                    x_ = tf.forward(x_, p)

                p -= pld_step
        else:
            for tf in self.transformer_block:
                x_ = tf.forward(x_, p)

        x_ = self.mlm_loss.forward(x_)

        return x_


"""
BERT Embedding is composed of token, segment, position.
"""
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_dropout, hidden_size, padding_index):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size)
        self.segment_embedding = SegmentEmbedding(hidden_size, padding_index)
        self.position_embedding = PositionEmbedding(hidden_size, max_len)
        self.dropout = nn.Dropout(p=embedding_dropout)


    def forward(self, x, segment):
        x_ = (
            self.token_embedding(x)
            + self.segment_embedding(segment)
            + self.position_embedding(x)
        )
        return self.dropout(x_)


"""
Mapping each token to hidden size vector.
"""
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_size):
        super().__init__(vocab_size, hidden_size)


"""
Mapping segment token to hidden size vector.
Each sentences seperate 0, 1 and padding index.
"""
class SegmentEmbedding(nn.Embedding):
    def __init__(self, hidden_size, padding_index):
        super().__init__(3, hidden_size, padding_idx=padding_index)


"""
Positional embedding values are all different.
So, it need to be nn.Module, not nn.Embedding.

pe(pos, 2i) = sin(pos/10000**(2i/d_model))
pe(pos, 2i+1) = cos(pos/10000**(2i/d_model))
"""
class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super().__init__()

        position = torch.arange(0, max_len)
        position_embedding = position / (10000 ** (2 * position / hidden_size))
        position_embedding = torch.stack([position_embedding]*hidden_size, dim=1)
        
        position_embedding.requires_grad = False
        self.register_buffer("pe", position_embedding)


    def forward(self, x):
        return self.pe


"""
Transformer Block is composed of layer normalization, multi head attention, position-wise feed forward network.
For progressive layer dropping, normalize with (1/p) after calculate each sublayer.
"""
class TransformerBlock(nn.Module):
    def __init__(
        self, max_len, 
        hidden_size, hidden_dropout, attention_heads, 
        feed_forward_size
    ):
        super().__init__()
    
        self.pre_layer_norm_1 = nn.LayerNorm([max_len, hidden_size])
        self.dropout_1 = nn.Dropout(p=hidden_dropout)
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, attention_heads, hidden_dropout)

        self.pre_layer_norm_2 = nn.LayerNorm([max_len, hidden_size])
        self.dropout_2 = nn.Dropout(p=hidden_dropout)
        self.feed_forward_1 = nn.Linear(hidden_size, feed_forward_size)
        self.feed_forward_2 = nn.Linear(feed_forward_size, hidden_size)
        self.activation = nn.GELU()


    def forward(self, x, p):
        x_ = self.pre_layer_norm_1(x)
        x_ = self.dropout_1(x_)
        x_ = x_.view([x_.shape[1], x_.shape[0], x_.shape[2]])
        x_ = self.multi_head_attention(x_, x_, x_)[0]
        x_ = x_.view([x_.shape[1], x_.shape[0], x_.shape[2]])
        x = (x + x_) * (1/p)

        x_ = self.pre_layer_norm_2(x)
        x_ = self.dropout_2(x_)
        x_ = self.feed_forward_1(x_)
        x_ = self.feed_forward_2(x_)
        x_ = self.activation(x_)
        x_ = (x + x_) * (1/p)

        return x_


"""
RoBERTa has MLM only, not NSP(next sentence prediction).
Log softmax make the loss inscrease, so model can get better performance.
"""
class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    
    def forward(self, x):
        x_ = self.linear(x)
        x_ = self.softmax(x_)
        return x_
