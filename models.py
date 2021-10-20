import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat


class Config(object):
    def __init__(self):
        self.embed_dim = 100
        self.attn_inner_dim = 200
        self.n_heads = 10
        assert self.attn_inner_dim % self.n_heads == 0

        self.padding_size = 30
        self.UNK = 1
        self.PAD = 0
        self.dropout_rate = 0.1
        self.punctuation = [',', '.', ':', '$', "'", ';', '£', '"', "“", "„", "#", "(", ")", "[", "]", "{", "}", "!", "?"]


config = Config()


class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=config.PAD)

    def forward(self, x):
        x = self.embedding(torch.tensor(x))
        return x


class Multihead_Attention(nn.Module):
    def __init__(self, embed_dim, inner_dim, n_heads):
        super(Multihead_Attention, self).__init__()
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        assert self.inner_dim % self.n_heads == 0

        self.mq = nn.Linear(embed_dim, inner_dim, bias=False)
        self.mk = nn.Linear(embed_dim, inner_dim, bias=False)
        self.mv = nn.Linear(embed_dim, inner_dim, bias=False)

        self.out = nn.Linear(inner_dim, embed_dim)
        self.dk = 1 / math.sqrt(inner_dim)

    def generate_mask(self, dim):
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matrix), 1)
        return ~mask

    def forward(self, x, y, requires_mask=False):
        _, n, _ = x.shape
        q = self.mq(x)
        k = self.mk(x)
        v = self.mv(y)
        alpha = torch.einsum('b i k, b j k -> b i j', q, k) * self.dk
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # alpha(:, i, j) means align(query_i, key_j)
        if requires_mask:
            mask = self.generate_mask(n)
            alpha.masked_fill(mask, value=float("-inf"))
        context_vector = torch.einsum('b i k, b k j -> b i j', alpha, v)
        output = self.out(context_vector)
        return output


class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output


class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm((config.padding_size, config.embed_dim))

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_output)
        out = self.layerNorm(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.multi_atten = Multihead_Attention(embed_dim, config.attn_inner_dim, config.n_heads)
        self.feed_forward = Feed_Forward(config.embed_dim)

        self.add_norm = Add_Norm()

    def forward(self, x):
        b, n, e = x.shape
        pos_encoding = torch.tensor([i for i in range(e)], dtype=torch.float32)
        x += repeat(pos_encoding, '() () e -> b n e')
        output = self.add_norm(x, self.multi_atten, y=x)
        output = self.add_norm(output, self.feed_forward)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.multi_atten = Multihead_Attention(config.embed_dim, config.attn_inner_dim, config.n_heads)
        self.feed_forward = Feed_Forward(config.attn_inner_dim)
        self.add_norm = Add_Norm()

    def forward(self, x, encoder_output):  # batch_size * seq_len 并且 x 的类型不是tensor，是普通list
        b, n, e = x.shape
        pos_encoding = torch.tensor([i for i in range(e)], dtype=torch.float32)
        x += repeat(pos_encoding, '() () e -> b n e')
        output = self.add_norm(x, self.muti_atten, y=x, requires_mask=True)
        output = self.add_norm(output, self.muti_atten, y=encoder_output, requires_mask=True)
        output = self.add_norm(output, self.feed_forward)
        return output


class Transformer_layer(nn.Module):
    def __init__(self):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_input, x_output):
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)
        return encoder_output, decoder_output


class Transformer(nn.Module):
    def __init__(self, layer_nums, src_vocab_size, dst_vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size=src_vocab_size)
        self.embedding_output = Embedding(vocab_size=dst_vocab_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.attn_inner_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_layer() for _ in range(layer_nums)])

    def forward(self, src_input, dst_output):
        src_input = self.embedding_input(src_input)
        dst_output = self.embedding_output(dst_output)

        _, output = self.model(src_input, dst_output)

        output = self.linear(output)
        output = self.softmax(output)

        return output
