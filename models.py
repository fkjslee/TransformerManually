import torch
import numpy as np
import math
from einops import rearrange, repeat


class Multihead_Attention(torch.nn.Module):
    def __init__(self, config):
        super(Multihead_Attention, self).__init__()
        self.n_heads = config.n_heads
        self.inner_dim = config.attn_inner_dim
        assert self.inner_dim % self.n_heads == 0

        self.mq = torch.nn.Linear(config.embed_dim, self.inner_dim, bias=False)
        self.mk = torch.nn.Linear(config.embed_dim, self.inner_dim, bias=False)
        self.mv = torch.nn.Linear(config.embed_dim, self.inner_dim, bias=False)

        self.out = torch.nn.Linear(self.inner_dim, config.embed_dim)
        self.dk = 1 / math.sqrt(self.inner_dim)
        self.config = config

    def generate_mask(self, dim):
        matrix = np.ones((dim, dim))
        return ~torch.Tensor(np.tril(matrix, 0)).bool().to(self.config.device_ids[0])

    def forward(self, x, y, mask_sequence=False):
        _, n, _ = x.shape
        q = self.mq(y)
        k = self.mk(x)
        v = self.mv(x)
        alpha = torch.einsum('b i k, b j k -> b i j', q, k) * self.dk
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # alpha(:, i, j) means align(query_i, key_j)
        if mask_sequence:
            mask = self.generate_mask(n)
            alpha.masked_fill(mask.to(alpha.device), value=float("-inf"))
        context_vector = torch.einsum('b i k, b k j -> b i j', alpha, v)
        return self.out(context_vector)


class Feed_Forward(torch.nn.Module):
    def __init__(self, config):
        super(Feed_Forward, self).__init__()
        self.L1 = torch.nn.Linear(config.embed_dim, config.hidden_dim)
        self.L2 = torch.nn.Linear(config.hidden_dim, config.embed_dim)

    def forward(self, x):
        return self.L2(torch.nn.ReLU()(self.L1(x)))


class EncoderLayer(torch.nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.multi_atten = Multihead_Attention(config)
        self.feed_forward = Feed_Forward(config)
        self.multihead_layer_norm = torch.nn.LayerNorm(config.embed_dim)
        self.FF_layer_norm = torch.nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        multihead_res = self.multi_atten(x, x) + x
        multihead_res = self.multihead_layer_norm(multihead_res)
        FF_res = self.feed_forward(multihead_res) + multihead_res
        FF_res = self.FF_layer_norm(FF_res)
        return FF_res


class DecoderLayer(torch.nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.masked_multi_atten = Multihead_Attention(config)
        self.masked_multi_atten_layer_norm = torch.nn.LayerNorm(config.embed_dim)
        self.multi_atten = Multihead_Attention(config)
        self.multi_atten_layer_norm = torch.nn.LayerNorm(config.embed_dim)
        self.feed_forward = Feed_Forward(config)
        self.FF_layer_norm = torch.nn.LayerNorm(config.embed_dim)

    def forward(self, dst_output, encoder_output):
        masked_multihead_atten = self.masked_multi_atten(dst_output, dst_output, mask_sequence=True) + dst_output
        masked_multihead_atten = self.masked_multi_atten_layer_norm(masked_multihead_atten)
        multihead_atten = self.multi_atten(encoder_output, masked_multihead_atten) + masked_multihead_atten
        multihead_atten = self.multi_atten_layer_norm(multihead_atten)
        FF_res = self.feed_forward(multihead_atten) + multihead_atten
        FF_res = self.FF_layer_norm(FF_res)
        return FF_res


class Transformer(torch.nn.Module):
    def __init__(self, config, src_vocab_size, dst_vocab_size):
        super(Transformer, self).__init__()
        self.embedding_input = torch.nn.Embedding(src_vocab_size, config.embed_dim)
        self.encoder = torch.nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_nums)])
        self.embedding_output = torch.nn.Embedding(dst_vocab_size, config.embed_dim)
        self.decoder = torch.nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_nums)])
        self.linear = torch.nn.Linear(config.embed_dim, dst_vocab_size)
        self.config = config

    def forward(self, src_input, dst_output):
        b, n = src_input.shape
        pos_encoding = torch.tensor([i for i in range(n)], dtype=torch.long, device=src_input.device)
        src_input = src_input + repeat(pos_encoding, 'n -> b n', b=b)
        src_input = self.embedding_input(src_input)
        for module in self.encoder:
            src_input = module(src_input)
        b, n = dst_output.shape
        pos_encoding = torch.tensor([i for i in range(n)], dtype=torch.long, device=src_input.device)
        dst_output = dst_output + repeat(pos_encoding, 'n -> b n', b=b)
        dst_output = self.embedding_output(dst_output)
        for module in self.decoder:
            dst_output = module(src_input, dst_output)
        return self.linear(dst_output)
