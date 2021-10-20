"""
reference:  https://github.com/lucidrains/vit-pytorch
            https://zhuanlan.zhihu.com/p/420474770
            https://github.com/OpenNMT/OpenNMT-py
"""

import tqdm
import os
import datasets
import logging
import torch
import collections
from models import Transformer, config
from vocabular import Vocabular
from transformer_dataset import TransTrainDataset, split_sentence
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    loaded_dataset = datasets.load_dataset("wmt14", "de-en")
    # src_vocab = Vocabular(os.path.join("./cache", "vocab", "de.txt"))
    # dst_vocab = Vocabular(os.path.join("./cache", "vocab", "en.txt"))
    # train_dataset = TransTrainDataset(loaded_dataset, src_vocab, dst_vocab, "train")
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    # model = Transformer(6, len(src_vocab), len(dst_vocab), 30)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    #
    # for epoch in range(50):
    #     model.train()
    #     tq = tqdm.tqdm(train_loader, desc="Train in Epoch {}".format(epoch))
    #     for batch_src, batch_dst in tq:
    #         preds_dst = model(batch_src, batch_dst)
