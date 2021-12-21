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
import torchtext
import collections
from models import Transformer
from vocabular import Vocabular
from transformer_dataset import TransTrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)



class Config(object):
    def __init__(self):
        self.embed_dim = 100
        self.attn_inner_dim = 200
        self.hidden_dim = 100
        self.n_heads = 10
        assert self.attn_inner_dim % self.n_heads == 0

        self.padding_size = 30
        self.UNK = 1
        self.PAD = 0
        self.dropout_rate = 0.1
        self.encoder_nums = 6
        self.decoder_nums = 8
        self.punctuation = [',', '.', ':', '$', "'", ';', '£', '"', "“", "„", "#", "(", ")", "[", "]", "{", "}", "!", "?"]
        self.device_ids = [0, 1, 2]


config = Config()

if __name__ == "__main__":
    torch.manual_seed(0)
    loaded_dataset = datasets.load_dataset("wmt14", "de-en")
    src_vocab = Vocabular(os.path.join("./cache", "vocab", "de.txt"))
    dst_vocab = Vocabular(os.path.join("./cache", "vocab", "en.txt"))
    train_dataset = TransTrainDataset(loaded_dataset, src_vocab, dst_vocab, "train", config)
    train_loader = DataLoader(train_dataset, batch_size=32 * len(config.device_ids), shuffle=False)
    device_ids = config.device_ids
    model = Transformer(config, len(src_vocab), len(dst_vocab))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    writer = SummaryWriter("./board")

    global_step = 0
    for epoch in range(50):
        model.train()
        tq = tqdm.tqdm(train_loader, desc="Train in Epoch {}".format(epoch))
        for batch_src, batch_dst in tq:
            batch_src = batch_src.cuda(device=0)
            batch_dst: torch.Tensor = batch_dst.cuda(device=0)
            preds_dst: torch.Tensor = model(batch_src, batch_dst)
            loss = torch.nn.CrossEntropyLoss()(preds_dst.permute((0, 2, 1)), batch_dst)
            confidence = torch.mean(torch.nn.Softmax(dim=-1)(preds_dst).max(dim=-1).values)
            preds_dst = preds_dst.argmax(dim=-1)
            batch_dst = batch_dst.unsqueeze(dim=1)
            preds_dst = [['{}'.format(word) for word in sentence] for sentence in preds_dst]
            batch_dst = [[['{}'.format(word) for word in sentence[0]]] for sentence in batch_dst]
            bleu_score = torchtext.data.bleu_score(preds_dst, batch_dst)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tq.set_postfix({"loss": loss.item(), "bleu": bleu_score, "confidence": confidence.item()})
            writer.add_scalar("loss", loss.item(), global_step)
            writer.add_scalar("bleu", bleu_score, global_step)
            writer.add_scalar("confidence", confidence.item(), global_step)
            global_step += 1

        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "optimizer": optimizer}, "./model.bin")
