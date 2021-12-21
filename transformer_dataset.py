import torch
from torch.utils.data import Dataset
from vocabular import Vocabular


class TransTrainDataset(Dataset):
    def __init__(self, loaded_dataset, src_vocab, dst_vocab, mode, config):
        self.data = loaded_dataset[mode]
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab
        self.config = config

    def __len__(self):
        return 138

    def __getitem__(self, idx):
        """
        :return: sentence -> word -> one-hot-encoding
        """

        def pad(word_list, config):
            if len(word_list) < config.padding_size:
                word_list.extend([config.PAD] * (config.padding_size - len(word_list)))
            else:
                word_list = word_list[:config.padding_size]
            return word_list


        src_sentence = list(self.data[idx]['translation'].values())[0]
        src_ont_hot = []
        for word in src_sentence.strip().split(' '):
            src_ont_hot.append(self.src_vocab.stoi.get(word, self.src_vocab.stoi['UNK']))

        dst_sentence = list(self.data[idx]['translation'].values())[1]
        dst_one_hot = []
        for word in dst_sentence.strip().split(' '):
            dst_one_hot.append(self.dst_vocab.stoi.get(word, self.dst_vocab.stoi['UNK']))
        return torch.tensor(pad(src_ont_hot, self.config)), torch.tensor(pad(dst_one_hot, self.config))