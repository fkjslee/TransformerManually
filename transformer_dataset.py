from torch.utils.data import Dataset
from vocabular import Vocabular
from utils import split_sentence
from models import config


class TransTrainDataset(Dataset):
    def __init__(self, loaded_dataset, src_vocab, dst_vocab, config, mode):
        self.data = loaded_dataset[mode]
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        :param item: index
        :return: sentence -> word -> one-hot-encoding
        """

        def pad(word_list, config):
            if len(word_list) < config.padding_size:
                word_list.extend([config.PAD] * (config.padding_size - len(word_list)))
            else:
                word_list = word_list[:config.padding_size]
            return word_list


        src_ont_hot = []
        for word in self.src_sentences[item].strip().split(' '):
            src_ont_hot.append(self.src_vocab.stoi[word])

        dst_one_hot = []
        for word in self.dst_sentences[item].strip().split(' '):
            dst_one_hot.append(self.dst_vocab.stoi[word])
        return pad(src_ont_hot, self.config), pad(dst_one_hot, self.config)