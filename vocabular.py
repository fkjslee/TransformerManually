import os
import collections
import datasets
import tqdm


class Vocabular:

    def __init__(self, path):
        self.itos = []
        self.stoi = {}
        with open(path, encoding="utf-8") as f:
            for line in f.readlines():
                word = line.strip().split(' ')[0]
                self.itos.append(word)
                self.stoi[word] = len(self.itos)

    def __len__(self):
        return len(self.itos)


def build_vocab_cache(loaded_dataset, over_write=True):
    vocab_root_path = os.path.join("./", "cache", "vocab")
    if not os.path.exists(vocab_root_path) or over_write:
        if not os.path.exists(vocab_root_path):
            os.makedirs(vocab_root_path)
        src_word = collections.defaultdict(lambda: 0)
        dst_word = collections.defaultdict(lambda: 0)
        for i, translation in enumerate(tqdm.tqdm(loaded_dataset['train'], desc="Build {}".format('train'))):
            for word in translation['translation']['de'].strip().split(' '):
                src_word[word] += 1
            for word in translation['translation']['en'].strip().split(' '):
                dst_word[word] += 1
        with open(os.path.join(vocab_root_path, "de.txt"), "w", encoding="utf-8") as f_src, open(os.path.join(vocab_root_path, "en.txt"), "w", encoding="utf-8") as f_dst:
            for key in ['PAD', 'UNK']:
                f_src.write('{} {}\n'.format(key, 0))
                f_dst.write('{} {}\n'.format(key, 0))
            for (key, val) in sorted(src_word.items(), key=lambda item: item[1], reverse=True):
                if val <= 5:
                    break
                f_src.write('{} {}\n'.format(key, val))
            for (key, val) in sorted(dst_word.items(), key=lambda item: item[1], reverse=True):
                if val <= 5:
                    break
                f_dst.write('{} {}\n'.format(key, val))


if __name__ == "__main__":
    loaded_dataset = datasets.load_dataset("wmt14", "de-en")
    build_vocab_cache(loaded_dataset)
