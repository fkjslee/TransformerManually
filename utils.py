
def split_sentence(sentence, config):
    res = []
    for word in sentence.strip().split(' '):
        word_list = []
        last_word_list = []
        while word and word[0] in config.punctuation:
            word_list.append(word[0])
            word = word[1:]
        while word and word[-1] in config.punctuation:
            last_word_list.append(word[-1])
            word = word[:-1]
        last_word_list.reverse()
        res.extend(word_list + [word] + last_word_list)
    return res