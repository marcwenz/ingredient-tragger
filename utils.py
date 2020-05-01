class Vocabulary:

    def __init__(self):
        self.w2i = {}
        self.i2w = {}

    def _add(self, word):
        if word in self.w2i.keys():
            return self.w2i[word]

        idx = len(self.w2i)
        self.w2i[word] = idx
        self.i2w[idx] = word

        return self.w2i[word]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2w[item]
        elif isinstance(item, str):
            return self.w2i[item]
        elif hasattr(item, "__iter__"):
            return [self[ele] for ele in item]
        else:
            raise ValueError(f"Unknown type: {type(item)}")

    def __contains__(self, item):
        return item in self.w2i or item in self.i2w

    def __len__(self):
        return len(self.w2i)


def populateVocab(words, vocab):
    """
    Adds list words to the Vocabulary vocab
    """
    for w in words:
        vocab._add(w)
    return vocab
