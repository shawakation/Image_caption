class Vocabulary(object):
    def __init__(self):
        self.index = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.index
            self.idx2word[self.index] = word
            self.index += 1

    def __call__(self, word: str) -> int:
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']

    def __len__(self) -> int:
        return self.index


if __name__ == '__main__':
    print(Vocabulary)
    print('it\'s only a sample')
