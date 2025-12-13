class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3

        self._init_special_tokens()

    def _init_special_tokens(self):
        special_tokens = {
            '<PAD>': self.pad_token_id,
            '<UNK>': self.unk_token_id,
            '<EOS>': self.eos_token_id,
            '<BOS>': self.bos_token_id
        }
        self.vocab.update(special_tokens)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts, min_freq=1):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[word] = token_id
                self.reverse_vocab[token_id] = word

        self.vocab_size = len(self.vocab)

    def encode(self, text):
        tokens = text.split()
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids):
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                word = self.reverse_vocab[token_id]
                if word not in ['<PAD>', '<BOS>', '<EOS>']:
                    words.append(word)
        return ' '.join(words)

