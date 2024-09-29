class Vocabulary:
    def __init__(self, freq_threshold):
        self.index_to_string = {
            0: "<PAD>", 
            1: "<SOS>", 
            2: "<EOS>", 
            3: "<UNK>"}
        
        self.string_to_index = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.index_to_string)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.string_to_index[word] = idx
                    self.index_to_string[idx] = word
                    idx += 1
    
    def tokenize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.string_to_index[token] if token in self.string_to_index else self.string_to_index["<UNK>"]
            for token in tokenized_text
        ]
    
    def decode(self, token_indices):
        """
        Convert a list of token indices into a string of words.
        
        Args:
            token_indices (list): List of token indices to convert.
        
        Returns:
            str: The decoded sentence.
        """
        words = []
        for index in token_indices:
            word = self.index_to_string.get(index)
            if word == "<EOS>":
                break
            words.append(word)
        
        return ' '.join(words)