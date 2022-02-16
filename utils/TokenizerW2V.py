import sentencepiece as spm
import numpy as np


class Vocab(object):
    """vocabulary (terminal symbols or path names or label(method names))"""

   
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.itosubtokens = {}
        self.freq = {}

    
    def append(self, name, index=None, subtokens=None):
        if name not in self.stoi:
            if index is None:
                index = len(self.stoi)
            if self.freq.get(index) is None:
                self.freq[index] = 0
            self.stoi[name] = index
            self.itos[index] = name
            if subtokens is not None:
                self.itosubtokens[index] = subtokens
            self.freq[index] += 1

    def get_freq_list(self):
        freq = self.freq
        freq_list = [0] * self.len()
        for i in range(self.len()):
            freq_list[i] = freq[i]
        return freq_list

    def len(self):
        return len(self.stoi)


class TokenIns:
    def __init__(
        self,
        word2vec_file="./vocab/tokens/emb.txt",
        tokenizer_file="./vocab/tokens/fun.model",
    ):
        # better way to make w2v not this global?
        # it is so messed up with the absolute path, preprocessing, and when creating node representations
        # word tokenizer
        # Load word tokenizer and word2vec model
        self.word_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)
        self.parse_emb_txt(word2vec_file)
        #print("Load Subtokens Index of Statement")
        #self.statement_subtokens = pickle.load( open("./vocab/tokens/statement_subtokens.pt", "rb") )
        #print("\nEmbedding length is %d.\n" % self.vector_size)
     
 

    def getVocabSize(self):
        return len(self.word2vec)

    def parse_emb_txt(self, embtxtfile):
        #print("Loading Glove Model")
        self.word2vec = {"<pad>":np.zeros(100), "<unk>":np.zeros(100)}
        self.wordIndex = {"<pad>":0,"<unk>":1}
        self.vector_size = 100
        with open(embtxtfile,'r') as f:
            for line in f:
                splitLines = line.split()
                if len(splitLines) == 2:
                    continue
                word = splitLines[0]
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                self.word2vec[word] = wordEmbedding
                self.wordIndex[word] = len(self.wordIndex)

        #print(len(self.word2vec)," words loaded!")
    
    def get_tokens_id(self, setence):
        subtokens = self.word_tokenizer.encode(setence.strip(), out_type=str)
        ids = []
        for s in subtokens:
            if s in self.wordIndex:
               ids.append( self.wordIndex[s] )  
            else:
                ids.append(self.wordIndex["<unk>"] )
        return ids

