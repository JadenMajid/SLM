import numpy as np
import re
import pandas as pd
from typing import Union
from mathutils import softmax
import pickle

class BigramDataset:
    X: np.array
    X_dist: np.array
    stoi: dict[str, np.int32]
    itos: dict[np.int32, str]
    n: int
    d: int
    

    def __init__(self, text):
        self.bigramize(text)
        self.n = len(self.X)
        self.d = len(self.stoi.keys())
        return


    def process_text(self, text):
        outputdata : np.array=np.array([])
        escaped : str = text.replace('\xa0', ' ').replace('\n',' \n ').replace('  ', ' ')
        words = np.array(escaped.split(' '))
        return words

    def bigramize(self, text:str):
        words = self.process_text(text)

        int_to_bigram: dict[np.int32,str]={}
        bigram_to_int: dict[str,np.int32]={}
        
        bigrams = np.array([words[i] + ' ' + words[i+1] for i in range(len(words)-1)])
        unique_bigrams, counts = np.unique(bigrams, return_counts=True)
        X_distribution = softmax(counts)

        for i, bigram in enumerate(unique_bigrams):
            int_to_bigram[i] = bigram
            bigram_to_int[bigram] = i

        
        bigrams_i = [bigram_to_int[b] for b in bigrams]

        self.X, self.X_dist, self.stoi, self.itos = bigrams_i, X_distribution, bigram_to_int, int_to_bigram

    def unbigramize(self, bigram: np.int32)->str:
        return self.itos[bigram].split(' ')[0]


def import_southpark() -> str:
    df: pd.DataFrame = pd.read_csv("data/southpark.csv")
    df["Line"] = df["Character"] + " : " + df["Line"]
    df["id"] = "S" + df["Season"].astype(str).str.zfill(2) + "E" + df["Episode"].astype(str).str.zfill(2)
    grouped = df.groupby("id")["Line"].apply(" ".join)
    raw_string = "\n".join(grouped)
    return raw_string



if __name__ == "__main__":
    raw_data = import_southpark()
    ds = BigramDataset(raw_data)
    
    #print(np.all(ds.process_text(raw_data)[:-1] == ds.unbigramize(ds.X)))
    #print(len(ds.itos.keys()))
    with open('data/bigramdataset.pkl', 'wb') as f:
        pickle.dump(ds, f)