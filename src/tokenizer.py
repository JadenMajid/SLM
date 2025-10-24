import numpy as np
import re
import pandas as pd

def bigramize(text:str)->tuple[np.array, dict[str, i64], dict[i64, str]]:
    tokenmap : dict[i64,str]={}

    outputdata : np.array=np.array([])
    escaped : str = re.escape(text).replace("\\"," \\")
    words = np.array(escaped.split(' '))

    monograms, counts = np.unique(words, return_counts=True)

    print(monograms)


    return 

def import_southpark()->str:
    df : pd.Dataframe = pd.read_csv("data/southpark.csv")
    df["Line"] = df["Character"]+ " : " + df["Line"]
    print(df["Season"])
    df["id"] = "S"+ df["Season"].str.zfill(2)+ "E"+ df["Episode"].str.zfill(2)
    print(df.groupby(["id"]).aggregate({"Line":"sum"}).head(10))
    



if __name__ == "__main__":
    import_southpark()