import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from typing import Union, Tuple, List, Iterable

class dataTool():
    def __init__(self, trainDF_Path: str, testDF_Path: str):
        self.train = pd.read_csv(trainDF_Path)
        self.test = pd.read_csv(trainDF_Path)

    def cleaning(self):
        pass
    
    def split_data(self, data: pd.DataFrame, targetVar: str, split: bool = False, testSize: float = 0.25) -> Iterable[Union[pd.DataFrame, pd.DataFrame]]:
        df = data
        x_df = df.drop(targetVar, axis=1)
        y_df = df[targetVar]

        if split:
            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=testSize)
            self.brr

            return x_train, x_test, y_train, y_test
        
        self.brr

        return x_df, y_df
    
    def normalize(self, trainData, data: List[pd.DataFrame]) -> np.array: 
        outputList = []
        scalar = MinMaxScaler()
        scalar.fit(trainData)
        
        for item in data:
            normalData = scalar.transform(item)
            outputList.append(normalData)

        return outputList 

    @property
    def retrieveData(self) -> Union[pd.DataFrame, pd.DataFrame]:
        self.brr

        return self.train, self.test

    @property
    def brr(self):
        print(f'Data go {"brr"}')