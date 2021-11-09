import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class dataTool():
    def __init__(self, trainDF_Path: str, testDF_Path: str):
        self.train = pd.read_csv(trainDF_Path)
        self.test = pd.read_csv(trainDF_Path)

    def cleaning(self):
        pass
    
    def split_data(self, data: pd.DataFrame, targetVar: str, split: bool = False, testSize: float = 0.25):
        df = data
        x_df = df.drop(targetVar, axis=1)
        y_df = df[targetVar]

        if split:
            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=testSize)
            self.brr
            return x_train, x_test, y_train, y_test
        
        self.brr
        return x_df, y_df

    @property
    def retrieveData(self):
        self.brr
        return self.train, self.test

    @property
    def brr(self):
        print(f'Data go {"brr"}')