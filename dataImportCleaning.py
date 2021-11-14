import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pathlib

from typing import Union, Tuple, List, Iterable

'''defity the fuction to import, split, clean and normalize data'''
class dataTool():
    def __init__(self, trainDF_Path: str, testDF_Path: str):
        '''importing training and test sets'''
        self.train = pd.read_csv(trainDF_Path)
        self.test = pd.read_csv(testDF_Path)
    #cleaning part
    def cleaning(self):
        '''to be defined'''
        pass
    #splitting part
    def split_data(self, data: pd.DataFrame, targetVar: str, split: bool = False, testSize: float = 0.25) -> Iterable[Union[pd.DataFrame, pd.DataFrame]]:
        '''split data based on users defne paramater'''
        df = data
        x_df = df.drop(targetVar, axis=1)
        y_df = df[targetVar]

        if split:

            x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=testSize)
            self.brr

            return x_train, x_test, y_train, y_test
        
        self.brr

        return x_df, y_df
    # normalizing part
    def normalize(self, trainData, data: List[pd.DataFrame]) -> np.array: 
        '''normalize data with minmaxscaler, using train data to train model'''
        outputList = []
        scalar = MinMaxScaler()
        scalar.fit(trainData)
        
        for item in data:
            normalData = scalar.transform(item)
            outputList.append(normalData)

        return outputList 
        
    def __create_directory(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    def write_models(self, path):
        '''
        Kaan write your code here :dogeBuff:
        and also a really quick but clear documentation for the method :dogeBuff:
        '''
        self.__create_directory(path) # the path should end with a '/' for example 'models/'
        # Here use the pickle library to "pickle" the models, the return from that is a .pkl file
        # which can then be imported and used for prediction on any platform, even webpages back code, cool huh?
        # dogeBuff
        return f'Models written, located in {path}'


    @property
    def retrieveData(self) -> Union[pd.DataFrame, pd.DataFrame]:
        self.brr

        return self.train, self.test

    @property
    def brr(self):
        print(f'Data go {"brr"}')