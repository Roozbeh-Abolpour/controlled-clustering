import pandas as pd
import numpy as np

class DataStream:
    def __init__(self,file_name):
        df=pd.read_csv(file_name)
        self.data=df.to_numpy()
        self.sample_size=self.data.shape[1]
        self.current_index=1

    def has_next_sample(self):
        return self.current_index < self.data.shape[0]

    def next_sample(self):
        if not self.has_next_sample():
            raise StopIteration("No more samples available.")        
        sample=self.data[self.current_index,:]
        self.current_index+=1
        return sample