import pickle
import numpy as np

class Selector():
    def __init__(self, batch_size, data_file):
        self.batch_size = batch_size
        with open(data_file, 'rb') as f:
            self.haikus = pickle.load(f)

    def select(self)
