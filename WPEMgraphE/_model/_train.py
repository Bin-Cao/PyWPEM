from ase.io import read,write
from ase.db import connect
import random


class CSNN_model(object):
    def __init__(self, database):
        if database.endswith('.db'): 
            self.db = connect(database)
        else:
            print('CSNN only supports data types of .db')

        # Get the total number of entries in the database
        total_entries = database.count()

        numbers = list(range(1, total_entries+1))
        random.shuffle(numbers)
       
        train_size = int(0.8 * total_entries)
        self.train_list = numbers[:train_size]
        self.val_list = numbers[train_size:]

    def train(self,):
        

        
        pass