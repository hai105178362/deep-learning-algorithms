import pandas as pd
from helper import phoneme_list as pl
import numpy as np



def generate_table(path,destination):
    id = np.arange(523)
    pmap = pl.PHONEME_MAP
    result = np.random.choice(pmap,523)
    assert len(result) == len(id)
    info = {'Id':id,'Predicted':result}
    df = pd.DataFrame(info)
    df.to_csv(path_or_buf=destination,index=False)



if __name__ == "__main__":
    PATH = "/Users/robert/Documents/CMU/19Fall/11785/11785-deep-learning/kaggle/rnn/dataset.nosync/HW3P2_Data/sample_submission.csv"
    PATH_DESKTOP = '/Users/robert/Desktop/hw3_kaggle.csv'
    generate_table(PATH,PATH_DESKTOP)