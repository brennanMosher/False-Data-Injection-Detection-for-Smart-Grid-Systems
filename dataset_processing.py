import csv    
from datetime import datetime
from pytimeparse import parse
import os
import numpy
import pandas as pd


# 70-30 split
def seventy_thirty (csvfilename):
    df = pd.read_csv('.\dataset\\' + csvfilename)
    rng = numpy.random .RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]
    
    train.to_csv('.\dataset\\' + "seventy_thirty\\" + 'training-' + csvfilename)
    test.to_csv('.\dataset\\' + "seventy_thirty\\" + 'testing-' + csvfilename)


seventy_thirty("data_injection_and_normal_events_dataset.csv")



