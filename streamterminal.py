import pandas as pd
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", help="batchsize", nargs='?', type=int, const=1, default=10)
args = parser.parse_args()
count = 0
for dtfrm_batch in pd.read_csv("train.csv", index_col=0 , chunksize = args.batchsize):
     print(dtfrm_batch, end='\n\n')
     time.sleep(2)
     count += 1 
     #dtfrm_batch.drop(0, inplace=True)
     dtfrm_batch = dtfrm_batch.iloc[1: , :]
     dtfrm_batch.to_csv("train_batch_"+str(count) + ".csv", encoding='utf-8-sig')     
     data = pd.read_csv("train_batch_"+str(count) + ".csv") 
     print(data)
