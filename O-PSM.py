import pandas as pd
import numpy as np
import os

def OPSM(value,bid,budget):
    total_value=0
    value=value.sum(axis=-1)
    bid=bid.sum(axis=-1)
    t=value/bid
    sorted_indices = np.argsort(t)
    sorted_indices_descending = sorted_indices[::-1]
    for i in sorted_indices_descending:
        if bid[i]<=budget/2*value[i]/(total_value+value[i]):
            total_value+=value[i]
        else:
            break
    return total_value

def get_total_value(path,budget):
    total_value=0.0
    value=np.load(os.path.join(path, 'value.npy')).astype(np.float32)
    bid=np.load(os.path.join(path, 'bid.npy')).astype(np.float32)
    for i in range(value.shape[0]):
        total_value+=OPSM(value[i],bid[i],budget)
    return total_value/value.shape[0]

for i in range(2,11,2):
    for budget in range(8,30,4):
        path=os.path.join(os.path.join("data",str(i)+"x6"),"train")
        print(f"bidder={i},budget={budget},total_value={get_total_value(path,budget)}")
    print()