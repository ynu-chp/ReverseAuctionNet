import pandas as pd
import numpy as np
import os

def get_total_value_budget(value,bid,budget):
    total_value=0
    value=value.flatten()
    bid=bid.flatten()
    p=value/bid
#     print(p)
    sorted_indices = np.argsort(p)
    sorted_indices_descending = sorted_indices[::-1]
    for i in sorted_indices_descending:
        if budget>=bid[i]:
            total_value+=value[i]
            budget-=bid[i]
        else:
            total_value+=budget*p[i]
            break
    return total_value


def get_total_value(path,budget):
    total_value=0.0
    value=np.load(os.path.join(path, 'value.npy')).astype(np.float32)
    bid=np.load(os.path.join(path, 'bid.npy')).astype(np.float32)
    for i in range(value.shape[0]):
        total_value+=get_total_value_budget(value[i],bid[i],budget)
    return total_value/value.shape[0]

for i in range(2,11,2):
    for budget in range(8,30,4):
        path=os.path.join(os.path.join("data",str(i)+"x6"),"test")
        print(f"bidder={i},budget={budget},total_value={get_total_value(path,budget)}")
    print()