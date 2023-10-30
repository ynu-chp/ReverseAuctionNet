import numpy as np
import os


def get_data(n_bidder,path,n_data):
    init_value=np.load("data/value.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    idx= np.random.choice(init_value.shape[0], n_bidder*n_data)

    value=init_value[idx,]
    bid=value/2

    for i in range(bid.shape[0]):
        for j in range(bid.shape[1]):
            bid[i,j]=max(0,np.random.normal(bid[i,j], bid[i,j]*0.1))
    value=value.reshape(n_data,n_bidder,-1)

    bid=bid.reshape(n_data,n_bidder,-1)

    path_dir=os.path.join("data",str(n_bidder)+'x'+str(bid.shape[-1]))
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_dir = os.path.join(path_dir, path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    np.save(os.path.join(path_dir,"value"), value, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "bid"), bid, allow_pickle=True, fix_imports=True)



print("generate data:")

train_dir="train"
test_dir="test"

# n=2
# train_n_head=1e5
# test_n_head=500
# get_data(int(n),train_dir,int(train_n_head))
# get_data(int(n),test_dir,int(test_n_head))
# print("bidder={},ok!".format(n))
#
#
# n=3
# train_n_head=1e5
# test_n_head=5000
# get_data(int(n),train_dir,int(train_n_head))
# get_data(int(n),test_dir,int(test_n_head))
# print("bidder={},ok!".format(n))
#
n=4
train_n_head=1e5
test_n_head=5000
get_data(int(n),train_dir,int(train_n_head))
get_data(int(n),test_dir,int(test_n_head))
print("bidder={},ok!".format(n))
#
#
# n=5
# train_n_head=1e5
# test_n_head=5000
# get_data(int(n),train_dir,int(train_n_head))
# get_data(int(n),test_dir,int(test_n_head))
# print("bidder={},ok!".format(n))

n=6
train_n_head=1e5
test_n_head=5000
get_data(int(n),train_dir,int(train_n_head))
get_data(int(n),test_dir,int(test_n_head))
print("bidder={},ok!".format(n))
#
#
# n=7
# train_n_head=1e5
# test_n_head=5000
# get_data(int(n),train_dir,int(train_n_head))
# get_data(int(n),test_dir,int(test_n_head))
# print("bidder={},ok!".format(n))
#
n=8
train_n_head=1e5
test_n_head=5000
get_data(int(n),train_dir,int(train_n_head))
get_data(int(n),test_dir,int(test_n_head))
print("bidder={},ok!".format(n))
#
#
# n=9
# train_n_head=1e5
# test_n_head=5000
# get_data(int(n),train_dir,int(train_n_head))
# get_data(int(n),test_dir,int(test_n_head))
# print("bidder={},ok!".format(n))
#
n=10
train_n_head=1e5
test_n_head=5000
get_data(int(n),train_dir,int(train_n_head))
get_data(int(n),test_dir,int(test_n_head))
print("bidder={},ok!".format(n))


