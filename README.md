## Introduction

This is the Pytorch implementation of our paper:`Revenue-Optimal Reverse Auction for Task Allocation in Mobile Crowdsensing through Attention Transformer` .


## Requirements


* Python >= 3.7
* Pytorch 1.10.0
* Argparse
* Logging
* Tqdm
* Scipy

## Usage

### Generate the data

```bash
python generate_data.py
#You can set the number of users
```

### Train ReverseAuctionNet

```bash
#poi=6
# bidder=2
python 2x6.py

# bidder=4
python 4x6.py

# bidder=6
python 6x6.py

# bidder=8
python 8x6.py

# bidder=10
python 10x6.py

```

## Acknowledgement

Our code is built upon the implementation of <https://arxiv.org/abs/2201.12489>.

