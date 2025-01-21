import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import argparse
import os
import os.path as osp
import csv
import faiss
import time
import warnings
import json
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import expm
from ogb.nodeproppred import PygNodePropPredDataset
import gdown


from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD


from torch_geometric.datasets import CoraFull, Reddit2, Flickr, Planetoid, Reddit, WikipediaNetwork
from torch_geometric.utils import add_remaining_self_loops, to_undirected, subgraph, get_laplacian
from torch_geometric.utils.loop import remove_self_loops 
from torch_geometric.nn.dense.linear import Linear
# from torch_sparse import SparseTensor
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, SGConv, APPNP, GATConv
import torch_geometric.transforms as T
from torch_geometric.utils.augmentation import mask_feature, add_random_edge
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.transforms import KNNGraph, AddSelfLoops
from torch_geometric.nn import LabelPropagation
from torch_geometric.utils import homophily, add_self_loops, to_dense_adj, is_undirected
import torch.nn.init as init



