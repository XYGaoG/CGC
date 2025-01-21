import argparse

def para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="reddit", help= 
    'cora, citeseer, pubmed, arxiv, flickr, reddit, products')
    parser.add_argument('--ratio', type=float, default= 0.001)
    # cora 0.026 citeseer 0.018  arxiv 0.0025 flickr 0.005  reddit 0.001
    parser.add_argument('--raw_data_dir', type=str, default="./data/")
    parser.add_argument('--result_path', type=str, default="./results")
    parser.add_argument('--cond_folder', type=str, default="./cond_graph/")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--test_gnn', type=str, default="GCN")
    parser.add_argument('--test_gnn_idx', type=int, default=0)

    parser.add_argument('--kernel', type=str, default="gcn", help='gcn, ppr, heat, cheby, sage')
    parser.add_argument('--conv_depth', type=int, default=2, help= 'number of conv depth of the original graph')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clustering', type=str, default='spectral', help='spectral, kmeans')
    parser.add_argument('--generate_adj', type=int, default=1, help='generate the condensed graph')
    parser.add_argument('--adj_T', type=float, default=0.95, help='threshold for condensed graph')
    parser.add_argument('--alpha', type=float, default=3, help='weight for smooth loss')
    parser.add_argument('--tau', type=float, default=10, help='the denoising ratio')
    parser.add_argument('--aug_ratio', type=float, default=0.55, help='the augmentation ratio')

    args = parser.parse_args()
    return args



def hyperpara(args):
    if args.dataset_name == 'cora':
        
        if args.ratio == 0.013:
            args.epoch=1200
            args.dropout = 0.8
            args.adj_T = 0.8
            args.alpha = 1
            args.tau = 0.5
            args.aug_ratio = 0.5

        if args.ratio == 0.026:
            args.dropout = 0.8
            args.adj_T = 0.8
            args.alpha = 7
            args.tau = 1        
            args.aug_ratio = 0.5

        if args.ratio == 0.052:
            args.dropout = 0.8  
            args.adj_T = 0.9    
            args.alpha = 7      
            args.tau = 0.1  
            args.aug_ratio = 0.3



    if args.dataset_name == 'citeseer':

        if args.ratio == 0.009:
            args.adj_T = 0.8
            args.alpha = 2
            args.tau = 0.05 
            args.aug_ratio = 0.6

        if args.ratio == 0.018:
            args.adj_T = 0.8
            args.alpha = 1
            args.tau = 0.05 
            args.aug_ratio = 0.1

        if args.ratio == 0.036:
            args.adj_T = 0.75
            args.alpha = 2.
            args.tau = 0.5      
            args.aug_ratio = 0.5


    if args.dataset_name == 'arxiv':
        
        if args.ratio == 0.0005:
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 5
            args.aug_ratio = 0.4

        if args.ratio == 0.0025:
            args.weight_decay=5e-3
            args.epoch = 1000
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 15
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.weight_decay=5e-3
            args.lr = 0.01
            args.dropout = 0.4
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 1
            args.aug_ratio = 3

    if args.dataset_name == 'flickr':

        if args.ratio == 0.001:
            args.lr=0.001
            args.epoch = 1000
            args.dropout = 0.47
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.1
            args.aug_ratio = 0.5

        if args.ratio == 0.005:
            args.dropout = 0.6
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.08
            args.aug_ratio = 0.3

        if args.ratio == 0.01:
            args.dropout = 0.9
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.01
            args.aug_ratio = 0.5

    if args.dataset_name == 'reddit':

        args.epoch = 1000
        if args.ratio == 0.0005:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.2

        if args.ratio == 0.001:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.55

        if args.ratio == 0.002:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.75

    return args


def hyperpara_noadj(args):
    if args.dataset_name == 'cora':

        if args.ratio == 0.013:
            args.dropout = 0.8
            args.tau = 0.7
            args.aug_ratio = 0.3

        if args.ratio == 0.026:
            args.dropout = 0.8
            args.tau = 10
            args.aug_ratio = 0.7

        if args.ratio == 0.052:
            args.dropout = 0.9      
            args.tau = 0.7
            args.aug_ratio = 0.3

    if args.dataset_name == 'citeseer':

        if args.ratio == 0.009:
            args.tau = 0.1 
            args.aug_ratio = 0.6

        if args.ratio == 0.018:
            args.lr = 0.02
            args.dropout = 0.0
            args.tau = 0.1
            args.aug_ratio = 0.8

        if args.ratio == 0.036:  
            args.weight_decay=0.08  
            args.tau = 10     
            args.aug_ratio = 0.7

    if args.dataset_name == 'arxiv':

        if args.ratio == 0.0005:
            args.epoch = 1000
            args.dropout = 0.7  
            args.tau = 0.3
            args.aug_ratio = 0.7

        if args.ratio == 0.0025:
            args.tau = 0.3
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.tau = 10
            args.aug_ratio = 0.8

    if args.dataset_name == 'flickr':

        if args.ratio == 0.001:
            args.dropout = 0.64 
            args.tau = 0.1 
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.dropout = 0.7
            args.tau = 0.08 
            args.aug_ratio = 0.5

        if args.ratio == 0.01:
            args.dropout = 0.9
            args.tau = 0.005
            args.aug_ratio = 0.6



    if args.dataset_name == 'reddit':

        if args.ratio == 0.0005:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.8

        if args.ratio == 0.001:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.2

        if args.ratio == 0.002:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.2


    return args
