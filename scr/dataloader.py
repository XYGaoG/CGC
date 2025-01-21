from ast import arg
from scr.para import *
from scr.module import *
from scr.models import *



def get_dataset(args):

    if args.dataset_name in ["cora"]:
        dataset = Planetoid(args.raw_data_dir, 'cora')
        data = dataset[0]

    elif args.dataset_name in ['citeseer']:
        dataset = Planetoid(args.raw_data_dir, 'citeseer')
        data = dataset[0]

    elif args.dataset_name == "arxiv":
        dataset_str=args.raw_data_dir+'ogbn-arxiv/raw/'
        # adj
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        nnodes = adj_full.shape[0]

        adj_full = adj_full + adj_full.T
        adj_full[adj_full > 1] = 1

        # split
        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask = torch.zeros(nnodes, dtype=torch.bool)
        test_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # label
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # feat
        feat = np.load(dataset_str+'feats.npy')
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        dataset = Data(x=torch.FloatTensor(feat).float(), 
                        edge_index=torch.LongTensor(np.array(adj_full.nonzero())), 
                        y=torch.LongTensor(labels), 
                        train_mask = train_mask,  
                        val_mask = val_mask, 
                        test_mask = test_mask)
        transform = T.ToUndirected()
        data = transform(dataset)

    elif args.dataset_name == "reddit":
        dataset_str=args.raw_data_dir+'reddit/raw/'
        # adj
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        nnodes = adj_full.shape[0]

        # split
        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask = torch.zeros(nnodes, dtype=torch.bool)
        test_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # label
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # feat
        feat = np.load(dataset_str+'feats.npy')
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        dataset = Data(x=torch.FloatTensor(feat).float(), 
                        edge_index=torch.LongTensor(np.array(adj_full.nonzero())), 
                        y=torch.LongTensor(labels), 
                        train_mask = train_mask,  
                        val_mask = val_mask, 
                        test_mask = test_mask)
        transform = T.ToUndirected()
        dataset = transform(dataset)
        data = inductive_processing(dataset)

    elif args.dataset_name == "flickr":
        dataset_str=args.raw_data_dir+'flickr/raw/'
        # adj
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        nnodes = adj_full.shape[0]

        # split
        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask = torch.zeros(nnodes, dtype=torch.bool)
        test_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # label
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # feat
        feat = np.load(dataset_str+'feats.npy')
        # feat_train = feat[idx_train]
        # scaler = StandardScaler()
        # scaler.fit(feat_train)
        # feat = scaler.transform(feat)

        dataset = Data(x=torch.FloatTensor(feat).float(), 
                        edge_index=torch.LongTensor(np.array(adj_full.nonzero())), 
                        y=torch.LongTensor(labels), 
                        train_mask = train_mask,  
                        val_mask = val_mask, 
                        test_mask = test_mask)
        transform = T.ToUndirected()
        dataset = transform(dataset)
        data = inductive_processing(dataset)

    elif args.dataset_name == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=args.raw_data_dir+'pygdata')
        dataset.data.y.squeeze_()
        data = dataset[0]


    return data


def inductive_processing(data):

    edge_index,_ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.train_mask]
    y = data.y[data.train_mask]
    g_train = Data(x=x, y=y, edge_index=edge_index)
    g_train.train_mask = torch.ones(len(x), dtype=torch.bool)

    edge_index,_ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.val_mask]
    y = data.y[data.val_mask]
    g_val = Data(x=x, y=y, edge_index=edge_index)
    g_val.val_mask = torch.ones(len(x), dtype=torch.bool)

    edge_index,_ = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.test_mask]
    y = data.y[data.test_mask]
    g_test = Data(x=x, y=y, edge_index=edge_index)
    g_test.test_mask = torch.ones(len(x), dtype=torch.bool)

    return [g_train, g_val, g_test]


def process_labels(class_map, nnodes):
    """
    setup vertex property map for output classests
    """
    num_vertices = nnodes
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        nclass = num_classes
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[int(k)] = v
    else:
        class_arr = np.zeros(num_vertices, dtype=int)
        for k, v in class_map.items():
            class_arr[int(k)] = v
        class_arr = class_arr - class_arr.min()
        nclass = max(class_arr) + 1
    return class_arr


def set_dataset(args, datasets):
    if args.dataset_name in ['flickr', 'reddit']:
        data, data_val, data_test = datasets
        data, data_val, data_test = data.to(args.device), data_val.to(args.device), data_test.to(args.device)
    else:
        data = datasets.to(args.device)
        data_val, data_test = None, None
    args.num_class = int(data.y.max()+1)
    return args, data, data_val, data_test 
