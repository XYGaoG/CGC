from scr.para import *
from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *

args = para()
args.result_path =  f'./results/'
args = create_folder(args)
args = device_setting(args)
seed_everything(args.seed)

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

##hyper para
args = hyperpara(args) if args.generate_adj == 1 else hyperpara_noadj(args)

## cond data
graph_file = args.folder+f'{args.dataset_name}_{args.ratio}.pt' if args.generate_adj == 1 else args.folder+f'{args.dataset_name}_noadj_{args.ratio}.pt'
# if os.path.exists(graph_file):
#     graph = torch.load(graph_file, map_location= args.device)
# else:
begin = time.time()
args, label_cond = generate_labels_syn(args, data)
H = conv_graph_multi(args, data)
model = linear_model(args, H, data, data_test)
H_aug, y_aug, conf = data_assessment(args, data, model, H)
M_norm = mask_generation_conf(H_aug, y_aug, args, 'spectral', conf)
h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))
if args.generate_adj == 1:
    a = get_adj(h, args.adj_T)
    x = get_feature(a, h, args.alpha)
    graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
else:
    graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))

args.cond_time = time.time()-begin
print('Condensation time:',  f'{args.cond_time:.3f}', 's')
print('#edges:', int(torch.sum(a).item())) if args.generate_adj == 1 else print('No adj')
print('#training labels:', data.train_mask.sum().item())
print('#augmented labels:', len(H_aug))
args.changed_label = len(H_aug)-data.train_mask.sum().item()
# torch.save(graph, graph_file)

# model training
graph=graph.to(args.device)
acc= []
for repeat in range(args.repeat): 
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    args.test_gnn = model.__class__.__name__
    acc.append(model_training(model, args, data, graph, data_val, data_test))
result_record(args, acc)