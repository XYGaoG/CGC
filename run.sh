#!/bin/bash 
gpu_=0

data_=cora
for r_ in 0.013 0.026 0.052
do
    python main.py --generate_adj 0 --gpu $gpu_ --dataset $data_ --ratio $r_
    python main.py --generate_adj 1 --gpu $gpu_ --dataset $data_ --ratio $r_
done

data_=citeseer
for r_ in 0.009 0.018 0.036
do
    python main.py --generate_adj 0 --gpu $gpu_ --dataset $data_ --ratio $r_
    python main.py --generate_adj 1 --gpu $gpu_ --dataset $data_ --ratio $r_
done

data_=flickr
for r_ in 0.001 0.005 0.01
do
    python main.py --generate_adj 0 --gpu $gpu_ --dataset $data_ --ratio $r_
    python main.py --generate_adj 1 --gpu $gpu_ --dataset $data_ --ratio $r_
done

data_=arxiv
for r_ in 0.0005 0.0025 0.005
do
    python main.py --generate_adj 0 --gpu $gpu_ --dataset $data_ --ratio $r_
    python main.py --generate_adj 1 --gpu $gpu_ --dataset $data_ --ratio $r_
done

data_=reddit
for r_ in 0.0005 0.001 0.002
do
    python main.py --generate_adj 0 --gpu $gpu_ --dataset $data_ --ratio $r_
    python main.py --generate_adj 1 --gpu $gpu_ --dataset $data_ --ratio $r_
done
