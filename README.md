# Network-Science

# What is this? 
Here is a repository that examines the conceptual validity of specific centrality measures as proxies of modular behaviour. In particular, betweenness centrality and in-degree centrality is used to identify nodes that are not on the shortest path of many nodes, but that receive information from many nodes themselves, acting as information sinks. The question is whether such nodes integrate high-level information in a modular way, i.e. if removing them and their strongest incoming nodes results in a strong, class-level effect. 
The experiment trains a neural network on a few MNIST digits, evaluates the centralities of the network, and automatically ablates them from the network. Then, class-accuracies are reported per digit. 

## Installation
#### 0. install requirements.txt.
pip install -r requirements.txt

#### 1. Train a neural network, e.g. with 
python train.py --epochs 20 --batch-size 128 --lr 0.001 --hidden-layers 8,8,8 --save-dir ./mlp

#### 2. Run network_analysis.py
python network_analysis.py

#### 3. Run are_modules_indegree_nobetween.py
python are_modules_indegree_nobetween.py

## Reading the output: 
The experiment runs for multiple nodes that fit the criteria. Each detected node has their strongest incoming nodes imputed, and then the experiment reports the classification accuracies per digit. 

