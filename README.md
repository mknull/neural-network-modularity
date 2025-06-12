
# Neural network modularity

## Project Overview

This repository explores whether specific graph-theoretic centrality measures can identify subnetworks resembling **modular processing** in neural networks.

Specifically, we investigate whether neurons with **high in-degree centrality** (receiving many strong inputs) but **low betweenness centrality** (not on many shortest paths) play a special integrative role -potentially acting as **modular bottlenecks**. The intuition is that there are many inputs that can belong to a particular class, but information is needed upstream **only if the module is required**. 

The experiment:
- Trains a neural network on a subset of MNIST digits (by default 3, 5, 7).
- Constructs a directed acyclic graph of neurons and weighted connections.
- Identifies neurons in the hidden layers with high in-degree and low betweenness.
- Ablates each such neuron **and** its strongest incoming neurons -input neurons are spared.
- Evaluates how this targeted ablation impacts classification performance per digit.

---

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train a neural network (example below uses 3 hidden layers of 8 units each):
```bash
python train.py --epochs 30 --batch-size 128 --lr 0.001 --hidden-layers 8,8,8 --save-dir ./mlp
```
3. Analyze the trained network graph:
```bash
python network_analysis.py
```
4. Run the experiment:
```bash
python are_modules_indegree_nobetween.py
```
## Understanding the Output

The ablation experiment runs over multiple candidate neurons that match the centrality profile, disabling the modules individually. 

Specifically, for each such neuron:

   -The neuron and its strongest incoming connections (based on connection weights) are zeroed out.

   -The model is re-evaluated on test images.

  -Accuracy is reported per digit class. Class-specific deteroriations have disabled modules processing specific digits.

In short, the experiment allows you to ask:

    Do these centrally located “information sinks” support modular class representations?

## Results
![Results of an ablation experiment](https://i.imgur.com/Gow1ezv.png)

## Notes

The graph construction treats neurons as nodes and *absolute weight magnitude* as edge weights.

Shortest paths are similarly calculated via strongest weights. 

Input neurons are excluded from candidate selection to avoid trivial input masking.

## Citation

If you use this codebase or find it helpful in your research, please cite the following repository:

Panagiotou, F (2025). Neural Network Modularity [Computer software]. GitHub. https://github.com/mknull/neural-network-modularity
