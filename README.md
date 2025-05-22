
# Neural network modularity

## Project Overview

This repository explores whether graph-theoretic centrality measures can serve as proxies for **modular processing** in neural networks.

Specifically, we investigate whether neurons with **high in-degree centrality** (receiving many strong inputs) but **low betweenness centrality** (not on many shortest paths) play a special integrative role -potentially acting as **modular bottlenecks**.

The core experiment:
- Trains a neural network on a subset of MNIST digits (e.g., 3, 5, 7).
- Constructs a directed graph of neurons and weighted connections.
- Identifies hidden neurons with high in-degree and low betweenness.
- Ablates each such neuron **and** its strongest incoming neurons.
- Evaluates how this targeted ablation impacts classification performance *per digit*.

---

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train a neural network (example below uses 3 hidden layers of 8 units each):
```bash
python train.py --epochs 20 --batch-size 128 --lr 0.001 --hidden-layers 8,8,8 --save-dir ./mlp
```
3. Analyze the trained network graph:
```bash
python network_analysis.py
```
4. Run the ablation experiment:
```bash
python are_modules_indegree_nobetween.py
```
## Understanding the Output

The ablation experiment runs over multiple candidate neurons that match the high-in, low-betweenness profile.

For each such neuron:

   -The neuron and its strongest upstream inputs (based on connection weights) are zeroed out.

   -The model is re-evaluated on test images.

  -Accuracy is reported per digit class, helping determine if the ablated unit participated in class-specific processing.

This allows you to ask:

    Do these centrally located “information sinks” support modular class representations?

## Notes

The graph construction treats neurons as nodes and absolute weight magnitude as edge weights.

Input neurons are excluded from candidate selection to avoid trivial input masking.

## Citation

If you use this codebase or find it helpful in your research, please cite the following repository:

Panagiotou, F (2025). Neural Network Modularity [Computer software]. GitHub. https://github.com/mknull/neural-network-modularity
