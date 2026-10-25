# Machine Learning & Graph Neural Network Model Specification

## Zero-Day GraphIDS

---

## 1. Problem Formulation & Objective

Zero-Day malicious traffic detection is formalized as an **Unsupervised Graph Outlier Detection** problem.
Given a continuous stream of network flows $\mathcal{F} = \{f_1, f_2, \dots, f_T\}$, our objective is to map temporal windows of flows into a graph structure $G = (\mathcal{V}, \mathcal{E}, X)$ and learn a mapping function:
$$f_{\theta}: G \to \mathbb{R}^{D_{\text{embed}}}$$
such that:
1. Benign network communications collapse into a tightly constrained representation manifold $\mathcal{M}_{\text{normal}}$.
2. Novel, previously unencountered attack behaviors (Zero-Day exploits) generate out-of-manifold representations with high reconstruction or distance loss.

---

## 2. Graph Construction Pipeline

### Node Definition
Each individual network flow record within a temporal chunk window of size $W$ is treated as a node $v_i \in \mathcal{V}$.
The node feature vector $x_i \in \mathbb{R}^D$ is formed by standardized continuous features:
- `Flow Duration`
- `Total Fwd Packets` & `Total Backward Packets`
- `Total Length of Fwd Packets` & `Total Length of Bwd Packets`
- `Flow Bytes/s` & `Flow Packets/s`
- `Fwd Packet Length Mean` & `Bwd Packet Length Mean`
- `Packet Length Mean`

### Edge Generation ($k$-Nearest Neighbors Similarity Graph)
For each pair of flows $(v_i, v_j)$ in the chunk, the Euclidean feature distance $d(x_i, x_j) = \|x_i - x_j\|_2$ is computed.
An undirected edge $e_{ij} \in \mathcal{E}$ is instantiated if $v_j$ is among the $k$-nearest neighbors of $v_i$.
Edge weights are computed via Gaussian radial basis kernels:
$$w_{ij} = \exp\left(-\frac{d(x_i, x_j)^2}{2}\right)$$

---

## 3. GraphSAGE Message Passing Architecture

The encoder uses two GraphSAGE message-passing layers with mean neighbor aggregation:

```text
Input Node Features X [N x D_in]
        │
        ▼
   [ SAGEConv Layer 1 ] -> maps D_in to D_hidden (64)
   [ LayerNorm ]
   [ ReLU Activation ]
   [ Dropout p=0.1 ]
        │
        ▼
   [ SAGEConv Layer 2 ] -> maps D_hidden to D_embed (32)
   [ LayerNorm ]
   [ ReLU Activation ]
        │
        ├────────────────────────────────┐
        ▼                                ▼
[ Node Embeddings H ]           [ Graph Readout Pooling ]
     [N x D_embed]              Concat(MeanPool, MaxPool)
        │                                │
        ▼                                ▼
[ Decoder MLP Head ]           [ Graph Embedding h_G ]
Reconstructs original X              [1 x 64]
```

---

## 4. Self-Supervised Optimization Objectives

### 1. Masked Node Feature Reconstruction Loss
Forces the GNN encoder to capture intra-flow semantic distributions:
$$\mathcal{L}_{\text{recon}} = \frac{1}{|\mathcal{V}|} \sum_{i=1}^{|\mathcal{V}|} \|x_i - \hat{x}_i\|_2^2$$

### 2. Deep Graph Infomax (DGI) Contrastive Loss
Maximizes mutual information between local node embeddings $h_i$ and the global graph representation $s = \sigma(h_G)$.
A corrupted graph $\tilde{G} = (\mathcal{V}, \mathcal{E}, \tilde{X})$ is constructed by row-wise shuffling node features across the batch.
The bilinear discriminator $\mathcal{D}(h, s) = \sigma(h^T W s)$ scores correspondence:
$$\mathcal{L}_{\text{DGI}} = - \frac{1}{2N} \left[ \sum_{i=1}^N \log \mathcal{D}(h_i, s) + \sum_{j=1}^N \log (1 - \mathcal{D}(\tilde{h}_j, s)) \right]$$

### 3. Composite Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + 0.5 \cdot \mathcal{L}_{\text{DGI}}$$

---

## 5. Anomaly Scoring & Adaptive Thresholding

### Regularized Mahalanobis Distance
Given benign training graph representations $H_{\text{train}} = \{h_{G, 1}, \dots, h_{G, M}\}$, compute:
$$\mu = \frac{1}{M} \sum_{m=1}^M h_{G, m}$$
$$\Sigma_{\text{reg}} = \text{LedoitWolf}(H_{\text{train}}) + \epsilon I$$
For test graph embedding $h_{G, \text{test}}$:
$$D_M(h_{G, \text{test}}) = \sqrt{(h_{G, \text{test}} - \mu)^T \Sigma_{\text{reg}}^{-1} (h_{G, \text{test}} - \mu)}$$

### Threshold Calibration (Strictly on Validation Split)
Using validation anomaly scores $S_{\text{val}} = \{D_M(h_{G, v})\}$:
$$\tau_{\text{calibrated}} = \text{Percentile}\left(S_{\text{val}}, (1 - \alpha) \cdot 100\right)$$
where contamination parameter $\alpha \in [0.01, 0.05]$.

### False-Positive vs False-Negative Trade-Off
- Lower $\alpha$ (e.g. 0.01) sets a conservative higher threshold, minimizing false alarms in operational networks.
- Higher $\alpha$ (e.g. 0.05) lowers the boundary, increasing sensitivity to stealthy, low-volume zero-day anomalies.
