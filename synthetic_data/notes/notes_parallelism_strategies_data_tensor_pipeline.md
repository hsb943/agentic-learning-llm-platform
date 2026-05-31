Title
Parallelism Strategies (Data, Tensor, Pipeline)

Assumptions and Notation
This document assumes a distributed computing environment consisting of P identical computational devices (e.g., GPUs), interconnected by a high-bandwidth, low-latency network. Each device p ∈ {1, ..., P} has local memory M_p. The model is a deep neural network, potentially a Transformer-like architecture, composed of L sequential layers.

Key variables and dimensions:
*   N: Global batch size for training.
*   S: Sequence length of input tokens.
*   D_model: Hidden dimension or model dimension.
*   D_ff: Feed-forward network inner dimension, typically D_ff = 4 * D_model.
*   H: Number of attention heads.
*   D_head: Dimension of each attention head, typically D_head = D_model / H.
*   L: Total number of layers in the model.
*   P: Total number of parallel devices.
*   K: Number of pipeline stages (for Pipeline Parallelism).
*   M: Number of micro-batches (for Pipeline Parallelism).
*   θ: Set of all model parameters.
*   X ∈ ℝ^(N × S × D_model): Input tensor to the model.
*   W ∈ ℝ^(D_in × D_out): General weight matrix for a linear transformation.
*   A ∈ ℝ^(N_batch × S × D_model): Activation tensor.
*   ∇θ: Gradient tensor with respect to parameters θ.
*   F_comp: Computational cost of a single forward or backward pass for a specific operation or layer.
*   C_comm: Communication cost for a specific operation (e.g., all-reduce, all-gather, point-to-point).

Core Concepts and Mathematical Foundations
Parallelism strategies in deep learning aim to distribute the computational workload and memory footprint of training large models across multiple devices. This is primarily driven by two constraints: the inability of a single device to hold the entire model or a sufficiently large batch in memory, and the desire to reduce training time.

1.  Data Parallelism (DP):
    *   Concept: The model parameters θ are replicated on each device. The input data batch X is sharded across devices. Each device performs a full forward and backward pass on its local data shard. Gradients are then aggregated across all devices to update the model parameters synchronously.
    *   Mathematical Foundation: Let F(X, θ) be the model's forward pass and L(Y, Y_true) be the loss function. The objective is to minimize E[L(F(X, θ), Y_true)] over the entire dataset. For a mini-batch X, the global gradient is ∇θ = (1/N) Σ_{i=1}^N ∇θ_i, where ∇θ_i is the gradient from a single sample. In DP, each device p receives a sub-batch X_p ∈ ℝ^((N/P) × S × D_model) and computes a local gradient ∇θ_p = (1/(N/P)) Σ_{x_i ∈ X_p} ∇θ_i. The global gradient is then obtained by summing these local gradients: ∇θ_global = (1/P) Σ_{p=1}^P ∇θ_p. This summation is typically performed via an all-reduce communication primitive.
    *   Dimensional Reasoning: Input batch X ∈ ℝ^(N × S × D_model) is partitioned into P sub-batches X_p ∈ ℝ^((N/P) × S × D_model). Model parameters θ remain full-sized on each device.

2.  Tensor Parallelism (TP) / Model Parallelism within a layer:
    *   Concept: Individual operations within a layer (e.g., matrix multiplications, attention mechanisms) are sharded across devices. This means that parts of the weight matrices and corresponding activation tensors are distributed. Each device computes only a portion of the output tensor, and communication is required to reconstruct the full tensor or aggregate partial results.
    *   Mathematical Foundation: Consider a linear transformation Y = XW + B, where X ∈ ℝ^(N × D_in), W ∈ ℝ^(D_in × D_out), B ∈ ℝ^(D_out), and Y ∈ ℝ^(N × D_out).
        *   **Column-wise partitioning of W**: W is partitioned horizontally into P blocks: W = [W_1 | W_2 | ... | W_P], where W_p ∈ ℝ^(D_in × (D_out/P)). Each device p computes Y_p = XW_p + B_p, where B_p is the corresponding partition of the bias. The full output Y is then Y = [Y_1 | Y_2 | ... | Y_P]. This requires an all-gather operation for Y.
        *   **Row-wise partitioning of W**: W is partitioned vertically into P blocks: W = [W_1^T; W_2^T; ...; W_P^T]^T, where W_p ∈ ℝ^((D_in/P) × D_out). The input X must also be partitioned column-wise: X = [X_1 | X_2 | ... | X_P], where X_p ∈ ℝ^(N × (D_in/P)). Each device p computes Y_p = X_p W_p. The full output Y is then Y = Σ_{p=1}^P Y_p. This requires an all-reduce operation for Y.
    *   Dimensional Reasoning: For column-wise TP, W ∈ ℝ^(D_in × D_out) is split into W_p ∈ ℝ^(D_in × (D_out/P)). Input X ∈ ℝ^(N × D_in) is replicated. Output Y_p ∈ ℝ^(N × (D_out/P)). For row-wise TP, W ∈ ℝ^(D_in × D_out) is split into W_p ∈ ℝ^((D_in/P) × D_out). Input X ∈ ℝ^(N × D_in) is split into X_p ∈ ℝ^(N × (D_in/P)). Output Y_p ∈ ℝ^(N × D_out).

3.  Pipeline Parallelism (PP):
    *   Concept: The model's layers are partitioned into K sequential stages, with each stage assigned to a different device or group of devices. The input batch is further divided into M smaller micro-batches. These micro-batches are then processed in a pipeline fashion, allowing different stages to operate concurrently on different micro-batches.
    *   Mathematical Foundation: Let the model be a sequence of L layers, F = F_L ∘ ... ∘ F_1. This sequence is partitioned into K stages, F = G_K ∘ ... ∘ G_1, where G_k = F_{l_k_end} ∘ ... ∘ F_{l_k_start} represents the computation performed by stage k. Each stage k receives activations A_{k-1} from stage k-1, computes G_k(A_{k-1}), and sends A_k to stage k+1. The input batch X is divided into M micro-batches X_m.
    *   Dimensional Reasoning: Model layers L are partitioned into K stages. Each stage k holds a subset of layers and their parameters. Activations A_k ∈ ℝ^((N/M) × S × D_model) are passed between stages.

Mechanism and Formal Derivation

1.  Data Parallelism (DP)
    1.  **Model Replication**: Each of the P devices receives and stores a complete copy of the model parameters θ. This means device p holds θ_p = θ.
    2.  **Data Sharding**: The global input batch X ∈ ℝ^(N × S × D_model) is deterministically partitioned into P disjoint sub-batches X_p ∈ ℝ^((N/P) × S × D_model). Each device p receives X_p.
    3.  **Local Forward Pass**: Each device p independently performs a forward pass using its local sub-batch X_p and its local model parameters θ_p, computing local outputs Y_p = F(X_p, θ_p).
    4.  **Local Backward Pass**: Based on the local outputs Y_p and corresponding labels, each device p computes the local gradients ∇θ_p with respect to its parameters θ_p.
    5.  **Gradient Aggregation (All-Reduce)**: All P devices participate in an all-reduce operation on their local gradients ∇θ_p. This operation sums the local gradients across all devices and broadcasts the global sum back to all devices. The resulting global gradient is ∇θ_global = (1/P) Σ_{p=1}^P ∇θ_p.
    6.  **Parameter Update**: Each device p updates its local parameters θ_p using the globally aggregated gradient ∇θ_global and the chosen optimizer (e.g., θ_p ← θ_p - η * ∇θ_global, where η is the learning rate). Since all devices receive the same ∇θ_global, their parameters remain synchronized.

2.  Tensor Parallelism (TP) - Column-wise Partitioning of Linear Layers
    Consider a linear layer Y = XW + B, where X ∈ ℝ^(N × D_in), W ∈ ℝ^(D_in × D_out), B ∈ ℝ^(D_out), and Y ∈ ℝ^(N × D_out).
    1.  **Parameter Partitioning**: The weight matrix W is partitioned column-wise into P blocks: W = [W_1 | W_2 | ... | W_P], where W_p ∈ ℝ^(D_in × (D_out/P)). Similarly, the bias vector B is partitioned: B = [B_1 | B_2 | ... | B_P], where B_p ∈ ℝ^(D_out/P). Each device p stores W_p and B_p. The input tensor X ∈ ℝ^(N × D_in) is replicated across all P devices.
    2.  **Local Forward Computation**: Each device p independently computes a partial output Y_p = XW_p + B_p. The dimension of Y_p is ℝ^(N × (D_out/P)).
    3.  **Output Aggregation (All-Gather)**: To obtain the full output Y ∈ ℝ^(N × D_out) for subsequent layers, an all-gather operation is performed. Each device p sends its Y_p to all other devices, and all devices concatenate these partial outputs to form Y = [Y_1 | Y_2 | ... | Y_P].
    4.  **Backward Pass - Gradient of Output**: Let ∇Y ∈ ℝ^(N × D_out) be the gradient received from the subsequent layer. This ∇Y is partitioned column-wise into ∇Y_p ∈ ℝ^(N × (D_out/P)) and sent to device p.
    5.  **Backward Pass - Gradient of Weights and Bias**: Each device p computes the local gradients for its parameters: ∇W_p = X^T ∇Y_p and ∇B_p = Σ_{i=1}^N (∇Y_p)_i.
    6.  **Backward Pass - Gradient of Input**: Each device p computes a partial gradient for the input: ∇X_p = ∇Y_p W_p^T. Since X was replicated, the full gradient ∇X is the sum of these partial gradients. An all-reduce operation is performed on ∇X_p to obtain ∇X = Σ_{p=1}^P ∇X_p. This ∇X is then passed to the preceding layer.

3.  Pipeline Parallelism (PP)
    Consider a model with L layers, partitioned into K stages, where stage k (on device k) contains layers F_{l_k_start} to F_{l_k_end}. The global batch X ∈ ℝ^(N × S × D_model) is divided into M micro-batches X_m ∈ ℝ^((N/M) × S × D_model).
    1.  **Model Partitioning**: The L layers of the model are divided into K sequential stages. Each stage k is assigned to a distinct device (or group of devices). Device k stores only the parameters for layers within stage k.
    2.  **Micro