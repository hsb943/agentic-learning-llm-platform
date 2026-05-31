Self-Attention vs Cross-Attention

Assumptions and Notation
This document assumes a foundational understanding of linear algebra, matrix operations, and basic neural network concepts. All vectors are column vectors unless explicitly stated as row vectors within matrix definitions.

Variables and Dimensions:
*   X: Input sequence 1. X ∈ ℝ^(n × d_model)
*   Y: Input sequence 2. Y ∈ ℝ^(m × d_model)
*   n: Length of sequence X (number of tokens/elements).
*   m: Length of sequence Y (number of tokens/elements).
*   d_model: Dimensionality of the input and output representations (embedding dimension).
*   h: Number of attention heads in Multi-Head Attention.
*   d_k: Dimensionality of the Query and Key vectors for a single head. d_k = d_model / h.
*   d_v: Dimensionality of the Value vectors for a single head. d_v = d_model / h.
*   W_Q: Query projection matrix. W_Q ∈ ℝ^(d_model × d_k) for a single head, or ℝ^(d_model × (h * d_k)) for concatenated heads.
*   W_K: Key projection matrix. W_K ∈ ℝ^(d_model × d_k) for a single head, or ℝ^(d_model × (h * d_k)) for concatenated heads.
*   W_V: Value projection matrix. W_V ∈ ℝ^(d_model × d_v) for a single head, or ℝ^(d_model × (h * d_v)) for concatenated heads.
*   W_O: Output projection matrix for Multi-Head Attention. W_O ∈ ℝ^((h * d_v) × d_model).
*   Q: Query matrix. Q ∈ ℝ^(n × d_k) (for self-attention) or Q ∈ ℝ^(n × d_k) (for cross-attention).
*   K: Key matrix. K ∈ ℝ^(n × d_k) (for self-attention) or K ∈ ℝ^(m × d_k) (for cross-attention).
*   V: Value matrix. V ∈ ℝ^(n × d_v) (for self-attention) or V ∈ ℝ^(m × d_v) (for cross-attention).
*   A: Attention scores matrix. A ∈ ℝ^(n × n) (for self-attention) or A ∈ ℝ^(n × m) (for cross-attention).
*   P: Attention weights matrix (after softmax). P ∈ ℝ^(n × n) (for self-attention) or P ∈ ℝ^(n × m) (for cross-attention).
*   Z: Output of a single attention head. Z ∈ ℝ^(n × d_v).
*   Z_multihead: Concatenated output of all attention heads. Z_multihead ∈ ℝ^(n × (h * d_v)).
*   Output: Final output of Multi-Head Attention. Output ∈ ℝ^(n × d_model).

Core Concepts and Mathematical Foundations
The attention mechanism computes a weighted sum of "value" vectors, where the weights are determined by the similarity between a "query" vector and a set of "key" vectors. This allows a model to focus on relevant parts of an input sequence or another sequence.

Formal Definition of Scaled Dot-Product Attention:
Given a Query matrix Q, a Key matrix K, and a Value matrix V, the scaled dot-product attention is defined as:
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V

Geometric Interpretation:
The term Q K^T computes dot products between each query vector (row of Q) and each key vector (column of K^T, which is a row of K). The dot product measures the cosine similarity between two vectors if they are L2-normalized, or a general measure of alignment. A larger dot product indicates higher similarity or relevance. The division by sqrt(d_k) scales these dot products to prevent the softmax function from saturating, which can lead to vanishing gradients.

Probabilistic Interpretation:
The softmax function transforms the raw attention scores into a probability distribution. For each query, the resulting row in P (softmax((Q K^T) / sqrt(d_k))) represents a discrete probability distribution over the corresponding key-value pairs. Each element P_ij indicates the probability that the i-th query attends to the j-th key/value. The output of the attention mechanism is then a weighted average of the value vectors, where the weights are these probabilities. This allows the model to "select" and combine information from the value vectors based on their relevance to the query.

Dimensional Reasoning:
*   Q ∈ ℝ^(n × d_k), K ∈ ℝ^(m × d_k), V ∈ ℝ^(m × d_v)
*   K^T ∈ ℝ^(d_k × m)
*   Q K^T: (n × d_k) * (d_k × m) = (n × m). This matrix contains the raw attention scores.
*   (Q K^T) / sqrt(d_k): Remains (n × m).
*   softmax((Q K^T) / sqrt(d_k)): Applied row-wise, so each row sums to 1. Resulting matrix P ∈ ℝ^(n × m).
*   P V: (n × m) * (m × d_v) = (n × d_v). This is the output of a single attention head.

Multi-Head Attention:
Multi-Head Attention extends the single attention mechanism by running h attention functions in parallel. Each head learns different linear projections (W_Q, W_K, W_V) for Q, K, and V, allowing it to attend to different parts of the input or different aspects of the information. The outputs from these h heads are then concatenated and linearly projected (W_O) to produce the final output, which typically has the same d_model dimensionality as the input.
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)

Mechanism and Formal Derivation

Self-Attention
Self-attention allows a sequence to attend to itself, enabling each element in the sequence to gather information from all other elements (including itself) in the same sequence. This captures internal dependencies and relationships within the sequence.

Derivation Steps for Self-Attention (Single Head):
1.  **Input Representation**: An input sequence X ∈ ℝ^(n × d_model) is provided. Each row x_i ∈ ℝ^(1 × d_model) represents a token embedding.
2.  **Linear Projections for Query, Key, Value**: The input sequence X is linearly transformed into Query (Q), Key (K), and Value (V) matrices using distinct projection matrices W_Q, W_K, W_V.
    *   Q = X W_Q. Q ∈ ℝ^(n × d_model) * ℝ^(d_model × d_k) = ℝ^(n × d_k).
    *   K = X W_K. K ∈ ℝ^(n × d_model) * ℝ^(d_model × d_k) = ℝ^(n × d_k).
    *   V = X W_V. V ∈ ℝ^(n × d_model) * ℝ^(d_model × d_v) = ℝ^(n × d_v).
    *   Here, the source for Q, K, and V is the same sequence X.
3.  **Compute Raw Attention Scores**: The dot product between each query vector and all key vectors is computed.
    *   A = Q K^T. A ∈ ℝ^(n × d_k) * ℝ^(d_k × n) = ℝ^(n × n).
    *   Each element A_ij represents the raw attention score of the i-th query token to the j-th key token.
4.  **Scale Attention Scores**: The raw attention scores are scaled by the square root of the key dimension, sqrt(d_k), to stabilize gradients.
    *   A_scaled = A / sqrt(d_k). A_scaled ∈ ℝ^(n × n).
5.  **Apply Softmax**: A softmax function is applied row-wise to A_scaled to obtain attention weights P. Each row of P sums to 1.
    *   P = softmax(A_scaled). P ∈ ℝ^(n × n).
    *   P_ij represents the normalized weight (probability) that the i-th token attends to the j-th token.
6.  **Compute Weighted Sum of Values**: The attention weights P are multiplied by the Value matrix V to produce the output Z.
    *   Z = P V. Z ∈ ℝ^(n × n) * ℝ^(n × d_v) = ℝ^(n × d_v).
    *   Each row z_i of Z is a weighted sum of all value vectors, where the weights are determined by how much the i-th query attends to each key.

Multi-Head Self-Attention:
For Multi-Head Self-Attention, steps 2-6 are performed h times in parallel, each with its own W_Q_i, W_K_i, W_V_i. The outputs Z_i ∈ ℝ^(n × d_v) from each head are concatenated along the feature dimension:
Z_multihead = Concat(Z_1, Z_2, ..., Z_h). Z_multihead ∈ ℝ^(n × (h * d_v)).
Finally, Z_multihead is projected back to d_model dimensions using W_O:
Output = Z_multihead W_O. Output ∈ ℝ^(n × (h * d_v)) * ℝ^((h * d_v) × d_model) = ℝ^(n × d_model).

Cross-Attention
Cross-attention allows a sequence (the query sequence) to attend to a different sequence (the key-value sequence). This is typically used to integrate information from a source sequence into a target sequence, such as in encoder-decoder architectures where the decoder queries the encoder's output.

Derivation Steps for Cross-Attention (Single Head):
1.  **Input Representations**: Two distinct input sequences are provided:
    *   Query sequence X ∈ ℝ^(n × d_model).
    *   Key-Value sequence Y ∈ ℝ^(m × d_model).
2.  **Linear Projections for Query**: The query sequence X is linearly transformed into the Query (Q) matrix.
    *   Q = X W_Q. Q ∈ ℝ^(n × d_model) * ℝ^(d_model × d_k) = ℝ^(n × d_k).
3.  **Linear Projections for Key, Value**: The key-value sequence Y is linearly transformed into the Key (K) and Value (V) matrices.
    *   K = Y W_K. K ∈ ℝ^(m × d_model) * ℝ^(d_model × d_k) = ℝ^(m × d_k).
    *   V = Y W_V. V ∈ ℝ^(m × d_model) * ℝ^(d_model × d_v) = ℝ^(m × d_v).
    *   Here, Q originates from X, while K and V originate from Y.
4.  **Compute Raw Attention Scores**: The dot product between each query vector (from X) and all key vectors (from Y) is computed.
    *   A = Q K^T. A ∈ ℝ^(n × d_k) * ℝ^(d_k × m) = ℝ^(n × m).
    *   Each element A_ij represents the raw attention score of the i-th query token (from X) to the j-th key token (from Y).
5.  **Scale Attention Scores**: The raw attention scores are scaled by sqrt(d_k).
    *   A_scaled = A / sqrt(d_k). A_scaled ∈ ℝ^(n × m).
6.  **Apply Softmax**: A softmax function is applied row-wise to A_scaled to obtain attention weights P. Each row of P sums to 1.
    *   P = softmax(A_scaled). P ∈ ℝ^(n × m).
    *   P_ij represents the normalized weight (probability) that the i-th query token (from X) attends to the j-th key token (from Y).
7.  **Compute Weighted Sum of Values**: The attention weights P are multiplied by the Value matrix V (from Y) to produce the output Z.
    *   Z = P V. Z ∈ ℝ^(n × m) * ℝ^(m × d_v) = ℝ^(n × d_v).
    *   Each row z_i of Z is a weighted sum of value vectors from Y, where the weights are determined by how much the i-th query token from X attends to each key token from Y.

Multi-Head Cross-Attention:
Similar to self-attention, steps 2-7 are performed h times in parallel, each with its own W_Q_i, W_K_i, W_V_i. The outputs Z_i ∈ ℝ^(n × d_v) are concatenated:
Z_multihead = Concat(Z_1, Z_2, ..., Z_h). Z_multihead ∈ ℝ^(n × (h * d_v)).
Finally, Z_multihead is projected back to d_model dimensions using W_O:
Output = Z_multihead W_O. Output ∈ ℝ^(n × (h * d_v)) * ℝ^((h * d_v) × d_model) = ℝ^(n × d_model).

Computational and Complexity Analysis

The computational complexity of attention mechanisms is dominated by matrix multiplications. We consider the number of floating-point operations (FLOPs).

Time Complexity (per head):
1.  **Linear Projections**:
    *   Self-Attention: X W_Q, X W_K, X W_V. Each is O(n * d_model * d_k) or O(n * d_model * d_v). Total: O(n * d_model * (d_k + d_k + d_v)).
    *   Cross-Attention: X W_Q (O(n * d_model * d_k)), Y W_K (O(m * d_model * d_k)), Y W_V (O(m * d_model * d_v)). Total: O((n + m) * d_model * d_k + m * d_model * d_v).
2.  **Query-Key Dot Product (Q K^T)**:
    *   Self-Attention: Q ∈ ℝ^(n × d_k), K^T ∈ ℝ^(d_k × n). Result is ℝ^(n × n). Complexity: O(n * n * d_k).
    *   Cross-Attention: Q ∈ ℝ^(n × d_k), K^T ∈ ℝ^(d_k × m). Result is ℝ^(n × m). Complexity: O(n * m * d_k).
3.  **Softmax**: Applied to a matrix of size (n × n) or (n × m). Complexity: O(n^2) or O(n * m).
4.  **Attention Weights * Values (P V)**:
    *   Self-Attention: P ∈ ℝ^(n × n), V ∈ ℝ^(n × d_v). Result is ℝ^(n × d_v). Complexity: O(n * n * d_v).
    *   Cross-Attention: P ∈ ℝ^(n × m), V ∈ ℝ^(m × d_v). Result is ℝ^(n × d_v). Complexity: O(n * m * d_v).

Total Time Complexity (for Multi-Head Attention, assuming d_k = d_v = d_model / h):
*   **Self-Attention**:
    *   Projections: O(n * d_model^2).
    *   Q K^T: O(h * n^2 * d_k) = O(h * n^2 * (d_model/h)) = O(n^2 * d_model).
    *   P V: O(h * n^2 * d_v) = O(h * n^2 * (d_model/h)) = O(n^2 * d_model).
    *   Output Projection: O(n * (h * d_v) * d_model) = O(n * d_model^2).
    *   Overall: O(n^2 * d_model + n * d_model^2). For typical transformer settings where n > d_model, this is dominated by O(n^2 * d_model).
*   **Cross-Attention**:
    *   Projections: O((n + m) * d_model^2).
    *   Q K^T: O(h * n * m * d_k) = O(h * n * m * (d_model/h)) = O(n * m * d_model).
    *   P V: O(h * n * m * d_v) = O(h * n * m * (d_model/h)) = O(n * m * d_model).
    *   Output Projection: O(n * d_model^2).
    *   Overall: O(n * m * d_model + (n + m) * d_model^2). For typical settings where n, m > d_model, this is dominated by O(n * m * d_model).

Memory Complexity (for Multi-Head Attention, assuming d_k = d_v = d_model / h):
The dominant memory cost comes from storing the Q, K, V matrices and the attention scores matrix.
*   **Self-Attention**:
    *   Q, K, V: Each is O(n * d_model). Total O(n * d_model).
    *   Attention scores (A or P): O(n * n).
    *   Overall: O(n^2 + n * d_model).
*   **Cross-Attention**:
    *   Q: O(n * d_model). K, V: O(m * d_model). Total O((n + m) * d_model).
    *   Attention scores (A or P): O(n * m).
    *   Overall: O(n * m + (n + m) * d_model).

Effect of Scaling Key Parameters:
*   **Sequence Lengths (n, m)**: Self-attention scales quadratically with n, making it computationally expensive for very long sequences. Cross-attention scales linearly with n and m, but quadratically with their product. If n and m are similar, the scaling is effectively quadratic. If one sequence is much shorter (e.g., n << m), cross-attention can be more efficient than self-attention on the longer sequence.
*   **Model Dimension (d_model)**: Both scale linearly with d_model in the dominant terms (Q K^T, P V) when d_k, d_v are proportional to d_model/h. However, the projection layers contribute O(d_model^2