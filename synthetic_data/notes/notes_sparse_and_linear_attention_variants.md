Title
Sparse and Linear Attention Variants

Assumptions and Notation
The following variables and dimensions are used consistently throughout this document:
*   n: Sequence length, representing the number of tokens in an input sequence.
*   d_model: Dimensionality of the input and output representations for each token.
*   d_k: Dimensionality of query (Q) and key (K) vectors. Typically d_k = d_model / h.
*   d_v: Dimensionality of value (V) vectors. Typically d_v = d_model / h.
*   h: Number of attention heads.
*   Q: Query matrix, Q ∈ ℝ^(n × d_k). Each row q_i ∈ ℝ^(d_k) is a query vector for the i-th token.
*   K: Key matrix, K ∈ ℝ^(n × d_k). Each row k_j ∈ ℝ^(d_k) is a key vector for the j-th token.
*   V: Value matrix, V ∈ ℝ^(n × d_v). Each row v_j ∈ ℝ^(d_v) is a value vector for the j-th token.
*   A: Attention score matrix, A ∈ ℝ^(n × n).
*   P: Attention probability matrix, P ∈ ℝ^(n × n).
*   O: Output matrix, O ∈ ℝ^(n × d_v).
*   M: Attention mask matrix, M ∈ ℝ^(n × n), with elements m_ij ∈ {0, 1, -∞}.
*   w: Window size for local attention, an integer.
*   d_f: Feature dimension for kernel functions in linear attention, d_f ∈ ℕ.
*   φ: Kernel function, φ: ℝ^(d_k) → ℝ^(d_f).

Core Concepts and Mathematical Foundations
Standard self-attention computes a weighted sum of value vectors for each query vector, where weights are determined by the similarity between the query and key vectors. Formally, for a single attention head, the output O is given by:
O = Attention(Q, K, V) = softmax(QK^T)V

The core computational bottleneck in standard self-attention arises from the explicit computation and storage of the attention score matrix QK^T, which is of size n × n. This leads to a quadratic dependency on the sequence length n for both time and memory complexity.

Sparse attention variants address this by restricting the set of (query, key) pairs for which attention scores are computed. Instead of computing all n² pairwise interactions, only a subset of interactions, defined by a sparsity pattern, are considered. This is typically achieved by applying a mask M to the attention scores, effectively setting scores for disallowed connections to negative infinity before the softmax operation. The geometric interpretation is that the attention graph, which is fully connected in standard self-attention, becomes sparse.

Linear attention variants aim to avoid the explicit n × n attention matrix altogether. They achieve this by re-arranging the attention computation using a property of matrix multiplication and a suitable kernel function. Specifically, if the softmax function can be approximated or replaced by a positive kernel function φ such that exp(q_i^T k_j) ≈ φ(q_i)^T φ(k_j), then the attention mechanism can be rewritten to change the order of operations from (QK^T)V to Q(K^T V) or similar, thereby avoiding the n × n intermediate matrix. The probabilistic interpretation of standard attention, where softmax(QK^T) represents a probability distribution over keys for each query, is altered in linear attention, often requiring explicit normalization.

Dimensional reasoning for standard attention:
*   Q ∈ ℝ^(n × d_k), K ∈ ℝ^(n × d_k)
*   QK^T ∈ ℝ^(n × d_k) ℝ^(d_k × n) = ℝ^(n × n)
*   softmax(QK^T) ∈ ℝ^(n × n) (element-wise softmax over rows)
*   softmax(QK^T)V ∈ ℝ^(n × n) ℝ^(n × d_v) = ℝ^(n × d_v)
The output O has the same shape as V, which is consistent with each token producing an output vector of dimension d_v.

Mechanism and Formal Derivation

Step 1: Standard Self-Attention Review
Given query matrix Q ∈ ℝ^(n × d_k), key matrix K ∈ ℝ^(n × d_k), and value matrix V ∈ ℝ^(n × d_v), the standard self-attention mechanism computes the output O ∈ ℝ^(n × d_v) as follows:
1.  Compute attention scores: A = QK^T.
    *   A_ij = q_i^T k_j, where q_i is the i-th row of Q and k_j is the j-th row of K.
    *   Shape: A ∈ ℝ^(n × n).
2.  Apply softmax function row-wise to obtain attention probabilities: P = softmax(A).
    *   P_ij = exp(A_ij) / Σ_l exp(A_il).
    *   Shape: P ∈ ℝ^(n × n).
3.  Compute the weighted sum of values: O = PV.
    *   O_i = Σ_j P_ij v_j, where O_i is the i-th row of O and v_j is the j-th row of V.
    *   Shape: O ∈ ℝ^(n × d_v).

Step 2: Sparse Attention - General Principle
Sparse attention modifies Step 1 by introducing an attention mask M ∈ ℝ^(n × n) before the softmax operation. The mask M has elements m_ij, where m_ij = 0 if the connection between query i and key j is allowed, and m_ij = -∞ (or a very large negative number) if it is disallowed.
1.  Compute masked attention scores: A' = QK^T + M.
    *   A'_ij = q_i^T k_j + m_ij.
    *   Shape: A' ∈ ℝ^(n × n).
2.  Apply softmax function row-wise: P = softmax(A').
    *   If m_ij = -∞, then exp(A'_ij) approaches 0, effectively setting P_ij to 0.
    *   Shape: P ∈ ℝ^(n × n).
3.  Compute the weighted sum of values: O = PV.
    *   Shape: O ∈ ℝ^(n × d_v).
The key is that for many sparse patterns, the number of non-zero elements in each row of P (and thus non-negative elements in M) is significantly less than n.

Step 3: Sparse Attention - Example: Local Attention
Local attention restricts each query token to attend only to key tokens within a fixed-size window around its position. Let w be the window size.
1.  Define the mask M:
    *   m_ij = 0 if |i - j| ≤ w.
    *   m_ij = -∞ otherwise.
2.  Compute masked attention scores: A' = QK^T + M.
    *   For each query q_i, only q_i^T k_j where j is in the range [max(0, i-w), min(n-1, i+w)] are considered.
    *   The number of active connections for each query is at most 2w+1.
3.  Apply softmax row-wise to A' to get P.
4.  Compute O = PV.
The computation of QK^T can still be O(n^2 d_k) if done naively. However, by only computing the relevant entries of QK^T, the complexity can be reduced. For each query q_i, we compute (2w+1) dot products.
*   The number of active (i,j) pairs is approximately n * (2w+1).
*   Each dot product q_i^T k_j takes O(d_k) time.
*   Total time for scores: O(n * w * d_k).
*   Softmax and weighted sum: O(n * w * d_v).
*   Total complexity: O(n * w * d_k + n * w * d_v).

Step 4: Linear Attention - General Principle
Linear attention aims to reformulate Attention(Q, K, V) = softmax(QK^T)V to avoid the n × n matrix multiplication. The core idea is to approximate or replace the softmax(QK^T) term with a product of feature maps applied to Q and K, allowing for a re-ordering of matrix multiplications.
Let P_ij = exp(q_i^T k_j) / Σ_l exp(q_i^T k_l).
The output for the i-th token is O_i = Σ_j P_ij v_j = Σ_j (exp(q_i^T k_j) / Σ_l exp(q_i^T k_l)) v_j.
This can be written as O_i = (Σ_j exp(q_i^T k_j) v_j) / (Σ_l exp(q_i^T k_l)).
The challenge is the term exp(q_i^T k_j). If we can find a positive kernel function φ: ℝ^(d_k) → ℝ^(d_f) such that exp(q_i^T k_j) ≈ φ(q_i)^T φ(k_j), then we can substitute this approximation.
Let φ(Q) be the matrix where each row is φ(q_i), so φ(Q) ∈ ℝ^(n × d_f). Similarly, φ(K) ∈ ℝ^(n × d_f).
Then, exp(q_i^T k_j) ≈ φ(q_i)^T φ(k_j) is the (i,j)-th element of φ(Q)φ(K)^T.

Step 5: Linear Attention - Formal Derivation (Kernel Approximation)
Using the kernel approximation from Step 4, we substitute φ(q_i)^T φ(k_j) for exp(q_i^T k_j):
1.  Approximate the numerator: Σ_j exp(q_i^T k_j) v_j ≈ Σ_j (φ(q_i)^T φ(k_j)) v_j.
    *   This can be rewritten as φ(q_i)^T (Σ_j φ(k_j) v_j).
    *   Let K' = φ(K) ∈ ℝ^(n × d_f) and V' = V ∈ ℝ^(n × d_v).
    *   The term Σ_j φ(k_j) v_j is the j-th row of K' multiplied by the j-th row of V'. This is not a standard matrix product.
    *   Instead, consider the matrix product (φ(K)^T V) ∈ ℝ^(d_f × n) ℝ^(n × d_v) = ℝ^(d_f × d_v).
    *   Then, φ(q_i)^T (φ(K)^T V) is a row vector of dimension d_v. This represents the i-th row of φ(Q)(φ(K)^T V).
    *   Shape: φ(Q) ∈ ℝ^(n × d_f), (φ(K)^T V) ∈ ℝ^(d_f × d_v). Product is ℝ^(n × d_v).

Step 6: Linear Attention - Normalization
The denominator Σ_l exp(q_i^T k_l) also needs to be approximated using the kernel function:
1.  Approximate the denominator: D_i = Σ_l exp(q_i^T k_l) ≈ Σ_l (φ(q_i)^T φ(k_l)).
    *   This can be rewritten as φ(q_i)^T (Σ_l φ(k_l)).
    *   Let S_K = Σ_l φ(k_l) ∈ ℝ^(d_f).
    *   Then D_i ≈ φ(q_i)^T S_K. This is a scalar.
2.  Combine numerator and denominator to get the final output O_i for the i-th token:
    *   O_i = (φ(q_i)^T (Σ_j φ(k_j) v_j)) / (φ(q_i)^T (Σ_l φ(k_l))).
    *   In matrix form, let S_KV = φ(K)^T V ∈ ℝ^(d_f × d_v) and S_K = φ(K)^T 1_n ∈ ℝ^(d_f), where 1_n is a column vector of ones of length n.
    *   The numerator is φ(Q) S_KV.
    *   The denominator is a diagonal matrix D where D_ii = φ(q_i)^T S_K.
    *   The final output O = D^(-1) (φ(Q) S_KV).
    *   Shape: D^(-1) ∈ ℝ^(n × n), φ(Q) ∈ ℝ^(n × d_f), S_KV ∈ ℝ^(d_f × d_v). Product is ℝ^(n × d_v).

Computational and Complexity Analysis

Standard Self-Attention:
*   Time Complexity:
    *   QK^T: O(n * d_k * n) = O(n^2 d_k).
    *   softmax: O(n^2).
    *   PV: O(n * n * d_v) = O(n^2 d_v).
    *   Total: O(n^2 d_k + n^2 d_v).
*   Memory Complexity:
    *   QK^T (attention scores): O(n^2).
    *   P (attention probabilities): O(n^2).
    *   Total: O(n^2).
*   Effect of scaling: Quadratic in n, linear in d_k, d_v. Dominant factor is n.

Sparse Attention (e.g., Local Attention with window size w):
*   Time Complexity:
    *   Computing relevant QK^T entries: O(n * w * d_k).
    *   softmax over w elements per row: O(n * w).
    *   PV (weighted sum over w elements per row): O(n * w * d_v).
    *   Total: O(n * w * d_k + n * w * d_v).
*   Memory Complexity:
    *   Storing relevant QK^T entries: O(n * w).
    *   Storing relevant P entries: O(n * w).
    *   Total: O(n * w).
*   Effect of scaling: Linear in n, linear in w, linear in d_k, d_v. If w is constant, it's O(n).

Linear Attention (using kernel function φ mapping to d_f dimensions):
*   Time Complexity:
    *   Compute φ(Q) ∈ ℝ^(n × d_f): O(n * d_k * d_f) (if φ is a linear projection followed by non-linearity, or similar).
    *   Compute φ(K) ∈ ℝ^(n × d_f): O(n * d_k * d_f).
    *   Compute S_KV = φ(K)^T V ∈ ℝ^(d_f × d_v): O(n * d_f * d_v).
    *   Compute S_K = φ(K)^T 1_n ∈ ℝ^(d_f): O(n * d_f).
    *   Compute numerator φ(Q) S_KV ∈ ℝ^(n × d_v): O(n * d_f * d_v).
    *   Compute denominator D_ii = φ(q_i)^T S_K: O(n * d_f).
    *   Inverse D^(-1) and final multiplication: O(n * d_v).
    *   Total: O(n * d_k * d_f + n * d_f * d_v).
*   Memory Complexity:
    *   φ(Q): O(n * d_f).
    *   φ(K): O(n * d_f).
    *   S_KV: O(d_f * d_v).
    *   S_K: O(d_f).
    *   Total: O(n * d_f + d_f * d_v).
*   Effect of scaling: Linear in n, linear in d_f, linear in d_k, d_v. If d_f is constant or grows sub-linearly with n, it's O(n).

Trade-offs:
*   Standard Attention: High expressivity, exact attention scores, but O(n^2) complexity limits n.
*   Sparse Attention: Reduces complexity to O(n * w) or O(n * k) (where k is average sparsity), but restricts information flow to predefined patterns. The choice of pattern (e.g., local, strided, global tokens) is critical and often heuristic.
*   Linear Attention: Achieves O(n) complexity by avoiding the n × n matrix. However, it relies on a kernel approximation of softmax, which can reduce expressivity and accuracy. The choice of kernel function