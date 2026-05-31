FlashAttention and Memory-Efficient Attention

Assumptions and Notation
This document assumes familiarity with the fundamental concepts of the Transformer architecture, specifically the scaled dot-product attention mechanism. The primary goal is to detail methods for optimizing its memory and computational efficiency.

Key variables and dimensions:
-   Q ∈ ℝ^(n × d_k): Query matrix, where n is the sequence length and d_k is the dimension of query/key vectors.
-   K ∈ ℝ^(n × d_k): Key matrix, with n sequence length and d_k dimension.
-   V ∈ ℝ^(n × d_v): Value matrix, with n sequence length and d_v dimension.
-   S ∈ ℝ^(n × n): Logit matrix, representing the unnormalized attention scores. S = QKᵀ / √d_k.
-   P ∈ ℝ^(n × n): Attention probability matrix, where P = softmax(S).
-   O ∈ ℝ^(n × d_v): Output matrix of the attention mechanism. O = PV.
-   L ∈ ℝ^n: Row-wise normalization factor vector for softmax. L_i = Σ_j exp(S_ij - m_i).
-   m ∈ ℝ^n: Row-wise maximum value vector for numerical stability in softmax. m_i = max_j(S_ij).
-   B_r: Block size for rows of Q (query block size).
-   B_c: Block size for columns of K and V (key/value block size).
-   HBM: High-Bandwidth Memory (e.g., GPU global memory).
-   SRAM: On-chip Static Random-Access Memory (e.g., GPU shared memory or L1 cache).

Core Concepts and Mathematical Foundations
The standard scaled dot-product attention mechanism is defined as:
Attention(Q, K, V) = softmax(QKᵀ / √d_k)V

1.  **Scaled Dot-Product (QKᵀ / √d_k):**
    -   The product QKᵀ computes the dot product similarity between each query vector q_i (row i of Q) and each key vector k_j (row j of K).
    -   Q ∈ ℝ^(n × d_k), K ∈ ℝ^(n × d_k).
    -   QKᵀ ∈ ℝ^(n × n).
    -   The scaling factor 1/√d_k is applied to prevent the dot products from growing too large in magnitude, which can push the softmax function into regions with very small gradients, hindering learning.
    -   Let S_ij = (q_i ⋅ k_j) / √d_k. The matrix S ∈ ℝ^(n × n) contains these unnormalized attention scores (logits).

2.  **Softmax Normalization:**
    -   The softmax function is applied row-wise to S to obtain the attention probability matrix P.
    -   P_ij = exp(S_ij) / Σ_k exp(S_ik).
    -   Each row of P sums to 1, representing a probability distribution over the input sequence for each query.
    -   P ∈ ℝ^(n × n).

3.  **Weighted Sum of Values (PV):**
    -   The attention probabilities P are then used to compute a weighted sum of the value vectors.
    -   O_i = Σ_j P_ij V_j, where V_j is row j of V.
    -   O ∈ ℝ^(n × d_v).

**Problem Statement: Memory Bottleneck**
The primary memory bottleneck in standard attention arises from the materialization of the intermediate matrices S and P, both of which are of size n × n. For a sequence length n, storing these matrices requires O(n²) memory. For large n (e.g., n = 65536), this can exceed the capacity of high-bandwidth memory (HBM) on accelerators, or at least significantly limit batch sizes.

**Key Idea of Memory-Efficient Attention**
Memory-efficient attention techniques, particularly FlashAttention, aim to circumvent this O(n²) memory bottleneck by:
1.  **Avoiding Materialization:** Not storing the full S and P matrices in HBM.
2.  **Tiling/Blocking:** Processing the input Q, K, V matrices in smaller blocks.
3.  **On-chip Memory (SRAM) Utilization:** Performing the computationally intensive parts (QKᵀ, softmax, PV) entirely within fast, but small, on-chip memory (SRAM) for each block, minimizing data movement to/from slower HBM.
4.  **Online Softmax:** Developing a mathematically equivalent way to compute the softmax and its subsequent matrix multiplication without needing the full P matrix at any single point.

Mechanism and Formal Derivation
FlashAttention reorders the operations of standard attention to reduce HBM accesses. It leverages tiling and an "online" softmax computation.

**Step 1: Standard Attention Formulation (Baseline)**
The output O is given by:
O = softmax(QKᵀ / √d_k)V

Let S = QKᵀ / √d_k. Then O = softmax(S)V.
For each row i of Q, the output vector o_i is:
o_i = Σ_j (exp(S_ij) / Σ_k exp(S_ik)) v_j

**Step 2: Numerical Stability and Online Softmax Transformation**
To prevent numerical overflow/underflow when computing exp(S_ij), especially for large S_ij, a common trick is to subtract the row-wise maximum from S before exponentiation.
Let m_i = max_j(S_ij).
Then, exp(S_ij) / Σ_k exp(S_ik) = exp(S_ij - m_i) / Σ_k exp(S_ik - m_i).
This transformation is numerically stable and mathematically equivalent.
Let L_i = Σ_k exp(S_ik - m_i).
Then, o_i = Σ_j (exp(S_ij - m_i) / L_i) v_j.

The core challenge for memory efficiency is that L_i (and m_i) depend on *all* S_ik for a given row i. This means a full pass over K is seemingly required to compute L_i before any v_j can be weighted. FlashAttention addresses this by updating m_i and L_i incrementally.

**Step 3: Tiling Q, K, V for On-Chip Computation**
The input matrices Q, K, V are divided into blocks.
-   Q is divided into B_r × d_k blocks: Q_1, Q_2, ..., Q_{N_r}, where N_r = n / B_r.
-   K and V are divided into B_c × d_k and B_c × d_v blocks respectively: K_1, K_2, ..., K_{N_c} and V_1, V_2, ..., V_{N_c}, where N_c = n / B_c.

The computation proceeds by iterating over blocks of Q (Q_i) and, for each Q_i, iterating over blocks of K and V (K_j, V_j).
For a given Q_i ∈ ℝ^(B_r × d_k), we want to compute its corresponding output block O_i ∈ ℝ^(B_r × d_v).
O_i = softmax(Q_i Kᵀ / √d_k)V.
This still requires the full K and V. The online softmax is key.

**Step 4: Incremental Softmax Normalization (Online Softmax)**
Consider computing the output for a single block Q_i. We iterate through the K_j and V_j blocks.
Let O_i_acc be the accumulated output for Q_i, and m_i_acc and L_i_acc be the accumulated row-wise maximums and sums for the rows in Q_i. These are initialized to -∞ and 0 respectively.

For each block K_j ∈ ℝ^(B_c × d_k) and V_j ∈ ℝ^(B_c × d_v):
1.  **Compute partial logits:** S_ij = Q_i K_jᵀ / √d_k. This matrix S_ij ∈ ℝ^(B_r × B_c) is computed entirely in SRAM.
2.  **Update row-wise maximums and sums:** For each row r in Q_i (i.e., for each row of S_ij):
    -   Let m_r_old = m_i_acc[r] and L_r_old = L_i_acc[r].
    -   Let m_r_new_block = max_c(S_ij[r, c]).
    -   Let m_r_new = max(m_r_old, m_r_new_block).
    -   Update L_r_acc: L_r_acc[r] = L_r_old * exp(m_r_old - m_r_new) + Σ_c exp(S_ij[r, c] - m_r_new).
    -   Update m_i_acc[r] = m_r_new.
    -   This step is crucial: it allows updating the normalization factors incrementally without storing the full S matrix.

3.  **Update partial output:** For each row r in Q_i:
    -   O_i_acc[r] = O_i_acc[r] * exp(m_r_old - m_r_new) * (L_r_old / L_r_acc[r]) + Σ_c (exp(S_ij[r, c] - m_r_new) / L_r_acc[r]) * V_j[c, :].
    -   This update rule ensures that the accumulated output O_i_acc is correctly scaled with the *current* normalization factors. The terms exp(m_r_old - m_r_new) and (L_r_old / L_r_acc[r]) re-scale the previously accumulated output to match the new maximum and sum.

This incremental update is performed for all K_j, V_j blocks. After iterating through all j, O_i_acc will contain the final output block O_i.

**Step 5: Two-Pass Algorithm (FlashAttention Specific Optimization)**
While the online softmax in Step 4 is mathematically sound, its implementation can be complex due to the need to re-scale the accumulated output at each step. FlashAttention simplifies this by using a two-pass approach, which is particularly efficient on GPUs.

**Pass 1 (Forward Pass):**
For each Q_i block:
1.  Initialize O_i_scaled ∈ ℝ^(B_r × d_v) to zeros, m_i ∈ ℝ^(B_r) to -∞, and L_i ∈ ℝ^(B_r) to zeros.
2.  Iterate through K_j and V_j blocks (j = 1 to N_c):
    a.  Load Q_i, K_j, V_j into SRAM.
    b.  Compute S_ij = Q_i K_jᵀ / √d_k in SRAM.
    c.  For each row r in Q_i:
        i.   Let m_r_old = m_i[r] and L_r_old = L_i[r].
        ii.  Let m_r_new = max(m_r_old, max_c(S_ij[r, c])).
        iii. Update O_i_scaled[r] = O_i_scaled[r] * exp(m_r_old - m_r_new) + Σ_c exp(S_ij[r, c] - m_r_new) * V_j[c, :].
        iv.  Update L_i[r] = L_r_old * exp(m_r_old - m_r_new) + Σ_c exp(S_ij[r, c] - m_r_new).
        v.   Update m_i[r] = m_r_new.
    d.  Store m_i and L_i (for each row of Q_i) to HBM. These are needed for the backward pass.
3.  After iterating through all K_j, V_j blocks, O_i_scaled contains the unnormalized, scaled output. Store O_i_scaled to HBM.

At the end of Pass 1, we have O_scaled ∈ ℝ^(n × d_v), m ∈ ℝ^n, and L ∈ ℝ^n stored in HBM. The final output O is O_scaled normalized by L: O_ij = O_scaled_ij / L_i. This final division is performed as a separate element-wise operation.

**Step 6: Backward Pass (Gradient Computation)**
The backward pass for attention also involves an O(n²) memory bottleneck if the full attention matrix P is stored. FlashAttention recomputes the necessary parts of the attention matrix on-the-fly during the backward pass, using the stored m and L from the forward pass. This avoids storing P.

For the backward pass, given dO/dO_scaled and dO/dL, we need to compute dL/dS and dO_scaled/dS. This requires re-evaluating S_ij and P_ij blocks.
1.  Load Q_i, K_j, V_j, m_i, L_i, dO_i/dO_scaled_i, dO_i/dL_i into SRAM.
2.  Recompute S_ij = Q_i K_jᵀ / √d_k in SRAM.
3.  Recompute P_ij = softmax(S_ij) using the stored m_i and L_i.
4.  Compute gradients dS_ij, dQ_i, dK_j, dV_j using P_ij and the incoming gradients.
This recomputation strategy is memory-efficient because only small blocks are