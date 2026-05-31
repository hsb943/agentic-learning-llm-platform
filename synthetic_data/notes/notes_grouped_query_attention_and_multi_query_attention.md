Title
Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

Assumptions and Notation
This document assumes familiarity with the fundamental concepts of the Transformer architecture, specifically the self-attention mechanism. The following notation is used consistently:

-   b: Batch size.
-   n: Sequence length (number of tokens in a sequence).
-   d_model: Dimensionality of the input and output representations for each token.
-   h: Total number of attention heads.
-   g: Number of key/value groups. For Multi-Query Attention (MQA), g=1. For Multi-Head Attention (MHA), g=h. For Grouped Query Attention (GQA), 1 < g < h.
-   d_k: Dimensionality of the query and key vectors for a single attention head. Typically, d_k = d_model / h.
-   d_v: Dimensionality of the value vectors for a single attention head. Typically, d_v = d_model / h.
-   X: Input tensor to the attention block, X ∈ ℝ^(b × n × d_model).
-   W_Q: Query projection matrix.
-   W_K: Key projection matrix.
-   W_V: Value projection matrix.
-   W_O: Output projection matrix.
-   Q: Query tensor.
-   K: Key tensor.
-   V: Value tensor.
-   S: Attention score tensor (logits before softmax).
-   A: Attention weight tensor (after softmax).
-   O: Output tensor from an attention head.
-   O_concat: Concatenated output tensor from all attention heads.
-   Y: Final output tensor from the attention block.

Core Concepts and Mathematical Foundations

Multi-Head Attention (MHA) serves as the baseline for understanding MQA and GQA. In MHA, each of 'h' attention heads independently projects the input X into its own Query (Q_i), Key (K_i), and Value (V_i) spaces.

Formal Definitions:
1.  **Multi-Head Attention (MHA):**
    -   Each of the 'h' attention heads has its own distinct projection matrices for queries, keys, and values.
    -   For head i ∈ {1, ..., h}:
        -   Q_i = X W_Q_i, where W_Q_i ∈ ℝ^(d_model × d_k).
        -   K_i = X W_K_i, where W_K_i ∈ ℝ^(d_model × d_k).
        -   V_i = X W_V_i, where W_V_i ∈ ℝ^(d_model × d_v).
    -   The attention mechanism for head i is: Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / sqrt(d_k)) V_i.
    -   The outputs of all heads are concatenated and linearly projected: Y = Concat(Attention_1, ..., Attention_h) W_O.

2.  **Multi-Query Attention (MQA):**
    -   MQA is a specialized form of MHA where all 'h' query heads share the *same* Key (K) and Value (V) projection matrices and thus the same projected K and V tensors.
    -   There is only one set of K and V projection matrices, effectively g=1.
    -   For head i ∈ {1, ..., h}:
        -   Q_i = X W_Q_i, where W_Q_i ∈ ℝ^(d_model × d_k).
        -   K = X W_K, where W_K ∈ ℝ^(d_model × d_k).
        -   V = X W_V, where W_V ∈ ℝ^(d_model × d_v).
    -   The attention mechanism for head i is: Attention(Q_i, K, V) = softmax(Q_i K^T / sqrt(d_k)) V.
    -   The outputs are concatenated and projected as in MHA.

3.  **Grouped Query Attention (GQA):**
    -   GQA generalizes MQA by allowing 'g' distinct groups of Key and Value projection matrices, where 1 < g < h. Each group of K and V is shared among a subset of h/g query heads.
    -   Let h_g = h/g be the number of query heads per group.
    -   For each group j ∈ {1, ..., g}:
        -   K_j = X W_K_j, where W_K_j ∈ ℝ^(d_model × d_k).
        -   V_j = X W_V_j, where W_V_j ∈ ℝ^(d_model × d_v).
    -   For each query head i ∈ {1, ..., h}:
        -   Q_i = X W_Q_i, where W_Q_i ∈ ℝ^(d_model × d_k).
        -   The query head i is assigned to group j = ceil(i / h_g). It uses K_j and V_j.
    -   The attention mechanism for head i is: Attention(Q_i, K_j, V_j) = softmax(Q_i K_j^T / sqrt(d_k)) V_j.
    -   The outputs are concatenated and projected as in MHA.

Geometric or Probabilistic Interpretation:
-   **MHA:** Each head learns a distinct "perspective" or "filter" on the input sequence, allowing it to identify different types of relationships (e.g., syntactic dependencies, semantic similarities). The K and V projections define the space of features that each head can attend to and the values it can extract.
-   **MQA:** All query heads operate on a single, shared "contextual representation" defined by the common K and V projections. While each query head still has its unique W_Q_i to define what it *looks for*, it must find those patterns within the *same* underlying key-value space. This can be interpreted as forcing all heads to agree on a fundamental set of features or a single "world view" of the input, potentially reducing the model's capacity to capture diverse relationships but improving efficiency.
-   **GQA:** Provides a middle ground. Instead of a single shared context (MQA) or entirely independent contexts (MHA), GQA allows for 'g' distinct contextual representations. Query heads within the same group share a K/V context, while heads in different groups operate on different K/V contexts. This allows for some diversity in feature extraction while still benefiting from K/V sharing.

Dimensional Reasoning:
-   **Input:** X ∈ ℝ^(b × n × d_model).
-   **MHA Projections:**
    -   Total Query parameters: h * d_model * d_k.
    -   Total Key parameters: h * d_model * d_k.
    -   Total Value parameters: h * d_model * d_v.
    -   Projected Q_i, K_i ∈ ℝ^(b × n × d_k).
    -   Projected V_i ∈ ℝ^(b × n × d_v).
-   **MQA Projections:**
    -   Total Query parameters: h * d_model * d_k.
    -   Total Key parameters: 1 * d_model * d_k.
    -   Total Value parameters: 1 * d_model * d_v.
    -   Projected Q_i ∈ ℝ^(b × n × d_k).
    -   Projected K, V ∈ ℝ^(b × n × d_k) and ℝ^(b × n × d_v) respectively.
-   **GQA Projections:**
    -   Total Query parameters: h * d_model * d_k.
    -   Total Key parameters: g * d_model * d_k.
    -   Total Value parameters: g * d_model * d_v.
    -   Projected Q_i ∈ ℝ^(b × n × d_k).
    -   Projected K_j, V_j ∈ ℝ^(b × n × d_k) and ℝ^(b × n × d_v) respectively.

The key difference in dimensionality lies in the number of distinct K and V projection matrices, which directly impacts the number of parameters and the size of the Key-Value (KV) cache during inference.

Mechanism and Formal Derivation

The attention mechanism for GQA (which encompasses MQA as a special case where g=1 and MHA where g=h) can be formally derived through the following steps:

Step 1: Input Projections for Queries, Keys, and Values
The input sequence X ∈ ℝ^(b × n × d_model) is linearly projected to obtain query, key, and value representations.
-   **Query Projections:** For each of the 'h' attention heads, a distinct query projection matrix W_Q_i ∈ ℝ^(d_model × d_k) is used.
    -   Q_i = X W_Q_i
    -   Resulting Q_i ∈ ℝ^(b × n × d_k) for each head i ∈ {1, ..., h}.
    -   These can be stacked for efficient computation: Q_all = [Q_1; ...; Q_h] (conceptually, often implemented as a single large matrix multiplication X W_Q_all where W_Q_all ∈ ℝ^(d_model × (h * d_k))).
    -   Reshaped Q_all ∈ ℝ^(b × h × n × d_k).
-   **Key Projections:** For each of the 'g' key groups, a distinct key projection matrix W_K_j ∈ ℝ^(d_model × d_k) is used.
    -   K_j = X W_K_j
    -   Resulting K_j ∈ ℝ^(b × n × d_k) for each group j ∈ {1, ..., g}.
    -   These can be stacked: K_all = [K_1; ...; K_g] (conceptually, often implemented as X W_K_all where W_K_all ∈ ℝ^(d_model × (g * d_k))).
    -   Reshaped K_all ∈ ℝ^(b × g × n × d_k).
-   **Value Projections:** Similarly, for each of the 'g' value groups, a distinct value projection matrix W_V_j ∈ ℝ^(d_model × d_v) is used.
    -   V_j = X W_V_j
    -   Resulting V_j ∈ ℝ^(b × n × d_v) for each group j ∈ {1, ..., g}.
    -   These can be stacked: V_all = [V_1; ...; V_g] (conceptually, often implemented as X W_V_all where W_V_all ∈ ℝ^(d_model × (g * d_v))).
    -   Reshaped V_all ∈ ℝ^(b × g × n × d_v).

Step 2: Query-Key Dot Product for Attention Scores
For each query head i, it computes attention scores against the keys from its assigned group. Let h_g = h/g be the number of query heads per group. The group index for head i is j = ceil(i / h_g).
-   S_i = Q_i K_j^T / sqrt(d_k)
-   Q_i ∈ ℝ^(b × n × d_k)
-   K_j^T ∈ ℝ^(b × d_k × n)
-   Resulting S_i ∈ ℝ^(b × n × n). This tensor represents the raw attention scores (logits) for each query position attending to each key position within the sequence. The division by sqrt(d_k) is a scaling factor to prevent the dot products from becoming too large, which can push the softmax function into regions with very small gradients.

Step 3: Softmax Normalization
The attention scores S_i are normalized using the softmax function along the last dimension (the key dimension) to obtain attention weights.
-   A_i = softmax(S_i)
-   Resulting A_i ∈ ℝ^(b × n × n). Each row of A_i sums to 1, representing a probability distribution over the key positions for each query position.

Step 4: Weighted Sum of Values
The attention weights A_i are used to compute a weighted sum of the value vectors from the assigned group j.
-   O_i = A_i V_j
-   A_i ∈ ℝ^(b × n × n)
-   V_j ∈ ℝ^(b × n × d_v)
-   Resulting O_i ∈ ℝ^(b × n × d_v). This is the output of a single attention head, representing a context-aware representation for each token in the sequence, specific to that head's query perspective and its assigned key-value group.

Step 5: Concatenation of Head Outputs
The outputs from all 'h' attention heads are concatenated along the feature dimension.
-   O_concat = Concat(O_1, O_2, ..., O_h)
-   O_concat ∈ ℝ^(b × n × (h * d_v)).

Step 6: Final Linear Projection
The concatenated output is linearly projected back to the model's dimensionality d_model using a final output projection matrix W_O ∈ ℝ^((h * d_v) × d_model).
-   Y = O_concat W_O
-   Resulting Y ∈ ℝ^(b × n × d_model). This is the final output of the GQA block, ready to be passed to subsequent layers (e.g., feed-forward network, residual connection).

Computational and Complexity Analysis

The primary motivation for MQA and GQA is to reduce computational and memory costs, particularly during inference with long sequences.

Time Complexity:
The dominant operations in an attention block are matrix multiplications.
-   **Projection Operations (Q, K, V):**
    -   Queries: X W_Q_all, where W_Q_all ∈ ℝ^(d_model × (h * d_k)). Cost: O(b * n * d_model * h * d_k).
    -   Keys: X W_K_all,