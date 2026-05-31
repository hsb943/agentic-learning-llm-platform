KV Cache for Autoregressive Inference

Assumptions and Notation
This document assumes familiarity with the fundamental architecture of the Transformer model, specifically the multi-head self-attention mechanism. Autoregressive inference implies that tokens are generated sequentially, with each new token conditioned on all previously generated tokens.

The following variables and their dimensions are used throughout this document:

B: Batch size. Number of independent sequences processed concurrently.
L: Current sequence length. The number of tokens generated so far, including the current token.
L_max: Maximum possible sequence length. The maximum number of tokens that can be generated.
d_model: Model dimension. The dimensionality of the token embeddings and the output of the attention block.
H: Number of attention heads.
d_k: Key and Query dimension per head. Typically d_k = d_model / H.
d_v: Value dimension per head. Typically d_v = d_model / H.

x_t ∈ ℝ^(B × 1 × d_model): Input embedding for the current token at time step t.
W_Q ∈ ℝ^(d_model × H × d_k): Query projection weight matrix.
W_K ∈ ℝ^(d_model × H × d_k): Key projection weight matrix.
W_V ∈ ℝ^(d_model × H × d_v): Value projection weight matrix.
W_O ∈ ℝ^(H × d_v × d_model): Output projection weight matrix.

Q_t ∈ ℝ^(B × H × 1 × d_k): Query vector for the current token at time step t, across all heads.
K_t ∈ ℝ^(B × H × 1 × d_k): Key vector for the current token at time step t, across all heads.
V_t ∈ ℝ^(B × H × 1 × d_v): Value vector for the current token at time step t, across all heads.

K_cache_t ∈ ℝ^(B × H × t × d_k): Accumulated Key cache up to time step t.
V_cache_t ∈ ℝ^(B × H × t × d_v): Accumulated Value cache up to time step t.

S_t ∈ ℝ^(B × H × 1 × t): Unnormalized attention scores for the current token at time step t.
A_t ∈ ℝ^(B × H × 1 × t): Attention weights (softmax output) for the current token at time step t.
O_t ∈ ℝ^(B × H × 1 × d_v): Output of the multi-head attention for the current token at time step t, before concatenation.
Y_t ∈ ℝ^(B × 1 × d_model): Final output of the attention block for the current token at time step t.

Core Concepts and Mathematical Foundations
The core of the Transformer architecture is the self-attention mechanism, which allows a model to weigh the importance of different parts of an input sequence when processing each token. For a given query Q, keys K, and values V, the attention function is defined as:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

In autoregressive inference, a model generates tokens one by one. To predict the next token, the model must attend to all previously generated tokens. Without optimization, this would involve recomputing the Key (K) and Value (V) projections for all preceding tokens at each time step. The KV Cache is a mechanism designed to store these K and V projections, thereby eliminating redundant computation.

Consider the self-attention computation for a single head at time step t. The query Q_t is derived from the current token x_t. The keys K and values V, however, must correspond to the entire sequence generated so far, i.e., {x_1, x_2, ..., x_t}.

Formally, for a single attention head and a single token x_t:
Q_t = x_t W_Q_h
K_t = x_t W_K_h
V_t = x_t W_V_h

Where W_Q_h, W_K_h, W_V_h are the weight matrices for a specific head h.
In the multi-head attention mechanism, these are computed for all heads simultaneously and then reshaped.

The attention scores for the current token Q_t are computed against all keys K_1, K_2, ..., K_t.
S_t = Q_t [K_1^T, K_2^T, ..., K_t^T] / sqrt(d_k)

The attention weights A_t are then derived by applying softmax to S_t.
A_t = softmax(S_t)

Finally, the output O_t is a weighted sum of the value vectors V_1, V_2, ..., V_t.
O_t = A_t [V_1, V_2, ..., V_t]^T

The KV Cache stores the concatenated sequence of K and V vectors, K_cache_t and V_cache_t, from previous time steps. At each new time step t, only the K_t and V_t for the current token x_t need to be computed. These new K_t and V_t are then appended to K_cache_{t-1} and V_cache_{t-1} to form K_cache_t and V_cache_t, respectively. The query Q_t then attends to these accumulated caches. This avoids the recomputation of K_i and V_i for i < t.

Dimensional reasoning:
The input token embedding x_t has shape B × 1 × d_model.
The projection matrices W_Q, W_K, W_V have shapes d_model × H × d_k, d_model × H × d_k, d_model × H × d_v respectively.
After projection and reshaping for multi-head attention, Q_t, K_t, V_t will have shapes B × H × 1 × d_k, B × H × 1 × d_k, B × H × 1 × d_v.
The accumulated K_cache_t and V_cache_t will have shapes B × H × t × d_k and B × H × t × d_v.
The attention score matrix S_t, resulting from Q_t K_cache_t^T, will have shape (B × H × 1 × d_k) × (B × H × d_k × t) -> B × H × 1 × t. This represents the scores of the current query against all t keys.
The attention weights A_t will retain the shape B × H × 1 × t.
The output O_t, resulting from A_t V_cache_t, will have shape (B × H × 1 × t) × (B × H × t × d_v) -> B × H × 1 × d_v. This is the weighted sum of value vectors for the current token.

Mechanism and Formal Derivation
The KV Cache mechanism optimizes the multi-head self-attention computation during autoregressive inference. The process can be broken down into the following steps:

1.  Initialization (Time Step t=0):
    Before generating the first token, the Key and Value caches are empty.
    K_cache_0 = [] (empty list or tensor)
    V_cache_0 = [] (empty list or tensor)

2.  First Token Processing (Time Step t=1):
    a.  Input: The initial input token embedding x_1 ∈ ℝ^(B × 1 × d_model) is provided.
    b.  Projection: For each attention head h ∈ {1, ..., H}, the query, key, and value vectors for the current token are computed:
        Q_1_h = x_1 W_Q_h ∈ ℝ^(B × 1 × d_k)
        K_1_h = x_1 W_K_h ∈ ℝ^(B × 1 × d_k)
        V_1_h = x_1 W_V_h ∈ ℝ^(B × 1 × d_v)
        These are then concatenated across heads to form Q_1 ∈ ℝ^(B × H × 1 × d_k), K_1 ∈ ℝ^(B × H × 1 × d_k), V_1 ∈ ℝ^(B × H × 1 × d_v).
    c.  Cache Update: The computed K_1 and V_1 are stored in the caches.
        K_cache_1 = K_1
        V_cache_1 = V_1
        Shapes: K_cache_1 ∈ ℝ^(B × H × 1 × d_k), V_cache_1 ∈ ℝ^(B × H × 1 × d_v).
    d.  Attention Calculation: The attention scores S_1 are computed by multiplying Q_1 with the transpose of K_cache_1, scaled by sqrt(d_k).
        S_1 = (Q_1 K_cache_1^T) / sqrt(d_k) ∈ ℝ^(B × H × 1 × 1)
        The attention weights A_1 are obtained by applying the softmax function to S_1.
        A_1 = softmax(S_1) ∈ ℝ^(B × H × 1 × 1)
        The output O_1 is the weighted sum of V_cache_1.
        O_1 = A_1 V_cache_1 ∈ ℝ^(B × H × 1 × d_v)
    e.  Output Projection: The outputs from all heads are concatenated and projected back to d_model.
        O_1_concat = concat(O_1_h for h in H) ∈ ℝ^(B × 1 × (H * d_v))
        Y_1 = O_1_concat W_O ∈ ℝ^(B × 1 × d_model)
        Y_1 is then passed to subsequent layers (e.g., feed-forward network) and ultimately used to predict the next token.

3.  Subsequent Token Processing (Time Step t > 1):
    a.  Input: The newly generated token's embedding x_t ∈ ℝ^(B × 1 × d_model) is provided.
    b.  Projection: Similar to step 2b, Q_t, K_t, V_t are computed for the current token x_t.
        Q_t ∈ ℝ^(B × H × 1 × d_k)
        K_t ∈ ℝ^(B × H × 1 × d_k)
        V_t ∈ ℝ^(B × H × 1 × d_v)

4.  Cache Update: The newly computed K_t and V_t are appended to the existing caches from the previous time step (t-1).
    K_cache_t = concat(K_cache_{t-1}, K_t, axis=2) ∈ ℝ^(B × H × t × d_k)
    V_cache_t = concat(V_cache_{t-1}, V_t, axis=2) ∈ ℝ^(B × H × t × d_v)
    The axis=2 indicates concatenation along the sequence length dimension.

5.  Attention Calculation: The query Q_t (for the current token only) attends to the *entire* accumulated K_cache_t and V_cache_t.
    a.  Attention Scores:
        S_t = (Q_t K_cache_t^T) / sqrt(d_k) ∈ ℝ^(B × H × 1 × t)
        Note that K_cache_t^T implies transposing the last two dimensions, resulting in shape ℝ^(B × H × d_k × t).
    b.  Attention Weights:
        A_t = softmax(S_t) ∈ ℝ^(B × H × 1 × t)
    c.  Weighted Sum of Values:
        O_t = A_t V_cache_t ∈ ℝ^(B × H × 1 × d_v)

6.  Output Projection: The outputs from all heads are concatenated and projected back to d_model.
    O_t_concat = concat(O_t_h for h in H) ∈ ℝ^(B × 1 × (H * d_v))
    Y_t = O_t_concat W_O ∈ ℝ^(B × 1 × d_model)
    Y_t is then used to predict the next token, and the process repeats from step 3 for t+1.

This derivation explicitly shows that at each step t, only Q_t, K_t, and V_t for the *current* token are computed from scratch. The K and V vectors for all previous tokens are retrieved from the cache, avoiding redundant computation.

Computational and Complexity Analysis

The KV Cache significantly alters the computational and memory complexity of autoregressive inference compared to recomputing attention from scratch at each step. We analyze the complexity for generating a sequence of length L.

Time Complexity:

Without KV Cache (Recomputation at each step):
For each token t from 1 to L:
  Input: x_t ∈ ℝ^(B × 1 × d_model).
  The attention mechanism needs to process the entire sequence {x_1, ..., x_t}.
  1.  Query, Key, Value Projections: For each token in the current sequence (length t), Q, K, V matrices are computed. This involves matrix multiplications of shape (B × t × d_model) with (d_model × d_k) for Q, K, and (d_model × d_v) for V.
      Complexity per token t: O(t * d_model * (d_k + d_v)).
      Total for sequence L: Sum_{t=1 to L} O(t * d_model * (d_k + d_v)) = O(L^2 * d_model * (d_k + d_v)).
  2.  Attention Scores (QK^T): Q ∈ ℝ^(B × H × t × d_k), K^T ∈ ℝ^(B × H × d_k × t).
      Complexity per token t: O(H * t * d_k * t) = O(H * t^2 * d_k).
      Total for sequence L: Sum_{t=1 to L} O(H * t^2 * d_k) = O(H * L^3 * d_k).
  3.  Weighted Sum of Values (AV): A ∈ ℝ^(B × H × t × t), V ∈ ℝ^(B × H × t × d_v).
      Complexity per token t: O(H * t * t * d_v) = O(H * t^2 * d_v).
      Total for sequence L: Sum_{t=1 to L} O(H * t^2 * d_v) = O(H * L^3 * d_v).
Overall Time Complexity (without KV Cache): O(H * L^3 * (d_k + d_v)) = O(L^3 * d_model).

With KV Cache:
For