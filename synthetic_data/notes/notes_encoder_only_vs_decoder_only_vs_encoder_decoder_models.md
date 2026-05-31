Encoder-Only vs Decoder-Only vs Encoder-Decoder Models: A Technical Reference

Assumptions and Notation

This document assumes familiarity with fundamental linear algebra operations, calculus, and basic concepts of neural networks. The core computational unit is the Transformer block, which relies on Multi-Head Self-Attention (MHA) and Feed-Forward Networks (FFN).

Key Variables and Dimensions:
- B: Batch size, representing the number of independent sequences processed in parallel.
- L: Sequence length, the number of tokens in a sequence.
- L_enc: Encoder input sequence length.
- L_dec: Decoder output sequence length.
- V: Vocabulary size, the total number of unique tokens.
- d_model: Dimensionality of the model's internal representations (embedding dimension).
- d_ff: Dimensionality of the inner layer of the Feed-Forward Network. Typically d_ff = 4 * d_model.
- h: Number of attention heads in Multi-Head Attention.
- d_k: Dimensionality of Query (Q) and Key (K) vectors for a single attention head. Typically d_k = d_model / h.
- d_v: Dimensionality of Value (V) vectors for a single attention head. Typically d_v = d_model / h.
- N_enc: Number of stacked encoder layers.
- N_dec: Number of stacked decoder layers.

Matrix Shape Notation:
- X ∈ ℝ^(B × L × d_model): Input tensor representing a batch of L sequences, each token embedded in d_model dimensions.
- E ∈ ℝ^(V × d_model): Token embedding matrix.
- PE ∈ ℝ^(L × d_model): Positional Encoding matrix.
- W_Q, W_K, W_V ∈ ℝ^(d_model × d_k): Weight matrices for projecting input to Query, Key, Value for a single attention head.
- W_O ∈ ℝ^(h * d_v × d_model): Output projection matrix for Multi-Head Attention.
- W_1_FFN ∈ ℝ^(d_model × d_ff), W_2_FFN ∈ ℝ^(d_ff × d_model): Weight matrices for the Feed-Forward Network.
- Q, K, V ∈ ℝ^(B × L × d_k) or ℝ^(B × L × d_v): Query, Key, Value tensors.
- S ∈ ℝ^(B × L × L): Attention score matrix.
- A ∈ ℝ^(B × L × L): Attention weight matrix (after softmax).
- Z ∈ ℝ^(B × L × d_v): Output of a single attention head.
- H_enc ∈ ℝ^(B × L_enc × d_model): Output of the encoder stack.
- H_dec ∈ ℝ^(B × L_dec × d_model): Output of the decoder stack.

Core Concepts and Mathematical Foundations

The Transformer architecture, the foundation for these models, relies on three primary mechanisms: self-attention, positional encoding, and feed-forward networks, integrated with residual connections and layer normalization.

1.  **Self-Attention Mechanism:** This mechanism allows a model to weigh the importance of different parts of an input sequence when processing each token.
    -   **Scaled Dot-Product Attention:** Given Query (Q), Key (K), and Value (V) matrices, the attention output is computed as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        Where:
        -   Q ∈ ℝ^(B × L_Q × d_k)
        -   K ∈ ℝ^(B × L_K × d_k)
        -   V ∈ ℝ^(B × L_K × d_v)
        -   QK^T ∈ ℝ^(B × L_Q × L_K) represents the raw attention scores.
        -   sqrt(d_k) is a scaling factor to prevent large dot products from pushing the softmax into regions with tiny gradients.
        -   softmax is applied row-wise across the L_K dimension, ensuring attention weights sum to 1 for each query token.
        -   The output is a weighted sum of the Value vectors, Z ∈ ℝ^(B × L_Q × d_v).
    -   **Multi-Head Attention (MHA):** Instead of performing a single attention function, MHA projects Q, K, and V h times with different, learned linear projections. The outputs from these h heads are concatenated and then linearly projected to the desired output dimension. This allows the model to jointly attend to information from different representation subspaces at different positions.
        MHA(Q, K, V) = Concat(head_1, ..., head_h) W_O
        where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
        -   W_Q_i, W_K_i, W_V_i ∈ ℝ^(d_model × d_k) are the projection matrices for the i-th head.
        -   W_O ∈ ℝ^(h * d_v × d_model) is the final linear projection matrix.

2.  **Positional Encoding (PE):** Since self-attention is permutation-invariant, positional information must be explicitly injected. This is typically done by adding a positional encoding vector to the input embeddings.
    -   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    -   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    -   Where pos is the position in the sequence and i is the dimension index. This allows the model to learn relative positions.

3.  **Feed-Forward Network (FFN):** Each Transformer block contains a position-wise FFN, applied independently to each position. It consists of two linear transformations with a non-linear activation function (e.g., ReLU, GELU) in between.
    FFN(x) = max(0, x W_1_FFN + b_1) W_2_FFN + b_2
    -   x ∈ ℝ^(B × L × d_model)
    -   W_1_FFN ∈ ℝ^(d_model × d_ff), W_2_FFN ∈ ℝ^(d_ff × d_model)
    -   b_1 ∈ ℝ^(d_ff), b_2 ∈ ℝ^(d_model)

4.  **Residual Connections and Layer Normalization:**
    -   **Residual Connections:** Add the input of a sub-layer to its output (x + Sublayer(x)). This helps mitigate vanishing gradients in deep networks.
    -   **Layer Normalization:** Normalizes the activations across the feature dimension for each sample independently, stabilizing training. Applied before or after the residual connection.

Mechanism and Formal Derivation

The distinction between Encoder-Only, Decoder-Only, and Encoder-Decoder models lies in how these core components are arranged and how attention is constrained.

1.  **Input Embedding and Positional Encoding:**
    Let X_input ∈ ℤ^(B × L) be a batch of token ID sequences.
    a.  **Token Embedding:** Each token ID is mapped to a dense vector.
        X_emb = E[X_input] ∈ ℝ^(B × L × d_model)
    b.  **Positional Encoding:** A positional encoding matrix PE ∈ ℝ^(L × d_model) is added to X_emb.
        X = X_emb + PE ∈ ℝ^(B × L × d_model)
        This X serves as the initial input to the first Transformer layer.

2.  **Query, Key, Value Projections (Multi-Head):**
    For each attention head h_i (where i ∈ {1, ..., h}), the input X is linearly projected to Q_i, K_i, V_i.
    a.  Q_i = X W_Q_i ∈ ℝ^(B × L × d_k)
    b.  K_i = X W_K_i ∈ ℝ^(B × L × d_k)
    c.  V_i = X W_V_i ∈ ℝ^(B × L × d_v)
    Here, W_Q_i, W_K_i, W_V_i ∈ ℝ^(d_model × d_k) are distinct weight matrices for each head.

3.  **Scaled Dot-Product Attention (Per Head):**
    For each head h_i, the attention output Z_i is computed:
    a.  **Attention Scores:** S_i = Q_i K_i^T ∈ ℝ^(B × L × L)
    b.  **Scaling:** S_scaled_i = S_i / sqrt(d_k) ∈ ℝ^(B × L × L)
    c.  **Softmax:** A_i = softmax(S_scaled_i) ∈ ℝ^(B × L × L) (row-wise over the last dimension)
    d.  **Weighted Sum:** Z_i = A_i V_i ∈ ℝ^(B × L × d_v)

4.  **Multi-Head Attention Output:**
    The outputs from all h heads are concatenated and then projected.
    a.  **Concatenation:** Z_concat = Concat(Z_1, ..., Z_h) ∈ ℝ^(B × L × (h * d_v))
    b.  **Output Projection:** MHA_output = Z_concat W_O ∈ ℝ^(B × L × d_model)
    Where W_O ∈ ℝ^((h * d_v) × d_model).

5.  **Encoder Block Structure (Bidirectional Context):**
    An encoder block takes an input X_enc ∈ ℝ^(B × L_enc × d_model) and produces an output H_enc_layer ∈ ℝ^(B × L_enc × d_model).
    a.  **Self-Attention Sub-layer:**
        X_attn = LayerNorm(X_enc + MHA(X_enc, X_enc, X_enc))
        Here, the MHA allows each token to attend to all other tokens (including itself) in the input sequence, providing a bidirectional context.
    b.  **Feed-Forward Sub-layer:**
        H_enc_layer = LayerNorm(X_attn + FFN(X_attn))
    c.  **Encoder-Only Models:** Consist of a stack of N_enc identical encoder blocks. The final output H_enc ∈ ℝ^(B × L_enc × d_model) is a rich, context-aware representation of the input sequence. This output is typically used for tasks like classification, sequence labeling, or feature extraction.

6.  **Decoder Block Structure (Causal Self-Attention and Cross-Attention):**
    A decoder block takes an input X_dec ∈ ℝ^(B × L_dec × d_model) (for self-attention) and the encoder's output H_enc ∈ ℝ^(B × L_enc × d_model) (for cross-attention).
    a.  **Masked Multi-Head Self-Attention Sub-layer:**
        X_causal_attn = LayerNorm(X_dec + MHA_causal(X_dec, X_dec, X_dec))
        The key difference here is `MHA_causal`. During the calculation of attention scores S_scaled_i (Step 3b), a causal mask M ∈ {0, -∞}^(L_dec × L_dec) is applied. M is a lower triangular matrix (or upper triangular, depending on convention) where M_ij = 0 if j <= i (allowing attention to past/current tokens) and M_ij = -∞ if j > i (masking future tokens).
        S_masked_i = S_scaled_i + M
        This ensures that predictions for a token at position `t` can only depend on known outputs at positions less than `t`.
    b.  **Multi-Head Cross-Attention Sub-layer (for Encoder-Decoder models):**
        X_cross_attn = LayerNorm(X_causal_attn + MHA_cross(X_causal_attn, H_enc, H_enc))
        Here, the Queries are derived from the decoder's previous layer output (X_causal_attn), while Keys and Values are derived from the encoder's final output (H_enc). This allows the decoder to attend to the entire encoded input sequence, conditioning its generation on the source context.
        -   Q_cross_i = X_causal_attn W_Q_cross_i ∈ ℝ^(B × L_dec × d_k)
        -   K_cross_i = H_enc W_K_cross_i ∈ ℝ^(B × L_enc × d_k)
        -   V_cross_i = H_enc W_V_cross_i ∈ ℝ^(B × L_enc × d_v)
        The attention calculation proceeds as in Step 3, but with L_Q = L_dec and L_K = L_enc.
    c.  **Feed-Forward Sub-layer:**
        H_dec_layer = LayerNorm(X_cross_attn + FFN(X_cross_attn)) (for Encoder-Decoder)
        H_dec_layer = LayerNorm(X_causal_attn + FFN(X_causal_attn)) (for Decoder-Only)
    d.  **Decoder-Only Models:** Consist of a stack of N_dec decoder blocks, but *without* the cross-attention sub-layer (Step 6b). They only use masked self-attention and FFNs. The final output H_dec ∈ ℝ^(B × L_dec × d_model) is typically followed by a linear layer and softmax to predict the next token in an autoregressive manner.
    e.  **Encoder-Decoder Models:** Combine N_enc encoder blocks and N_dec decoder blocks. The decoder blocks receive the final output of the encoder stack (H_enc) to perform cross-attention. The final output H_dec is used for sequence generation.

Computational and Complexity Analysis

The computational and memory complexity of Transformer models are dominated by the attention mechanism, particularly its quadratic dependency on sequence length.

1.  **Time Complexity (per layer):**
    a.  **Multi-Head Attention (MHA):**
        -   Projection of Q, K, V: O(L * d_model * d_k * h) = O(L * d_model^2) (since d_k * h = d_model).
        -   QK^T (attention scores): O(L^2 * d_k * h) = O(L^2 * d_model).
        -   Softmax and AV: O(L^2 * d_v * h) = O(L^2 * d_model).
        -   Output projection: O(L * d_model * d_model) = O(L * d_model^2).
        -   Total MHA: O(L^2 * d_model + L * d_model^2).
    b.  **Feed-Forward Network (FFN):**
        -   Two linear layers: O(L * d_model * d_ff + L * d_ff * d_model) = O(L * d_model * d_ff).
        -   Typically d_ff = 4 * d_model, so O(L * d_model^2).
    c.  **Total per Encoder/Decoder Self-Attention Layer:** O(L^2 * d_model + L * d_model^2).
    d.  **Cross-Attention Layer (in Decoder):**
        -   Q from decoder (L_dec), K/V from encoder (L_enc).
        -   QK^T: O(L_dec * L_enc * d_model).
        -   AV: O(L_dec * L_enc * d_model).
        -   Total Cross-Attention: O(L_dec * L_enc * d_model + L_dec * d_model^2).

    **Overall Model Complexity:**
    -   **Encoder-Only (N_enc layers):** O(N_enc * (L_enc^2 * d_model + L_enc * d_model^2)).
    -   **Decoder-Only (N_dec layers):** O(N_dec * (L_dec^2 * d_model + L_dec * d_model^2)).
    -   **Encoder-Decoder (N_enc encoder, N_dec decoder layers):**
        O(N_enc * (L_enc^2 * d_model + L_enc * d_model^2) + N_dec * (L_dec^2 * d_model + L_dec * d_model^2 + L_dec * L_enc * d_model)).
    The quadratic dependency on sequence length (L^2) is the primary bottleneck for long sequences.

2.  **Memory Complexity (per layer):**
    a.  **Parameters:**
        -   MHA weights (Q, K, V, O): O(d_model^2).
        -   FFN weights: O(d_model * d_ff) = O(d_model^2).
        -   Total parameters per layer: O(d_model^2).
        -   Total model parameters: O(N * d_model^2).
    b.  **Activ