Positional Encoding: Sinusoidal, Learned, RoPE, and ALiBi

Assumptions and Notation
The following variables and notations are used consistently throughout this document:
L: Sequence length, representing the number of tokens in an input sequence.
L_max: Maximum sequence length supported by a model or embedding table.
d_model: Dimensionality of the model's hidden states or token embeddings.
d_k: Dimensionality of query (Q) and key (K) vectors in an attention mechanism. Typically, d_k = d_model / num_heads.
pos: An integer representing the absolute position of a token within a sequence, 0 <= pos < L.
i: An integer representing the index of a dimension within a vector, 0 <= i < d_model or 0 <= i < d_k.
k: An integer representing a frequency index or a pair index for sinusoidal functions.
X: Input token embeddings, X ∈ ℝ^(L × d_model).
P: Positional encoding matrix or vector, P ∈ ℝ^(L × d_model) or P ∈ ℝ^(d_model).
Q: Query matrix, Q ∈ ℝ^(L × d_k).
K: Key matrix, K ∈ ℝ^(L × d_k).
V: Value matrix, V ∈ ℝ^(L × d_k).
R_pos(x): A function or operator that applies a positional transformation to a vector x.
m: A scalar slope parameter used in ALiBi, typically positive.
theta_j: A frequency parameter for RoPE, specific to dimension j.

Core Concepts and Mathematical Foundations
Positional encoding mechanisms are essential in transformer architectures to inject information about the relative or absolute position of tokens in a sequence, as the self-attention mechanism is inherently permutation-invariant. Without positional information, the model cannot distinguish the order of tokens.

Sinusoidal Positional Encoding
Formal Definition: For a given position pos and dimension i, the sinusoidal positional encoding vector PE_pos ∈ ℝ^(d_model) is defined as:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
where 0 <= pos < L and 0 <= i < d_model/2. If d_model is odd, the last dimension might be handled by either sin or cos.
Geometric or Probabilistic Interpretation: This formulation allows the model to easily learn to attend to relative positions. The use of sine and cosine functions at different frequencies enables the representation of relative positions as a linear function in the high-dimensional space. Specifically, for any fixed offset k, PE(pos+k) can be expressed as a linear transformation of PE(pos) using trigonometric identities (sin(A+B) = sinAcosB + cosAsinB, cos(A+B) = cosAcosB - sinAsinB). This property is crucial for generalization to unseen sequence lengths.
Dimensional Reasoning: The term 10000^(2i/d_model) ensures that different dimensions correspond to different wavelengths, ranging from 2π (for i=0) to 2π * 10000 (for i=d_model/2 - 1). This allows the model to capture both fine-grained and coarse-grained positional information. The division by d_model normalizes the frequency range across the embedding dimensions.

Learned Positional Encoding
Formal Definition: Learned positional encoding involves a simple embedding layer, E_pos ∈ ℝ^(L_max × d_model), where L_max is the maximum sequence length the model is designed to handle. For an input sequence of length L, the positional encoding P_learned ∈ ℝ^(L × d_model) is obtained by selecting the first L rows from E_pos:
P_learned = E_pos[0:L, :]
The final input to the transformer layer is X + P_learned.
Geometric or Probabilistic Interpretation: There is no inherent geometric interpretation beyond being a direct lookup table. Each position pos is mapped to a unique, trainable vector. The model learns the optimal positional representations directly from data.
Dimensional Reasoning: The embedding table E_pos has L_max distinct vectors, each of dimension d_model. The selection process simply aligns the learned positional vectors with the input sequence tokens.

Rotary Positional Embedding (RoPE)
Formal Definition: RoPE applies a rotation to the query (Q) and key (K) vectors based on their absolute position before computing the attention scores. For a vector x ∈ ℝ^(d_k) at position pos, the rotary transformation R_pos(x) is defined by pairing dimensions and applying 2D rotations. For each pair of dimensions (j, j+1) where j is even:
R_pos(x)_j = x_j cos(pos * theta_j) - x_{j+1} sin(pos * theta_j)
R_pos(x)_{j+1} = x_j sin(pos * theta_j) + x_{j+1} cos(pos * theta_j)
where theta_j = 1 / 10000^(j/d_k). This can be compactly represented using complex numbers or rotation matrices.
The core property of RoPE is that the dot product between two rotated vectors depends only on their relative position:
(R_m(q))ᵀ (R_n(k)) = qᵀ R_{n-m}(k)
Geometric or Probabilistic Interpretation: RoPE explicitly encodes relative positional information by rotating query and key vectors in a way that their dot product naturally captures relative distance. Each pair of dimensions (j, j+1) forms a 2D plane where rotation occurs. The frequencies theta_j vary across these planes, allowing different dimensions to encode different scales of relative distance.
Dimensional Reasoning: d_k must be even for this pairwise rotation scheme. The frequencies theta_j are chosen similarly to sinusoidal PE, ensuring a range of wavelengths for positional encoding. The rotation is applied element-wise to the d_k dimensions of Q and K vectors.

Attention with Linear Biases (ALiBi)
Formal Definition: ALiBi modifies the attention mechanism by adding a static, non-learned bias directly to the attention scores before the softmax operation. For an attention head, the attention score between query at position i and key at position j is given by:
Attention_score(i, j) = (Q_i K_jᵀ / sqrt(d_k)) + m * (j - i)
where Q_i and K_j are the query and key vectors for positions i and j, respectively. The term m is a scalar slope parameter, typically negative, and often different for each attention head. The original paper uses m * |i - j| with a negative m, or -m * |i - j| with a positive m. We use the latter for consistency with penalizing distance.
Geometric or Probabilistic Interpretation: ALiBi introduces a linear penalty for attending to tokens that are far away. The bias term m * (j - i) directly encodes the relative distance between tokens i and j. A negative slope m (or positive m in -m*|i-j|) means that as the distance |i-j| increases, the attention score decreases, making it harder for the model to attend to distant tokens. This creates a strong inductive bias towards local attention.
Dimensional Reasoning: The bias m * (j - i) is a scalar added to each element of the attention score matrix S ∈ ℝ^(L × L). The slope m is typically a hyperparameter or learned per head. The term (j - i) represents the signed relative distance.

Mechanism and Formal Derivation

Sinusoidal Positional Encoding
1.  Define the absolute position index pos ∈ [0, L-1] and the dimension index i ∈ [0, d_model-1].
2.  Define the frequency scaling factor for each pair of dimensions. For a given dimension pair (2k, 2k+1), the frequency parameter is omega_k = 1 / 10000^(2k/d_model).
3.  The even-indexed dimension of the positional encoding vector PE_pos is computed as PE(pos, 2k) = sin(pos * omega_k).
4.  The odd-indexed dimension of the positional encoding vector PE_pos is computed as PE(pos, 2k+1) = cos(pos * omega_k).
5.  The full positional encoding vector for position pos is PE_pos ∈ ℝ^(d_model). This vector is then added to the input token embedding X_pos ∈ ℝ^(d_model) to produce the final input to the transformer layer: X'_pos = X_pos + PE_pos.
6.  Crucially, for any relative offset delta, PE(pos + delta) can be expressed as a linear transformation of PE(pos). Using trigonometric identities:
    PE(pos + delta, 2k) = sin((pos + delta) * omega_k) = sin(pos * omega_k)cos(delta * omega_k) + cos(pos * omega_k)sin(delta * omega_k)
    PE(pos + delta, 2k+1) = cos((pos + delta) * omega_k) = cos(pos * omega_k)cos(delta * omega_k) - sin(pos * omega_k)sin(delta * omega_k)
    This shows that relative positions are encoded as a linear combination of absolute positions, enabling the model to generalize to longer sequences.

Learned Positional Encoding
1.  Initialize an embedding matrix E_pos ∈ ℝ^(L_max × d_model) with trainable parameters.
2.  For an input sequence of length L, where L <= L_max, retrieve the first L rows from E_pos. Let this be P_learned ∈ ℝ^(L × d_model).
3.  The input token embeddings X ∈ ℝ^(L × d_model) are obtained from a standard token embedding layer.
4.  The positional encoding is applied by element-wise addition: X'_pos = X_pos + P_learned_pos for each position pos.
5.  The resulting matrix X' ∈ ℝ^(L × d_model) is then passed to the subsequent transformer layers.
6.  There is no further mathematical derivation for its mechanism beyond this direct addition. Its effectiveness relies entirely on the learned values within E_pos.

Rotary Positional Embedding (RoPE)
1.  Let q_m ∈ ℝ^(d_k) and k_n ∈ ℝ^(d_k) be the query and key vectors for positions m and n, respectively.
2.  Define the frequency parameters theta_j = 1 / 10000^(j/d_k) for j ∈ [0, d_k/2 - 1].
3.  For each vector x ∈ ℝ^(d_k) at position pos, apply the rotation R_pos(x). This can be represented as a block-diagonal matrix multiplication or element-wise operations. For each pair of dimensions (j, j+1) where j is even:
    [ R_pos(x)_j   ] = [ cos(pos * theta_{j/2})  -sin(pos * theta_{j/2}) ] [ x_j   ]
    [ R_pos(x)_{j+1} ]   [ sin(pos * theta_{j/2})   cos(pos * theta_{j/2}) ] [ x_{j+1} ]
    Note: The original RoPE paper uses theta_j for j=0 to d_k/2-1, so the index for theta should be j/2.
4.  The transformed query and key vectors are R_m(q_m) and R_n(k_n).
5.  The attention score between these transformed vectors is computed as their dot product: (R_m(q_m))ᵀ (R_n(k_n)).
6.  Derivation of the relative position property:
    Let q = [q_0, q_1, ..., q_{d_k-1}]ᵀ and k = [k_0, k_1, ..., k_{d_k-1}]ᵀ.
    For a 2D sub-vector [q_j, q_{j+1}]ᵀ and [k_j, k_{j+1}]ᵀ, the rotated components are:
    R_m(q)_j = q_j cos(m * theta_{j/2}) - q_{j+1} sin(m * theta_{j/2})
    R_m(q)_{j+1} = q_j sin(m * theta_{j/2}) + q_{j+1} cos(m * theta_{j/2})
    Similarly for R_n(k).
    The dot product of these 2D components is:
    (R_m(q)_j R_n(k)_j) + (R_m(q)_{j+1} R_n(k)_{j+1})
    = (q_j cos(mθ) - q_{j+1} sin(mθ))(k_j cos(nθ) - k_{j+1} sin(nθ)) + (q_j sin(mθ) + q_{j+1} cos(mθ))(k_j sin(nθ) + k_{j+1} cos(nθ))
    Expanding and simplifying using cos(A)cos(B) + sin(A)sin(B) = cos(A-B) and cos(A)sin(B) - sin(A)cos(B) = sin(B-A):
    = q_j k_j (cos(mθ)cos(nθ) + sin(mθ)sin(nθ)) + q_{j+1} k_{j+1} (sin(mθ)sin(nθ) + cos(mθ)cos(nθ))
      + q_{j+1} k_j (-sin(mθ)cos(nθ) + cos(mθ)sin(nθ)) + q_j k_{j+1} (-cos(mθ)sin(nθ) + sin(mθ)cos(nθ))
    = q_j k_j cos((m-n)θ) + q_{j+1} k_{j+1} cos((m-n)θ) + q_{j+1} k_j sin((n-m)θ) + q_j k_{j+1} sin((m-n)θ)
    = (q_j k_j + q_{j+1} k_{j+1}) cos((m-n)θ) + (q_j k_{j+1} - q_{j+1} k_j) sin((m-n)θ)
    This is equivalent to the dot product of q with R_{n-m}(k), demonstrating that the attention score depends only on the relative distance (n-m).

Attention with Linear Biases (ALiBi)
1.  Compute the standard scaled dot-product attention scores S ∈ ℝ^(L × L) for a given attention head:
    S_{i,j} = Q_i K_jᵀ / sqrt(d_k)
    where Q_i ∈ ℝ^(d_k) is the query vector for position i, and K_j ∈ ℝ^(d_k) is the key vector for position j.
2.  Define the bias matrix M ∈ ℝ^(L × L) where each element M_{i,j} is determined by the relative distance between positions i and j.
3.  For a given attention head, a scalar slope parameter m is chosen (typically negative, or positive if using -m*|i-j|).
4.  The bias for each (i, j) pair is calculated as M_{i,j} = m * (j - i). Note that the original ALiBi paper uses m * |i - j| with m < 0, or -m * |i - j| with m > 0. We will use the latter form for clarity: M_{i,j} =