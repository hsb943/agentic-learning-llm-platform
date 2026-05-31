Title
Probability Theory in Language Modeling

Assumptions and Notation
A language model (LM) is a probabilistic model that assigns a probability to a sequence of tokens. The fundamental assumption is that a sequence of tokens S = (t_1, t_2, ..., t_N) can be modeled as a realization from a probability distribution P(S). The model operates over a predefined vocabulary V, where each token t_i ∈ V. The size of the vocabulary is denoted by |V|.

Key Variables and Dimensions:
*   t_i: An individual token from the vocabulary V.
*   S = (t_1, ..., t_N): A sequence of N tokens.
*   N: Sequence length, the number of tokens in a sequence.
*   |V|: Vocabulary size, the total number of unique tokens.
*   d_model: Dimensionality of the token embeddings and hidden states within the model.
*   B: Batch size, the number of sequences processed concurrently.
*   W_emb ∈ ℝ^(|V| × d_model): Embedding matrix, mapping tokens to their vector representations.
*   X_emb ∈ ℝ^(N × d_model) or ℝ^(B × N × d_model): Embedded input sequence(s).
*   H ∈ ℝ^(N × d_model) or ℝ^(B × N × d_model): Contextualized hidden states, output of the core model architecture.
*   W_proj ∈ ℝ^(d_model × |V|): Output projection matrix, mapping hidden states to vocabulary-sized logits.
*   b_proj ∈ ℝ^(|V|): Output projection bias vector.
*   Z ∈ ℝ^(N × |V|) or ℝ^(B × N × |V|): Logits, unnormalized scores for each token in the vocabulary at each position.
*   P ∈ ℝ^(N × |V|) or ℝ^(B × N × |V|): Probability distribution over the vocabulary for each position.
*   θ: Set of all learnable parameters of the language model.
*   L_CE: Cross-entropy loss.
*   PP: Perplexity.

Core Concepts and Mathematical Foundations
Formal Definitions
1.  Language Model Probability: A language model defines a probability distribution over sequences of tokens. By the chain rule of probability, the probability of a sequence S = (t_1, ..., t_N) is decomposed as:
    P(S) = P(t_1, ..., t_N) = P(t_1) * P(t_2 | t_1) * P(t_3 | t_1, t_2) * ... * P(t_N | t_1, ..., t_{N-1})
    This can be written compactly as:
    P(S) = Π_{i=1}^{N} P(t_i | t_1, ..., t_{i-1})
    For neural language models, P(t_i | t_1, ..., t_{i-1}) is typically parameterized by a neural network, denoted as P_θ(t_i | t_1, ..., t_{i-1}).

2.  Conditional Probability of Next Token: The core task of a language model is to predict the next token t_i given the preceding context (t_1, ..., t_{i-1}). This is represented by the conditional probability P(t_i | context_i), where context_i = (t_1, ..., t_{i-1}).

3.  Softmax Function: The softmax function transforms a vector of arbitrary real-valued scores (logits) into a probability distribution. For a vector of logits z = (z_1, ..., z_k), the softmax output p = (p_1, ..., p_k) is given by:
    p_j = exp(z_j) / Σ_{k=1}^{K} exp(z_k)
    where Σ_{j=1}^{K} p_j = 1 and p_j > 0 for all j. In language modeling, K = |V|.

4.  Cross-Entropy Loss: For a given position i in a sequence, let y_i be the true one-hot encoded distribution (a vector of zeros with a one at the index of the true next token t'_i) and p_i be the predicted probability distribution over the vocabulary. The cross-entropy loss for this position is:
    L_CE(y_i, p_i) = - Σ_{k=1}^{|V|} y_i(k) * log(p_i(k))
    Since y_i is one-hot, this simplifies to L_CE(y_i, p_i) = - log(p_i(t'_i)), where p_i(t'_i) is the predicted probability of the true next token. For a sequence of N tokens, the average cross-entropy loss is:
    L_CE = - (1/N) Σ_{i=1}^{N} log(P_θ(t'_i | t_1, ..., t_{i-1}))
    This is equivalent to the negative average log-likelihood of the observed sequence.

5.  Perplexity: Perplexity is a measure of how well a probability distribution predicts a sample. It is defined as the exponential of the average cross-entropy loss:
    PP = exp(L_CE) = exp(- (1/N) Σ_{i=1}^{N} log(P_θ(t'_i | t_1, ..., t_{i-1})))
    A lower perplexity indicates a better model, as it implies the model assigns higher probabilities to the observed tokens. Perplexity can be interpreted as the weighted average number of choices the model has for each token.

Geometric or Probabilistic Interpretation
*   **Probability Distribution**: A probability distribution over |V| tokens can be visualized as a point within a (|V|-1)-dimensional simplex. Each vertex of the simplex corresponds to assigning probability 1 to a single token and 0 to all others. The softmax function maps the entire ℝ^|V| space (logits) onto this simplex.
*   **Logits**: Logits (z_k) can be interpreted as unnormalized log-probabilities. The difference z_j - z_k directly relates to the log-ratio of probabilities p_j / p_k.
*   **Cross-Entropy**: Cross-entropy measures the "distance" or divergence between two probability distributions. Specifically, it quantifies the average number of bits required to identify an event from a true distribution, given a model that assumes a different distribution. Minimizing cross-entropy is equivalent to maximizing the log-likelihood of the observed data under the model.

Dimensional Reasoning
*   Input tokens are discrete indices. They are mapped to continuous vectors via an embedding lookup.
*   The embedding matrix W_emb ∈ ℝ^(|V| × d_model) maps a one-hot vector of dimension |V| to a d_model-dimensional vector.
*   A sequence of N tokens, when embedded, forms a matrix X_emb ∈ ℝ^(N × d_model). If processed in batches, it becomes ℝ^(B × N × d_model).
*   The core model architecture (e.g., Transformer blocks) takes X_emb and produces contextualized representations H ∈ ℝ^(N × d_model) (or ℝ^(B × N × d_model)), preserving the sequence length and embedding dimension.
*   The output projection layer maps these d_model-dimensional hidden states back to |V|-dimensional logits. This involves a matrix multiplication H W_proj, where W_proj ∈ ℝ^(d_model × |V|), resulting in Z ∈ ℝ^(N × |V|) (or ℝ^(B × N × |V|)).
*   The softmax function is applied row-wise to Z, producing P ∈ ℝ^(N × |V|) (or ℝ^(B × N × |V|)), where each row sums to 1.
*   The loss function then operates on these probability distributions and the true next tokens, yielding a scalar loss value.

Mechanism and Formal Derivation
The process of a neural language model generating a probability distribution over the next token, given a sequence of preceding tokens, involves several distinct computational steps. We consider a single sequence for clarity; batch processing extends these operations trivially.

Step 1: Token Embedding
Input: A sequence of N discrete tokens S = (t_1, ..., t_N), where each t_i is an integer index from 0 to |V|-1.
Operation: Each token t_i is mapped to a continuous vector representation using an embedding matrix W_emb.
Formalism: Let W_emb ∈ ℝ^(|V| × d_model) be the embedding matrix. For each token t_i, its embedding vector e_i is the t_i-th row of W_emb.
X_emb = (e_1, ..., e_N)
Dimensionality: X_emb ∈ ℝ^(N × d_model).

Step 2: Contextual Representation Generation
Input: The embedded sequence X_emb ∈ ℝ^(N × d_model).
Operation: The model's core architecture (e.g., a stack of Transformer layers) processes X_emb to generate contextualized representations. This process integrates information from preceding tokens to enrich each token's representation. For a Transformer, this involves self-attention mechanisms, feed-forward networks, and residual connections.
Formalism: Let f_model be the function representing the entire stack of contextualization layers.
H = f_model(X_emb)
Dimensionality: H ∈ ℝ^(N × d_model). Each row h_i ∈ ℝ^(d_model) is the contextualized representation for the token at position i, incorporating information from t_1, ..., t_i (and potentially future tokens if not masked for causal language modeling). For causal LMs, h_i depends only on t_1, ..., t_i.

Step 3: Projection to Logits
Input: The contextualized hidden states H ∈ ℝ^(N × d_model).
Operation: Each contextualized hidden state h_i is projected into a vector of scores (logits) over the entire vocabulary. This is typically achieved via a linear transformation.
Formalism: Let W_proj ∈ ℝ^(d_model × |V|) be the output projection matrix and b_proj ∈ ℝ^(|V|) be the bias vector. For each position i, the logits z_i for the next token are calculated as:
z_i = h_i W_proj + b_proj
The full matrix of logits Z for the sequence is:
Z = H W_proj + 1_N b_proj^T
where 1_N is a column vector of N ones, and b_proj^T is the transpose of b_proj.
Dimensionality: Z ∈ ℝ^(N × |V|). Each row z_i ∈ ℝ^(|V|) contains the unnormalized log-probabilities for the token at position i to be any token in the vocabulary, given its context.

Step 4: Probability Distribution Calculation (Softmax)
Input: The logits Z ∈ ℝ^(N × |V|).
Operation: The softmax function is applied independently to each row of Z to convert the logits into a valid probability distribution over the vocabulary for each position.
Formalism: For each position i from 1 to N, and for each vocabulary token k from 1 to |V|:
P(t_k | context_i) = p_i(k) = exp(z_i(k)) / Σ_{j=1}^{|V|} exp(z_i(j))
This results in a matrix P where each row p_i is a probability distribution.
Dimensionality: P ∈ ℝ^(N × |V|). Each row p_i sums to 1.

Step 5: Likelihood Calculation
Input: The predicted probability distributions P ∈ ℝ^(N × |V|) and the true next tokens T' = (t'_1, ..., t'_N). Note that t'_i is the actual token that *should* appear at position i, which is typically t_{i+1} from the input sequence S, shifted by one position. For causal LMs, the model predicts t_{i+1} given t_1, ..., t_i.
Operation: The likelihood of the true next token at each position is extracted from the predicted distribution. The overall log-likelihood of the sequence is the sum of the log-likelihoods of individual tokens.
Formalism: For each position i, the probability of the true next token t'_i is P(t'_i | context_i) = p_i(t'_i).
The log-likelihood for the entire sequence T' given the input S is:
log P(T' | S) = Σ_{i=1}^{N} log(p_i(t'_i))
Dimensionality: The log-likelihood is a scalar value.

Step 6: Loss Function (Cross-Entropy)
Input: The log-likelihoods of the true next tokens.
Operation: The negative average log-likelihood is computed, which serves as the cross-entropy loss. This loss is minimized during training to optimize the model parameters θ.
Formalism:
L_CE = - (1/N) Σ_{i=1}^{N} log(p_i(t'_i))
This is the objective function for training.
Dimensionality: L_CE is a scalar value.

Computational and Complexity Analysis
The computational and memory complexity of language models are critical considerations, especially with scaling model size and input sequence length. We analyze these for a single forward pass.

Time Complexity
1.  **Token Embedding**:
    *   Operation: Lookup in W_emb.
    *   Complexity: O(N * d_model) for N tokens, as each lookup takes O(d_model) time (assuming efficient memory access).
2.  **Contextual Representation Generation (e.g., Transformer Block)**:
    *   Self-Attention: For a single attention head, computing query, key, value matrices is O(N * d_model^2). Computing attention scores (Q K^T) is O(N^2 * d_model). Computing output (Attention * V) is O(N^2 * d_model). If there are H heads, this is O(H * N^2 * d_model / H) = O(N^2 * d_model).
    *   Feed-Forward Networks (FFN): Typically two linear layers. O(N * d_model * d_ff + N * d_ff * d_model) = O(N * d_model * d_ff), where d_ff is the FFN hidden dimension (often 4 * d_model). So, O(N * d_model^2).
    *   Total per layer: O(N^2 * d_model + N * d_model^2). If N > d_model, attention dominates, leading to O(N^2 * d_model). If d_model > N, FFN might dominate.
    *   For L layers: O(L * (N^2 * d_model + N * d_model^2)).
3.  **Projection to Logits**:
    *   Operation: Matrix multiplication H W_proj.
    *   Complexity: O(N * d_model * |V|).
4.  **Softmax**:
    *   Operation: Exponentiation and summation for each row of Z.
    *   Complexity: O(N * |V|) for N rows, each of size |V|.
5.  **Loss Calculation**:
    *   Operation: Indexing and summation of log probabilities.
    *   Complexity: O(N).

Overall Time Complexity: The dominant terms are typically O(L * N^2 * d_model) from attention and O(N * d_model * |V|) from the output projection.
*   If |V| is very large and L*N is small, O(N * d_model * |V|) dominates.
*   If N is very large, O(L * N^2 * d_model) dominates.
*   For typical large LMs, both are significant.

Memory Complexity
1.  **Model Parameters**:
    *   Embedding matrix W_emb: O(|V| * d_model).
    *   Transformer layers (weights for Q, K, V, FFNs): O(L * d_model^2).
    *   Output projection W_proj: O(d_model * |V|).
    *   Total parameters: O(|V| * d_model + L * d_model^2 + d_model * |V|) = O(d_model * (|V| + L * d_model)).
2.  **Activations (during forward pass)**:
    *   Input embeddings X_emb: O(N * d_model).
    *   Hidden states H: O(N * d_model) per layer (if not storing all intermediate layer outputs).
    *   Logits Z: O(N * |V|).
    *   Total activations: O(N * d_model + N * |V|).

Overall Memory Complexity: The dominant terms are O(d_model * |V|) for parameters and O(N * |V|) for activations.
*   For very large vocabularies, the embedding and output projection matrices dominate