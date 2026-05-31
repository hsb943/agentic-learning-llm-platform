Title
Query, Key, and Value (QKV) Projections

Assumptions and Notation
This document assumes a foundational understanding of linear algebra, matrix operations, and basic neural network concepts. The primary context for QKV projections is within self-attention mechanisms, particularly in transformer architectures, though the mechanism itself is a general linear transformation.

The following variables and notation are used consistently:

*   X ∈ ℝ^(n × d_model): Input sequence of n token embeddings, where each token is represented by a d_model-dimensional vector.
    *   n: The sequence length, representing the number of tokens in the input.
    *   d_model: The dimensionality of the input and output embeddings of the overall model, often referred to as the model dimension.
*   W_Q ∈ ℝ^(d_model × d_k): The learnable weight matrix for the Query projection.
*   W_K ∈ ℝ^(d_model × d_k): The learnable weight matrix for the Key projection.
*   W_V ∈ ℝ^(d_model × d_v): The learnable weight matrix for the Value projection.
*   Q ∈ ℝ^(n × d_k): The resulting Query matrix, where each row is a d_k-dimensional query vector for a corresponding token.
*   K ∈ ℝ^(n × d_k): The resulting Key matrix, where each row is a d_k-dimensional key vector for a corresponding token.
*   V ∈ ℝ^(n × d_v): The resulting Value matrix, where each row is a d_v-dimensional value vector for a corresponding token.
*   d_k: The dimensionality of the Query and Key vectors. This is a hyperparameter, often set as d_model / h, where h is the number of attention heads.
*   d_v: The dimensionality of the Value vectors. This is a hyperparameter, often set as d_model / h.
*   h: The number of attention heads in a multi-head attention mechanism. While QKV projections are defined per head, this document focuses on the single-head projection for clarity, with d_k and d_v representing the per-head dimensions.

Core Concepts and Mathematical Foundations

Formal Definitions
Query, Key, and Value (QKV) projections are linear transformations applied to an input representation (typically a sequence of token embeddings) to generate three distinct sets of vectors: Queries (Q), Keys (K), and Values (V). These transformations are parameterized by learnable weight matrices, W_Q, W_K, and W_V, respectively.

1.  **Query (Q)**: The Query matrix is derived by multiplying the input sequence X by the Query weight matrix W_Q. Each row of Q represents a "query" vector for the corresponding input token. Conceptually, a query vector encodes "what information I am looking for" from other tokens in the sequence.
    Q = X W_Q

2.  **Key (K)**: The Key matrix is derived by multiplying the input sequence X by the Key weight matrix W_K. Each row of K represents a "key" vector for the corresponding input token. Conceptually, a key vector encodes "what information I have to offer" to other tokens in the sequence.
    K = X W_K

3.  **Value (V)**: The Value matrix is derived by multiplying the input sequence X by the Value weight matrix W_V. Each row of V represents a "value" vector for the corresponding input token. Conceptually, a value vector encodes "the content or information associated with what I have to offer" that should be retrieved if its key matches a query.
    V = X W_V

These projections are fundamental to attention mechanisms, where the similarity between queries and keys determines the attention weights, which are then used to compute a weighted sum of values.

Geometric or Probabilistic Interpretation
Geometrically, QKV projections transform the input embeddings from the d_model-dimensional space into three potentially different, specialized vector spaces: a query space, a key space, and a value space.

*   **Query Space (ℝ^d_k)**: Vectors in this space are designed to probe for relevance. The direction and magnitude of a query vector indicate the specific features or patterns it is seeking.
*   **Key Space (ℝ^d_k)**: Vectors in this space are designed to be matched against queries. The direction and magnitude of a key vector indicate the specific features or patterns it possesses and can offer. The dot product between a query vector and a key vector (q_i ⋅ k_j) serves as a measure of their semantic or contextual similarity. A larger dot product implies greater relevance or alignment between what token i is looking for and what token j has to offer.
*   **Value Space (ℝ^d_v)**: Vectors in this space contain the actual information content that is to be aggregated. Unlike query and key spaces, which are primarily for determining relevance, the value space holds the payload. The dimensionality d_v can be different from d_k, allowing for a separation of the "matching" dimension from the "content" dimension.

Probabilistically, the similarity score (q_i ⋅ k_j) can be interpreted as an unnormalized log-probability or a measure of affinity. After normalization (e.g., via softmax), these scores become probabilities representing the likelihood that a query vector attends to a particular key-value pair.

Dimensional Reasoning
The choice of dimensions d_k and d_v is critical for the computational efficiency and expressivity of the attention mechanism.

*   **d_model**: This is the base dimensionality of the input tokens. It represents the richness of the initial embedding.
*   **d_k**: The dimensionality of query and key vectors. It is crucial that Q and K have the same dimensionality (d_k) because their dot product (q_i ⋅ k_j) is a core operation for computing attention scores. If d_k were different, the dot product would not be well-defined or would require padding/truncation, which is not standard. A smaller d_k than d_model can act as a bottleneck, forcing the model to learn more compact and salient features for relevance matching. A larger d_k increases the capacity for distinguishing fine-grained relationships but also increases computational cost.
*   **d_v**: The dimensionality of value vectors. d_v can be equal to d_k, equal to d_model, or any other chosen dimension. It determines the capacity of the information that can be retrieved and aggregated. If d_v is smaller than d_model, it implies a compression of the original token information into a more concise representation for the purpose of aggregation. If d_v is larger, it implies an expansion, though this is less common in practice. In multi-head attention, d_k and d_v are often set to d_model / h to maintain the overall dimensionality of the aggregated output across heads.

The consistent application of matrix multiplication ensures that the output dimensions are correctly formed:
*   X (n × d_model) * W_Q (d_model × d_k) = Q (n × d_k)
*   X (n × d_model) * W_K (d_model × d_k) = K (n × d_k)
*   X (n × d_model) * W_V (d_model × d_v) = V (n × d_v)

Mechanism and Formal Derivation

The QKV projection mechanism involves a series of linear transformations applied to an input sequence. These transformations are independent for Queries, Keys, and Values, each utilizing its own distinct set of learnable parameters.

Step 1: Input Sequence Representation
The process begins with an input sequence of n tokens, where each token is represented by a d_model-dimensional embedding vector. This sequence is typically arranged as a matrix X.
Let X be the input matrix: X ∈ ℝ^(n × d_model).
Each row x_i ∈ ℝ^(1 × d_model) of X corresponds to the embedding of the i-th token in the sequence.

Step 2: Definition of Projection Weight Matrices
Three distinct, learnable weight matrices are defined. These matrices are the parameters that the model learns during training to transform the input embeddings into specialized Q, K, and V representations.
*   Query Weight Matrix: W_Q ∈ ℝ^(d_model × d_k)
*   Key Weight Matrix: W_K ∈ ℝ^(d_model × d_k)
*   Value Weight Matrix: W_V ∈ ℝ^(d_model × d_v)
These matrices are initialized randomly and updated via backpropagation.

Step 3: Query Projection
The Query matrix Q is computed by performing a matrix multiplication of the input sequence matrix X with the Query weight matrix W_Q. This operation transforms each d_model-dimensional input token embedding into a d_k-dimensional query vector.
Formal operation: Q = X W_Q
Dimensional consistency: (n × d_model) ⋅ (d_model × d_k) = (n × d_k)
The resulting matrix Q contains n query vectors, where the i-th row q_i ∈ ℝ^(1 × d_k) is the query vector for the i-th input token.

Step 4: Key Projection
The Key matrix K is computed by performing a matrix multiplication of the input sequence matrix X with the Key weight matrix W_K. This operation transforms each d_model-dimensional input token embedding into a d_k-dimensional key vector.
Formal operation: K = X W_K
Dimensional consistency: (n × d_model) ⋅ (d_model × d_k) = (n × d_k)
The resulting matrix K contains n key vectors, where the i-th row k_i ∈ ℝ^(1 × d_k) is the key vector for the i-th input token.

Step 5: Value Projection
The Value matrix V is computed by performing a matrix multiplication of the input sequence matrix X with the Value weight matrix W_V. This operation transforms each d_model-dimensional input token embedding into a d_v-dimensional value vector.
Formal operation: V = X W_V
Dimensional consistency: (n × d_model) ⋅ (d_model × d_v) = (n × d_v)
The resulting matrix V contains n value vectors, where the i-th row v_i ∈ ℝ^(1 × d_v) is the value vector for the i-th input token.

Step 6: Resulting Projections
Upon completion of these three independent linear transformations, we obtain the Query, Key, and Value matrices:
*   Q ∈ ℝ^(n × d_k)
*   K ∈ ℝ^(n × d_k)
*   V ∈ ℝ^(n × d_v)
These matrices are then used in subsequent steps of an attention mechanism. For instance, in scaled dot-product attention, the attention scores are computed as Q K^T / sqrt(d_k), and the output is derived by multiplying these scores with V.

Step 7: Decoupling of Representation Spaces
A critical aspect of QKV projections is the decoupling of the input embedding space (d_model) into specialized query, key, and value spaces (d_k, d_v). This allows the model to learn distinct transformations for:
*   Identifying "what to look for" (Query).
*   Identifying "what is available" (Key).
*   Representing "the content to be retrieved" (Value).
This separation enhances the model's ability to focus on relevant information by learning specific feature extractors for each role, rather than using the raw input embeddings directly for all three purposes. The learned weight matrices W_Q, W_K, W_V enable this specialization.

Computational and Complexity Analysis

The computational and memory requirements of QKV projections are primarily determined by the matrix multiplication operations and the storage of the weight matrices and intermediate Q, K, V matrices.

Time Complexity
Each projection (Q, K, V) involves a matrix multiplication of the input sequence X (n × d_model) with its respective weight matrix (d_model × d_k or d_model × d_v).

1.  **Query Projection (Q = X W_Q)**:
    *   Operation: (n × d_model) ⋅ (d_model × d_k)
    *   Complexity: O(n ⋅ d_model ⋅ d_k) floating-point operations (FLOPs).

2.  **Key Projection (K = X W_K)**:
    *   Operation: (n × d_model) ⋅ (d_model × d_k)
    *   Complexity: O(n ⋅ d_model ⋅ d_k) FLOPs.

3.  **Value Projection (V = X W_V)**:
    *   Operation: (n × d_model) ⋅ (d_model × d_v)
    *   Complexity: O(n ⋅ d_model ⋅ d_v) FLOPs.

**Total Time Complexity for QKV Projections**:
The total computational cost for generating Q, K, and V matrices is the sum of the complexities for each projection:
O(n ⋅ d_model ⋅ d_k + n ⋅ d_model ⋅ d_k + n ⋅ d_model ⋅ d_v)
= O(n ⋅ d_model ⋅ (2d_k + d_v))

In many transformer implementations, d_k = d_v = d_model / h. Substituting these values:
O(n ⋅ d_model ⋅ (2(d_model/h) + (d_model/h))) = O(n ⋅ d_model ⋅ (3d_model/h))
= O(n ⋅ d_model^2 / h)

Memory Complexity
Memory complexity involves storing the learnable weight matrices and the resulting Q, K, V matrices.

1.  **Weight Matrices (W_Q, W_K, W_V)**:
    *   W_Q: d_model × d_k parameters
    *   W_K: d_model × d_k parameters
    *   W_V: d_model × d_v parameters
    *   Total parameters: O(d_model ⋅ d_k + d_model ⋅ d_k + d_model ⋅ d_v) = O(d_model ⋅ (2d_k + d_v))
    *   Memory for weights: O(d_model ⋅ (2d_k + d_v))

2.  **Projected Matrices (Q, K, V)**:
    *   Q: n × d_k elements
    *   K: n × d_k elements
    *   V: n × d_v elements
    *   Total elements: O(n ⋅ d_k + n ⋅ d_k + n ⋅ d_v) = O(n ⋅ (2d_k + d_v))
    *   Memory for projected matrices: O(n ⋅ (2d_k + d_v))

**Total Memory Complexity**:
The total memory required is the sum of memory for weights and projected matrices:
O(d_model ⋅ (2d_k + d_v) + n ⋅ (2d_k + d_v))
= O((n + d_model) ⋅ (2d_k + d_v))

Effect of Scaling Key Parameters
*   **Sequence Length (n)**: Linearly impacts time complexity (O(n)) and memory complexity for projected matrices (O(n)). Longer sequences require proportionally more computation and memory for Q, K, V.
*   **Model Dimension (d_model)**: Quadratically impacts time complexity (O(d_model^2)) and linearly impacts memory complexity for weights (O(d_model)). A larger d_model significantly increases computational cost.
*   **Projection Dimensions (d_k, d_v)**: Linearly impacts both time complexity (O(d_k), O(d_v)) and memory complexity (O(d_k), O(d_v)). Increasing d_k or d_v directly increases computation and memory. When d_k = d_v = d_model / h, the impact is proportional to d_model/h.

Trade-offs
*   **Computational Cost vs. Expressivity**: Larger d_k and d_v allow for richer, more nuanced representations, potentially increasing the model's capacity to learn complex relationships. However, this comes at a direct cost in terms of FLOPs and memory.
*   **Parameter Count vs. Generalization**: A larger number of parameters (due to larger d_model, d_k, d_v) increases the model's capacity but also the risk of overfitting, especially with limited training data.
*   **Memory vs. Batch Size**: The memory required for Q, K, V matrices scales with n. This can limit the maximum batch size that can be processed, especially for very long sequences, as the total memory for all sequences in a batch becomes prohibitive.

Expressivity and Theoretical Implications

QKV projections are not merely dimensionality transformations; they are critical for the expressivity and information flow within attention mechanisms.

Rank or Capacity Considerations
The rank of the projection matrices W_Q, W_K, W_V directly influences the effective dimensionality and information capacity of the Q, K, and V spaces.
*   The maximum rank of W_Q, W_K, W_V is min(d_model, d_k) or min(d_model, d_v).
*   If d_k < d_model, the projection acts as a dimensionality reduction. This forces the model to learn a compressed, more salient representation of the input for querying and key matching. The effective rank of Q and K cannot exceed d_k, regardless of the rank of X. This can be a bottleneck, limiting the complexity of relationships that can be captured.
*   If d_k = d_model, the projection is a full-rank linear transformation (assuming W_Q, W_K are full rank). This preserves the full information content of the input embeddings in the query/key space, allowing for potentially more complex matching patterns.
*   Similarly, for V, if d_v < d_model, the value information is compressed. If d_v = d_model, the value information is preserved.
The choice of d_k and d_v thus directly controls the information capacity and potential for information loss or compression at this stage. A lower rank projection can act as a regularizer, preventing the model from overfitting to noise in high-dimensional input embeddings.

Information Flow Analysis
QKV projections fundamentally alter the information flow by creating specialized pathways for different aspects of attention:
1.  **Decoupling**: By using separate weight matrices, the model learns distinct transformations for "what to look for" (Q), "what is available" (K), and "what content to retrieve" (V). This decoupling allows for a more flexible and powerful attention mechanism than if raw input embeddings were used directly. For example, the features relevant for querying might be different from those relevant for providing content.
2.  **Feature Specialization**: Each projection matrix (W_Q, W_K, W_V) learns to extract specific features from the input embeddings that are most relevant to its designated role. W_Q learns features that make queries effective at finding relevant keys. W_K learns features that make keys effective at being found by relevant queries. W_V learns features that best represent the information to be aggregated.
3.  **Contextualization**: While the projections themselves are linear and local (each token's projection depends only on its own embedding), they set the stage for global contextualization. The subsequent attention mechanism uses these projected vectors to compute interactions across the entire sequence, allowing each token to gather information from all other tokens based on their learned QKV representations.

Comparison with Alternatives
The primary alternative to QKV projections would be to use the raw input embeddings directly as Queries, Keys, and Values, or to use a single shared projection matrix for all three.
*   **Using Raw Embeddings**: If Q=K=V=X, then the attention mechanism would directly compute similarities between raw token embeddings. This would severely limit expressivity because the same vector space would need to simultaneously encode "what I'm looking for," "what I have," and "my content." It would be difficult to learn distinct features for these different roles, leading to less nuanced attention.
*   **Shared Projection Matrix**: Using a single weight matrix W for Q, K, and V (e.g., Q=XW, K=XW, V=XW) would also restrict expressivity. While it introduces learnable parameters, it forces the query, key, and value spaces to be identical, preventing the specialization that separate W_Q, W_K, W_V