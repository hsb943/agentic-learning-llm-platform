Multi-Head Attention Architecture

Assumptions and Notation
The following variables and notations are used consistently throughout this document:

X ∈ ℝ^(L × d_model): Input sequence, where L is the sequence length and d_model is the dimensionality of the input embedding for each token.
Q ∈ ℝ^(L × d_model): Query matrix, derived from X.
K ∈ ℝ^(L × d_model): Key matrix, derived from X.
V ∈ ℝ^(L × d_model): Value matrix, derived from X.
h: Number of attention heads.
d_model: Dimensionality of the input and output representations of the Multi-Head Attention layer.
d_k: Dimensionality of the key and query vectors for a single attention head. Typically, d_k = d_model / h.
d_v: Dimensionality of the value vectors for a single attention head. Typically, d_v = d_model / h.
W_Q_i ∈ ℝ^(d_model × d_k): Weight matrix for projecting queries for the i-th head.
W_K_i ∈ ℝ^(d_model × d_k): Weight matrix for projecting keys for the i-th head.
W_V_i ∈ ℝ^(d_model × d_v): Weight matrix for projecting values for the i-th head.
W_O ∈ ℝ^((h ⋅ d_v) × d_model): Output linear projection matrix.
softmax(Z): The softmax function applied row-wise to a matrix Z, where softmax(z_j) = exp(z_j) / Σ_k exp(z_k).
[A; B]: Matrix concatenation along the column dimension.

Core Concepts and Mathematical Foundations
Multi-Head Attention is an extension of the Scaled Dot-Product Attention mechanism, designed to enhance the model's ability to focus on different parts of the input sequence from multiple "representation subspaces."

Formal Definitions
Scaled Dot-Product Attention:
Given query matrix Q ∈ ℝ^(L × d_k), key matrix K ∈ ℝ^(L × d_k), and value matrix V ∈ ℝ^(L × d_v), the output of Scaled Dot-Product Attention is defined as:
Attention(Q, K, V) = softmax(QKᵀ / √d_k)V

Geometric or Probabilistic Interpretation
1.  **Query-Key Similarity**: The product QKᵀ computes a similarity score between each query vector (row of Q) and each key vector (row of K). A higher dot product indicates greater similarity or relevance. The result is an attention score matrix S ∈ ℝ^(L × L), where S_ij represents the relevance of the j-th key to the i-th query.
2.  **Scaling**: Division by √d_k is a crucial scaling factor. Without it, for large d_k, the dot products QKᵀ can become very large in magnitude, pushing the softmax function into regions where its gradients are extremely small (saturating the softmax). This scaling helps to stabilize the training process by keeping the variance of the dot products consistent.
3.  **Softmax Normalization**: The softmax function converts the raw attention scores into a probability distribution over the keys for each query. Each row of the resulting matrix A ∈ ℝ^(L × L) sums to 1, indicating how much attention each query should pay to each key. A_ij represents the attention weight from the i-th query to the j-th key.
4.  **Weighted Sum of Values**: The attention weights A are then multiplied by the value matrix V. This operation computes a weighted sum of the value vectors, where the weights are determined by the attention scores. The output vector for each query is a linear combination of all value vectors, with coefficients reflecting their relevance. This allows the model to aggregate information from relevant parts of the input sequence.

Dimensional Reasoning
*   Q ∈ ℝ^(L × d_k), K ∈ ℝ^(L × d_k), V ∈ ℝ^(L × d_v)
*   Kᵀ ∈ ℝ^(d_k × L)
*   QKᵀ ∈ ℝ^(L × d_k) ⋅ ℝ^(d_k × L) = ℝ^(L × L)
*   QKᵀ / √d_k ∈ ℝ^(L × L) (scalar division does not change dimensions)
*   softmax(QKᵀ / √d_k) ∈ ℝ^(L × L) (softmax operates row-wise, preserving dimensions)
*   softmax(QKᵀ / √d_k)V ∈ ℝ^(L × L) ⋅ ℝ^(L × d_v) = ℝ^(L × d_v)
The output of a single attention head for a sequence of length L is a matrix of shape L × d_v.

Mechanism and Formal Derivation
Multi-Head Attention combines h independent Scaled Dot-Product Attention mechanisms, each operating on different linear projections of the input. The outputs of these heads are then concatenated and linearly transformed to produce the final output.

Step 1: Linear Projections for Each Head
For each of the h attention heads (indexed i from 1 to h), the input sequence X is linearly projected into distinct query, key, and value spaces. These projections are achieved using learned weight matrices W_Q_i, W_K_i, and W_V_i.
Given an input X ∈ ℝ^(L × d_model):
Q_i = X W_Q_i ∈ ℝ^(L × d_model) ⋅ ℝ^(d_model × d_k) = ℝ^(L × d_k)
K_i = X W_K_i ∈ ℝ^(L × d_model) ⋅ ℝ^(d_model × d_k) = ℝ^(L × d_k)
V_i = X W_V_i ∈ ℝ^(L × d_model) ⋅ ℝ^(d_model × d_v) = ℝ^(L × d_v)
These projections allow each head to learn different aspects or "representation subspaces" of the input.

Step 2: Scaled Dot-Product Attention for Each Head
Each set of projected queries (Q_i), keys (K_i), and values (V_i) is then passed through the Scaled Dot-Product Attention function independently.
head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i K_iᵀ / √d_k)V_i
The output of each head, head_i, is a matrix of shape ℝ^(L × d_v).

Step 3: Parallel Execution of Multiple Heads
Steps 1 and 2 are performed in parallel for all h heads. This results in h distinct output matrices:
head_1, head_2, ..., head_h
Each head_i ∈ ℝ^(L × d_v).

Step 4: Concatenation of Head Outputs
The outputs from all h attention heads are concatenated along the feature dimension (the last dimension).
Concatenated_Output = [head_1; head_2; ...; head_h]
The shape of Concatenated_Output is ℝ^(L × (h ⋅ d_v)). This combines the diverse perspectives learned by each head.

Step 5: Final Linear Projection
The concatenated output is then linearly transformed using a final weight matrix W_O ∈ ℝ^((h ⋅ d_v) × d_model). This projection maps the combined output back to the desired d_model dimensionality, which is typically the same as the input dimensionality for compatibility with residual connections in transformer architectures.
Final_Output = Concatenated_Output W_O ∈ ℝ^(L × (h ⋅ d_v)) ⋅ ℝ^((h ⋅ d_v) × d_model) = ℝ^(L × d_model)

Step 6: Full Multi-Head Attention Equation
Combining all steps, the Multi-Head Attention function can be formally expressed as:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
and Attention(Q', K', V') = softmax(Q'K'ᵀ / √d_k)V'
Note: In self-attention, the initial Q, K, V are typically derived from the same input X, i.e., Q=K=V=X. For encoder-decoder attention, Q comes from the decoder, and K, V come from the encoder output.

Computational and Complexity Analysis
The computational and memory complexity of Multi-Head Attention is primarily driven by the matrix multiplications involved. We assume d_k = d_v = d_model / h for typical implementations.

Time Complexity
1.  **Query, Key, Value Projections (Step 1)**: For each head i, computing Q_i, K_i, V_i involves matrix multiplications of X (L × d_model) with W_Q_i, W_K_i, W_V_i (d_model × d_k or d_model × d_v).
    *   Cost per head: O(L ⋅ d_model ⋅ d_k + L ⋅ d_model ⋅ d_k + L ⋅ d_model ⋅ d_v) = O(L ⋅ d_model ⋅ (2d_k + d_v)).
    *   Total for h heads: h ⋅ O(L ⋅ d_model ⋅ (2d_k + d_v)).
    *   Substituting d_k = d_v = d_model / h: h ⋅ O(L ⋅ d_model ⋅ (3d_model / h)) = O(L ⋅ d_model²).
2.  **QKᵀ Product (Step 2)**: For each head i, computing Q_i K_iᵀ involves (L × d_k) ⋅ (d_k × L).
    *   Cost per head: O(L² ⋅ d_k).
    *   Total for h heads: h ⋅ O(L² ⋅ d_k).
    *   Substituting d_k = d_model / h: h ⋅ O(L² ⋅ d_model / h) = O(L² ⋅ d_model).
3.  **Softmax and Attention Weights (Step 2)**: The softmax operation on an L × L matrix.
    *   Cost per head: O(L²).
    *   Total for h heads: h ⋅ O(L²).
4.  **Weighted Sum of Values (Step 2)**: For each head i, computing Attention_i V_i involves (L × L) ⋅ (L × d_v).
    *   Cost per head: O(L² ⋅ d_v).
    *   Total for h heads: h ⋅ O(L² ⋅ d_v).
    *   Substituting d_v = d_model / h: h ⋅ O(L² ⋅ d_model / h) = O(L² ⋅ d_model).
5.  **Concatenation (Step 4)**: This is a memory rearrangement operation, typically O(L ⋅ h ⋅ d_v).
6.  **Final Linear Projection (Step 5)**: (L × (h ⋅ d_v)) ⋅ ((h ⋅ d_v) × d_model).
    *   Cost: O(L ⋅ (h ⋅ d_v) ⋅ d_model).
    *   Substituting h ⋅ d_v = d_model: O(L ⋅ d_model²).

Overall Time Complexity: The dominant terms are O(L² ⋅ d_model) and O(L ⋅ d_model²).
Therefore, the total time complexity is O(L² ⋅ d_model + L ⋅ d_model²).
The quadratic dependence on sequence length L is a significant characteristic.

Memory Complexity
1.  **Weight Matrices**: W_Q_i, W_K_i, W_V_i for h heads, and W_O.
    *   h ⋅ (d_model ⋅ d_k + d_model ⋅ d_k + d_model ⋅ d_v) + (h ⋅ d_v) ⋅ d_model.
    *   Substituting d_k = d_v = d_model / h: h ⋅ (3 ⋅ d_model ⋅ d_model / h) + d_model ⋅ d_model = O(d_model²).
2.  **Intermediate Activations**:
    *   Q_i, K_i, V_i: O(L ⋅ d_k), O(L ⋅ d_v). Total h ⋅ O(L ⋅ d_model / h) = O(L ⋅ d_model).
    *   Attention scores (QKᵀ): O(L²). Total h ⋅ O(L²) = O(h ⋅ L²).
    *   Concatenated output: O(L ⋅ h ⋅ d_v) = O(L ⋅ d_model).
Overall Memory Complexity: The dominant term is O(L² + d_model²). The O(h ⋅ L²) term for attention scores can be substantial for large L and h.

Effect of Scaling Key Parameters
*   **Sequence Length (L)**: Quadratic increase in time and memory complexity. This is the primary bottleneck for processing very long sequences.
*   **Model Dimension (d_model)**: Quadratic increase in time complexity (due to projections) and memory complexity (due to weights).
*   **Number of Heads (h)**: If d_k and d_v are kept constant, increasing h linearly increases complexity. However, typically d_k = d_v = d_model / h, so increasing h while keeping d_model constant means d_k and d_v decrease. In this common configuration, h does not directly increase the dominant complexity terms (O(L² ⋅ d_model) and O(L ⋅ d_model²)) but rather distributes the d_model dimension across heads. The memory for attention scores still scales with h ⋅ L².

Trade-offs
Multi-Head Attention offers increased representational capacity and the ability to capture diverse relationships compared to a single attention head. This comes at the cost of increased computational complexity (especially O(L² ⋅ d_model)) and memory footprint (O(L² + d_model²)), making it resource-intensive for very long sequences. The choice of h and d_k/d_v involves balancing expressivity with computational budget.

Expressivity and Theoretical Implications
Rank or Capacity Considerations
Each attention head learns a distinct set of linear projections (W_Q_i, W_K_i, W_V_i). This allows each head to project the input sequence into a different "representation subspace." Consequently, each head can focus on different types of relationships or patterns within the input. For example, one head might learn to attend to syntactic dependencies, another to semantic similarities, and yet another to positional relationships. By having multiple heads, the model's overall capacity to capture a rich variety of information is significantly enhanced compared to a single attention mechanism that would have to average or conflate these different types of relationships. The final linear projection W_O then serves to combine these diverse "perspectives" into a unified output representation, potentially learning to weigh the importance of each head's contribution.

Information Flow Analysis
Multi-Head Attention enables a highly flexible and dynamic information flow. For every element in the sequence, information can be aggregated from all other elements, weighted by their learned relevance. This creates a fully connected, directed graph where each node (sequence element) can receive information from every other node. The multi-head aspect means that this information aggregation can occur along several independent "channels" or "filters," each specializing in a different type of information. This contrasts with recurrent neural networks (RNNs) where information flows sequentially, or convolutional neural networks (CNNs) where information flow is localized by kernel size. The ability