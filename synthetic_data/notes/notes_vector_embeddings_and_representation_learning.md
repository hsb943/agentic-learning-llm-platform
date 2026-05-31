Vector Embeddings and Representation Learning

Assumptions and Notation
Let X denote the input space of entities, which can be discrete (e.g., words, users, items) or high-dimensional continuous data.
Let V be a finite vocabulary or set of discrete entities, with cardinality |V|.
Let x ∈ X be an individual entity.
Let d ∈ ℕ be the dimensionality of the embedding space, where d << |V| or d << D_input.
Let D_input ∈ ℕ be the dimensionality of the raw input feature space for continuous data.
Let v ∈ ℝ^d be a vector embedding.
Let W_in ∈ ℝ^(|V| × d) be an input embedding matrix, where each row W_in[i, :] corresponds to the embedding for the i-th entity in V.
Let W_out ∈ ℝ^(d × |V|) be an output embedding matrix, where each column W_out[:, j] corresponds to the embedding for the j-th entity in V when used as a context or output.
Let σ(z) = 1 / (1 + exp(-z)) denote the sigmoid activation function.
Let N be the total number of training samples or instances.
Let K be the number of negative samples used in negative sampling.
Let L denote a loss function.
Let ⟨a, b⟩ = a^T b denote the inner product between vectors a and b.
Matrix shapes are denoted as M ∈ ℝ^(rows × columns).
Vector shapes are denoted as v ∈ ℝ^dimension.

Core Concepts and Mathematical Foundations
Vector embeddings are dense, real-valued vector representations of entities (e.g., words, documents, images, users, items) in a continuous vector space ℝ^d. The fundamental principle is that entities with similar semantic or functional properties are mapped to vectors that are close to each other in this embedding space.

Formal Definition: An embedding function E is a mapping E: X → ℝ^d, where X is the set of entities and d is the embedding dimension. For discrete entities, this often involves a lookup table, where each entity x_i ∈ V is associated with a unique vector v_i ∈ ℝ^d. For continuous or complex entities, E might be a neural network or another parameterized function.

Representation Learning: This is the process of automatically discovering the embedding function E or the embedding vectors themselves from data, rather than manually engineering features. The learning process is typically driven by an objective function that quantifies how well the embeddings capture desired relationships or perform a downstream task. The objective function implicitly defines what "similarity" means in the embedding space.

Geometric Interpretation:
The proximity of vectors in ℝ^d is used to quantify the similarity between the corresponding entities. Common distance metrics include:
1.  Cosine Similarity: For two embedding vectors v_i, v_j ∈ ℝ^d, their cosine similarity is defined as cos(θ) = (v_i^T v_j) / (||v_i||_2 ||v_j||_2). This measures the cosine of the angle between the vectors, ranging from -1 (opposite) to 1 (identical direction).
2.  Euclidean Distance: For two embedding vectors v_i, v_j ∈ ℝ^d, their Euclidean distance is ||v_i - v_j||_2 = sqrt(Σ_k (v_i[k] - v_j[k])^2). Smaller distances imply greater similarity.
The geometric arrangement of embeddings can also capture analogies, where vector arithmetic operations correspond to relational transformations (e.g., v_king - v_man + v_woman ≈ v_queen).

Probabilistic Interpretation:
Embeddings can be viewed as parameters of a probabilistic model that predicts relationships between entities. For instance, in natural language processing, embeddings might be learned such that the probability of observing a context word given a target word (or vice-versa) is maximized. The embedding vectors parameterize the conditional probability distributions P(context | target) or P(target | context). The learning objective often involves maximizing the likelihood of observed data under this probabilistic model.

Dimensional Reasoning:
The choice of embedding dimension d is critical.
Input Space: For discrete entities, the "input space" can be considered a one-hot encoding space of dimension |V|. This space is sparse and high-dimensional. For continuous data, D_input can be very large.
Embedding Space: The embedding space ℝ^d is dense and typically much lower-dimensional (d << |V| or d << D_input). This dimensionality reduction forces the model to learn compact, salient features.
Impact of d:
*   Small d: May lead to underfitting, where the embeddings lack the capacity to capture all relevant semantic distinctions. Entities that are distinct might be mapped too closely.
*   Large d: May lead to overfitting, where the embeddings capture noise or spurious correlations. It also increases memory requirements (O(|V| * d)) and computational cost (proportional to d).
The optimal d balances expressivity with computational efficiency and generalization.

Mechanism and Formal Derivation
We will derive the Skip-gram with Negative Sampling (SGNS) mechanism, a foundational method for learning word embeddings. The goal is to learn two sets of embeddings: input embeddings (W_in) and output/context embeddings (W_out).

Step 1: Input Representation and Embedding Lookup
Given a vocabulary V of size |V|, each word w_i ∈ V is initially represented by a one-hot vector x_i ∈ ℝ^|V|. This vector has a 1 at the index corresponding to w_i and 0s elsewhere.
The input embedding for word w_i, denoted v_i, is obtained by multiplying its one-hot vector by the input embedding matrix W_in ∈ ℝ^(|V| × d).
v_i = x_i^T W_in
Dimensionally: (1 × |V|) ⋅ (|V| × d) → (1 × d). This operation effectively selects the i-th row of W_in as the embedding v_i.

Step 2: Context Word Scoring
For a given input word w_i with embedding v_i ∈ ℝ^d, the model aims to predict its surrounding context words. Let w_j be a potential context word. Its "context" embedding is given by the j-th column of the output embedding matrix W_out ∈ ℝ^(d × |V|), denoted u_j ∈ ℝ^d.
The "score" for the pair (w_i, w_j) is computed as the inner product of their respective embeddings:
score(w_i, w_j) = ⟨v_i, u_j⟩ = v_i^T u_j
Dimensionally: (1 × d) ⋅ (d × 1) → (1 × 1). This score indicates the compatibility between the input word and the context word.

Step 3: Full Softmax Probability (Conceptual, for understanding)
In the original Skip-gram model, the probability of observing a context word w_j given an input word w_i is calculated using a softmax function over all possible words in the vocabulary:
P(w_j | w_i) = exp(⟨v_i, u_j⟩) / Σ_{k=1}^{|V|} exp(⟨v_i, u_k⟩)
The objective function for a training pair (w_i, w_j) would be to maximize this probability, typically by minimizing the negative log-likelihood:
L_softmax = -log P(w_j | w_i)
Dimensionally: The numerator is a scalar. The denominator involves |V| scalar computations and a sum. This sum over the entire vocabulary is computationally expensive, especially for large |V|.

Step 4: Negative Sampling Objective
To address the computational bottleneck of the full softmax, Negative Sampling (NS) is introduced. Instead of predicting the correct context word against all other words, NS transforms the problem into a binary classification task: distinguishing the true context word from a few randomly sampled "negative" words.
For each observed positive pair (w_i, w_j) (input word w_i, true context word w_j), we sample K negative context words {w_neg_1, ..., w_neg_K} from the vocabulary, typically according to their unigram frequency raised to a power (e.g., 0.75).
The objective function for a single positive pair (w_i, w_j) and its K negative samples is to maximize the probability of correctly classifying the positive pair and incorrectly classifying the negative pairs. This is achieved by maximizing:
L_NS_pair = log σ(⟨v_i, u_j⟩) + Σ_{k=1}^K log σ(-⟨v_i, u_neg_k⟩)
Here, u_neg_k is the context embedding for the k-th negative sample w_neg_k. The term -⟨v_i, u_neg_k⟩ encourages the score for negative pairs to be low, pushing σ(-score) towards 1.
Dimensionally: Each inner product is (1 × d) ⋅ (d × 1) → (1 × 1). The sigmoid function operates on a scalar. The sum involves K+1 terms.

Step 5: Total Loss Function
For a given training corpus, consisting of N observed (input, context) pairs, the total loss function L_NS is the sum of the individual L_NS_pair terms:
L_NS = - Σ_{n=1}^N [log σ(⟨v_n_in, u_n_pos⟩) + Σ_{k=1}^K log σ(-⟨v_n_in, u_n_neg_k⟩)]
The goal of training is to find the embedding matrices W_in and W_out that minimize this total loss function. This is typically done using stochastic gradient descent (SGD) or its variants.

Step 6: Gradient Calculation and Parameter Update
To minimize L_NS, we compute gradients with respect to the embedding vectors v_i and u_j (and u_neg_k).
Let E_pos = ⟨v_i, u_j⟩ and E_neg_k = ⟨v_i, u_neg_k⟩.
The derivative of L_NS_pair with respect to E_pos is (σ(E_pos) - 1).
The derivative of L_NS_pair with respect to E_neg_k is (σ(E_neg_k) - 0) = σ(E_neg_k).
Using the chain rule:
∂L_NS_pair / ∂v_i = (σ(E_pos) - 1) u_j + Σ_{k=1}^K (σ(E_neg_k) - 0) (-u_neg_k)
∂L_NS_pair / ∂u_j = (σ(E_pos) - 1) v_i
∂L_NS_pair / ∂u_neg_k = (σ(E_neg_k) - 0) (-v_i)
These gradients are then used to update the corresponding rows/columns in W_in and W_out using an optimizer (e.g., SGD with learning rate η):
v_i ← v_i - η * ∂L_NS_pair / ∂v_i
u_j ← u_j - η * ∂L_NS_pair / ∂u_j
u_neg_k ← u_neg_k - η * ∂L_NS_pair / ∂u_neg_k
Dimensionally: Gradients are vectors of dimension d. Updates are vector additions/subtractions.

Computational and Complexity Analysis
Time Complexity:
*   Embedding Lookup: For a given word, retrieving its embedding v_i from W_in is O(d) if it involves a matrix-vector product with a one-hot vector, or effectively O(1) if implemented as an array lookup.
*   Score Calculation: Computing ⟨v_i, u_j⟩ is O(d) for each pair.
*   Softmax (Full): Calculating P(w_j | w_i) involves computing |V| scores and an exponentiation and sum. This is O(|V| * d) per training sample. Gradient computation is also O(|V| * d).
*   Negative Sampling: For each positive pair, K negative samples are drawn. This involves K+1 score calculations (1 positive, K negative) and K+1 sigmoid evaluations. This is O((K+1) * d) = O(K * d) per training sample. Gradient computation is also O(K * d).
*   Total Training Time: If N is the total number of (input, context) pairs in the corpus, the total training time for SGNS is O(N * K * d). This is significantly faster than O(N * |V| * d) for full softmax when K << |V|.

Memory Complexity:
*   Embedding Matrices: The primary memory consumption comes from storing the embedding matrices W_in and W_out.
    *   W_in ∈ ℝ^(|V| × d) requires O(|V| * d) memory.
    *   W_out ∈ ℝ^(d × |V|) requires O(|V| * d) memory.
    *   Total memory for embeddings: O(|V| * d).
*   Intermediate Computations: During training, gradients and activations require temporary storage, typically proportional to d and the batch size.

Effect of Scaling Key Parameters:
*   |V| (Vocabulary Size): Directly impacts memory (O(|V| * d)) and, for full softmax, training time (O(|V| * d)). With negative sampling, its impact on training time is reduced to the sampling process, but memory remains linear.
*   d (Embedding Dimension): Directly impacts memory (O(|V| * d)) and training time (O(N * K * d)). Higher d allows for more expressive embeddings but increases computational cost and risk of overfitting.
*   N (Corpus Size): Directly impacts total training time (O(N * K * d)). Larger N generally leads to better quality embeddings, assuming sufficient computational resources.
*   K (Number of Negative Samples): Directly impacts training time (O(N * K * d)). A larger K provides more robust gradient signals but increases computation. Typical values are 5-20 for large datasets.

Trade-offs:
*   Expressivity vs. Computation: Increasing d enhances the capacity of embeddings to capture fine-grained semantic distinctions but increases memory and time complexity.
*   Training Time vs. Quality: Larger N and appropriate K values improve embedding quality but extend training duration.
*   Full Softmax vs. Negative Sampling: Full softmax provides a theoretically cleaner objective but is computationally prohibitive for large vocabularies. Negative sampling offers a practical and effective approximation with significantly reduced computational cost.

Expressivity and Theoretical Implications
Rank or Capacity Considerations:
The embedding matrix W_in ∈ ℝ^(|V| × d) has a maximum rank of min(|V|, d). If d < |V|, the embedding process inherently performs dimensionality reduction. This forces the model to learn a compressed representation where distinct entities share latent features. The capacity of the embedding space to distinguish between |V| entities is limited by d. If d is too small, distinct entities might be forced into similar regions, leading to a loss of information.

Information Flow Analysis:
Representation learning, particularly methods like Skip-gram, implicitly factorizes a word-context co-occurrence matrix. The embeddings v_i and u_j can be seen as latent factors that explain the observed co-occurrence patterns. The inner product ⟨v_i, u_j⟩ models the strength of association between w_i and w_j. The information about semantic and syntactic relationships flows from the co-occurrence statistics of the corpus into the learned embedding vectors. The objective function (e.g., maximizing log-likelihood of observed pairs) guides this information encoding.

Comparison with Alternatives:
*   One-hot encoding: High-dimensional, sparse, and lacks any notion of similarity between entities (all one-hot vectors are orthogonal). Embeddings overcome this by providing dense, low-dimensional, and semantically rich representations.
*   Count-based methods (e.g., TF-IDF, PPMI): These methods directly compute statistics from co-occurrence matrices. While they capture some semantic information, they often result in high-dimensional, sparse vectors and struggle with polysemy or capturing complex analogies. Embeddings, especially those learned via neural networks, can capture more nuanced relationships and are often more robust to noise. SGNS, for instance, has been shown to implicitly factorize a shifted Positive Pointwise Mutual Information (PPMI) matrix.

Failure Modes and Edge Cases
Numerical Instability:
*   Softmax Overflow/Underflow: In the full softmax calculation, exp(score) can become extremely large or small, leading to overflow or underflow. The log-sum-exp trick is used to compute log(Σ exp(x_i)) stably. Negative sampling avoids this by using sigmoid, which is numerically stable.
*   Vanishing/Exploding Gradients: While less common in shallow embedding models like Skip-gram compared to deep neural networks, very large or very small learning rates can still cause issues. If gradients become too small, learning stalls; if too large, updates become erratic.
*   NaN Values: Can arise from division by zero, log of zero/negative numbers, or extreme values in intermediate computations.

Degenerate Configurations:
*   Embedding Collapse: All embeddings might converge to a single point or a very small region in the embedding space. This can happen if the learning rate is too high, the objective function is trivial, or regularization is insufficient. If all embeddings are identical, they lose all discriminative power.
*   Anisotropy: Embeddings might occupy only a narrow cone within the embedding space, meaning most vectors point in similar directions. This limits the effective capacity of the embedding space and can degrade performance on tasks relying on angular differences (e.g., cosine similarity). This often occurs because the objective function primarily focuses on positive examples, pushing them close, but doesn't sufficiently spread out negative examples.
*   Out-of-Vocabulary (OOV) Items: Entities not present in the training vocabulary |V| will not have pre-trained embeddings. Common solutions include:
    *   Assigning a special "UNK" (unknown) token embedding.
    *   Using character-level or subword-level embeddings (e.g., FastText) to compose embeddings for OOV words.
    *   Learning OOV embeddings on the fly during inference or fine-tuning.

Scaling Limitations:
*   Large Vocabulary: For extremely large vocabularies (e.g., millions of words, items), even O(|V| * d) memory for embedding matrices can become prohibitive. This necessitates techniques like hashing embeddings or dynamic embedding creation.
*   High Dimensionality (d): While beneficial for expressivity, very high d can lead to overfitting, increased memory, and slower training. It also makes visualization and interpretation more challenging.
*   Limited Data: If the corpus size N is small, embeddings may not generalize well, leading to poor performance on unseen data. The model might overfit to the limited co-occurrence patterns.

Diagnostic Reasoning:
*   Monitor Loss Curve: A flat loss curve might indicate vanishing gradients or a too-small learning rate. An oscillating or exploding loss curve suggests an unstable training process, possibly due to a too-large learning rate.
*   Check Embedding Norms: If all embedding norms are very small or very large, it might