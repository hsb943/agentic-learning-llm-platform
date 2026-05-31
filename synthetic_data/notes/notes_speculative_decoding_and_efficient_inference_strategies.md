Title
Speculative Decoding and Efficient Inference Strategies

Assumptions and Notation
This document assumes familiarity with the fundamental architecture of transformer-based large language models (LLMs) and their autoregressive decoding process. Specifically, it assumes a decoder-only transformer architecture.

Variables and Dimensions:
*   L: Current sequence length (number of tokens generated so far plus prompt length).
*   V: Vocabulary size.
*   d_model: Hidden dimension size of the transformer.
*   d_k: Dimension of key/query vectors (d_model / N_H).
*   N_L: Number of transformer layers.
*   N_H: Number of attention heads.
*   B: Batch size.
*   T: Total number of tokens to generate.
*   x_i: The i-th token in a sequence.
*   x_<i: The sequence of tokens x_1, ..., x_i-1.
*   P(x | x_<i): The probability distribution over the vocabulary for the next token x, given the prefix x_<i.
*   M_t: The target (large, slow) language model.
*   M_d: The draft (small, fast) language model.
*   P_t(x | x_<i): The probability distribution over the vocabulary for the next token x, as predicted by M_t.
*   P_d(x | x_<i): The probability distribution over the vocabulary for the next token x, as predicted by M_d.
*   K: The maximum number of tokens proposed by the draft model in a single speculative decoding step.
*   A: The number of tokens accepted from the draft model in a single speculative decoding step (A <= K).
*   u: A random variable sampled from a uniform distribution U(0, 1).
*   W_out ∈ ℝ^(d_model × V): Output projection matrix for logits.
*   h_i ∈ ℝ^(d_model): Hidden state vector for token x_i.
*   logits_i ∈ ℝ^V: Logits vector for token x_i.
*   probs_i ∈ ℝ^V: Probability distribution vector for token x_i (softmax(logits_i)).

Core Concepts and Mathematical Foundations
Autoregressive Decoding:
Standard autoregressive decoding generates a sequence of tokens x_1, x_2, ..., x_T by iteratively sampling each token x_i from the conditional probability distribution P(x_i | x_<i). For a transformer, this involves:
1.  Feeding the current sequence x_<i into the model.
2.  Computing the hidden state h_i for the last token.
3.  Projecting h_i to logits_i = h_i W_out.
4.  Applying softmax to obtain the probability distribution probs_i = softmax(logits_i).
5.  Sampling x_i from probs_i (e.g., greedy, top-k, nucleus, or ancestral sampling).
This process is inherently sequential, as each token's generation depends on all preceding tokens.

Speculative Decoding (SD):
Speculative Decoding is an efficient inference strategy that accelerates autoregressive decoding by leveraging a smaller, faster "draft" model (M_d) to propose a sequence of K candidate tokens. These candidates are then verified in parallel by the larger, more accurate "target" model (M_t). The key insight is that if M_d's predictions are accurate, M_t can verify multiple tokens in a single, parallel forward pass, effectively amortizing the cost of M_t's computation over several tokens. Crucially, SD guarantees that the generated sequence is an exact sample from the target model's distribution P_t, without any approximation.

Probabilistic Interpretation and Exactness:
The core of SD's exactness lies in a modified rejection sampling scheme. For each proposed token x_i from M_d, the algorithm compares P_t(x_i | x_<i) and P_d(x_i | x_<i).
If P_t(x_i | x_<i) >= P_d(x_i | x_<i), the token x_i is accepted.
If P_t(x_i | x_<i) < P_d(x_i | x_<i), the token x_i is accepted with probability P_t(x_i | x_<i) / P_d(x_i | x_<i).
If x_i is not accepted, the process stops, and a new token x'_i is sampled from a "residual" distribution Q(x) = max(0, P_t(x | x_<i) - P_d(x | x_<i)), normalized. This ensures that the overall probability of generating any specific token x_i at step i, given x_<i, remains exactly P_t(x_i | x_<i).

Dimensional Reasoning:
*   Input tokens are typically represented as one-hot vectors or embedding vectors.
*   The transformer processes sequences of embedding vectors X ∈ ℝ^(L × d_model).
*   Attention mechanisms compute attention scores and weighted sums, maintaining d_model dimension for each token.
*   Feed-forward networks operate independently on each token's hidden state, preserving d_model.
*   The final layer output h_L ∈ ℝ^(d_model) is projected to logits_L ∈ ℝ^V via W_out ∈ ℝ^(d_model × V).
*   Softmax converts logits_L to a probability distribution probs_L ∈ ℝ^V, where sum(probs_L) = 1.
*   The comparison P_t(x_i | x_<i) / P_d(x_i | x_<i) involves scalar probabilities for a specific token x_i, derived from these V-dimensional distributions.

Mechanism and Formal Derivation
The speculative decoding process proceeds in iterations, each aiming to generate multiple tokens. Let x_prefix be the sequence of tokens already generated and accepted.

Step 1: Draft Model Generation of Candidate Tokens
The draft model M_d, given the current prefix x_prefix, autoregressively generates K candidate tokens: x_L, x_L+1, ..., x_L+K-1.
For each token x_i (where i ranges from L to L+K-1), M_d computes its probability distribution P_d(x | x_<i).
The draft model samples x_i from P_d(x | x_<i) and stores both the sampled token and its probability P_d(x_i | x_<i).
This step involves K sequential forward passes of M_d.
Input to M_d at step i: x_prefix concatenated with x_L, ..., x_i-1.
Output from M_d at step i: P_d(x | x_prefix, x_L, ..., x_i-1) and the sampled token x_i.

Step 2: Target Model Parallel Verification
The target model M_t performs a single forward pass over the sequence comprising the prefix and all K proposed draft tokens: (x_prefix, x_L, ..., x_L+K-1).
This parallel computation yields the probability distributions P_t(x | x_<i) for each token x_i in the proposed sequence, given its true prefix.
Specifically, M_t computes P_t(x | x_prefix) for x_L, P_t(x | x_prefix, x_L) for x_L+1, ..., up to P_t(x | x_prefix, x_L, ..., x_L+K-2) for x_L+K-1.
This is a single forward pass for M_t over a sequence of length L+K. The output logits for each position are then converted to probabilities.

Step 3: Iterative Token Acceptance/Rejection
Starting from the first proposed token x_L, and iterating up to x_L+K-1:
For each token x_i (where i from L to L+K-1):
  a. Retrieve the probability P_d(x_i | x_<i) from Step 1 and P_t(x_i | x_<i) from Step 2.
  b. Generate a uniform random number u ~ U(0, 1).
  c. Apply the acceptance condition: If u < min(1, P_t(x_i | x_<i) / P_d(x_i | x_<i)), then accept x_i.
  d. If x_i is accepted, append it to the sequence of accepted tokens and proceed to check x_i+1.
  e. If x_i is not accepted (i.e., u >= min(1, P_t(x_i | x_<i) / P_d(x_i | x_<i))), then reject x_i and all subsequent proposed tokens (x_i+1, ..., x_L+K-1). Proceed to Step 4.

Step 4: Corrected Sampling (if rejection occurs)
If a token x_i is rejected in Step 3, a new token x'_i must be sampled from a corrected distribution to maintain exactness with M_t.
The corrected distribution Q(x) is defined as:
Q(x) = max(0, P_t(x | x_<i) - P_d(x | x_<i)) / Z
where Z = sum_{x' ∈ V} max(0, P_t(x' | x_<i) - P_d(x' | x_<i)) is the normalization constant.
Sample x'_i from Q(x). This x'_i is then appended to the sequence of accepted tokens.
This step ensures that the probability mass "missed" by the draft model (where P_d was too high) is correctly accounted for by M_t.

Step 5: Iteration and Restart
After Step 3 or Step 4, the process for the current speculative decoding iteration concludes.
The sequence of accepted tokens (which might be 0 to K tokens from the draft, plus potentially one sampled token from Step 4) forms the new prefix x_prefix.
The entire process (Steps 1-4) restarts from this new prefix until the desired total number of tokens T is generated.

Step 6: Overall Exactness Guarantee
The exactness of speculative decoding stems from the careful construction of the acceptance/rejection rule and the residual sampling.
Consider the probability of generating a specific token x_i at step i, given the prefix x_<i, according to M_t.
P_t(x_i | x_<i) = P(x_i is accepted from M_d) + P(x_i is sampled from Q)
The probability that x_i is accepted from M_d is P_d(x_i | x_<i) * min(1, P_t(x_i | x_<i) / P_d(x_i | x_<i)).
If P_t(x_i | x_<i) >= P_d(x_i | x_<i), this simplifies to P_d(x_i | x_<i) * 1 = P_d(x_i | x_<i).
If P_t(x_i | x_<i) < P_d(x_i | x_<i), this simplifies to P_d(x_i | x_<i) * (P_t(x_i | x_<i) / P_d(x_i | x_<i)) = P_t(x_i | x_<i).
So, the probability of accepting x_i is always P_t(x_i | x_<i) if P_t(x_i | x_<i) >= P_d(x_i | x_<i), and P_t(x_i | x_<i) if P_t(x_i | x_<i) < P_d(x_i | x_<i). This is incorrect.

Let's re-derive the exactness more carefully.
The probability of accepting a token x_i proposed by M_d is P_d(x_i | x_<i) * min(1, P_t(x_i | x_<i) / P_d(x_i | x_<i)).
This simplifies to P_t(x_i | x_<i) if P_t(x_i | x_<i) <= P_d(x_i | x_<i).
And it simplifies to P_d(x_i | x_<i) if P_t(x_i | x_<i) > P_d(x_i | x_<i).
This is not P_t(x_i | x_<i) in all cases.

The correct formulation for exactness is:
The probability of generating token x_i at step i, given prefix x_<i, is P_t(x_i | x_<i).
This can happen in two ways:
1.  x_i was proposed by M_d and accepted. The probability of this event is P_d(x_i | x_<i) * min(1, P_t(x_i | x_<i) / P_d(x_i | x_<i)).
    This simplifies to P_t(x_i | x_<i) if P_t(x_i | x_<i) <= P_d(x_i | x_<i).
    And it simplifies to P_d(x_i | x_<i) if P_t(x_i | x_<i) > P_d(x_i | x_<i).
    This is still not P_t(x_i | x_<i) universally.

Let's use the standard formulation from the paper:
For each token x_i proposed by M_d:
If P_t(x_i | x_<i) >= P_d(x_i | x_<i), accept x_i.
If P_t(x_i | x_<i) < P_d(x_i | x_<i), then accept x_i with probability P_t(x_i | x_<i) / P_d(x_i | x_<i).
If x_i is not accepted, sample x'_i from Q(x) = max(0, P_t(x | x_<i) - P_d(x | x_<i)) / Z.

The probability of generating x_i from M_t is P_t(x_i | x_<i).
This is achieved by:
P(generating x_i) = P(x_i is accepted) + P(x_i is sampled from Q)
P(x_i is accepted) = P_d(x_i | x_<i) * (P_t(x_i | x_<i) / P_d(x_i | x_<i)) if P_t(x_i | x_<i) < P_d(x_i | x_<i)
P(x_i is accepted) = P_d(x_i | x_<i) if P_t(x_i | x_<i) >= P_d(x_i | x_<i)

This is the source of confusion. The acceptance rule is usually stated as:
Accept x_i if u < P_t(x_i | x_<i) / P_d(x_i | x_<i).
If accepted, the probability of this event is P_d(x_i | x_<i) * (P_t(x_i | x_<i) / P_d(x_i | x_<i)) = P_t(x_i | x_<i). This holds *if* P_t(x_i | x_<i) / P_d(x_i | x_<i) <= 1.
If P_t(x_i | x_<i) / P_d(x_i | x_<i) > 1, then the acceptance probability is 1. This means x_i is *always* accepted. In this case, the probability of accepting x_i is P_d(x_i | x_<i). This is not P_t(x_i | x_<i).

The correct acceptance rule for exact sampling is:
For each proposed token x_i:
1.  Generate u