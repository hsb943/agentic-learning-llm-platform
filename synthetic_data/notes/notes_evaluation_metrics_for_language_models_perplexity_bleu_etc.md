Title
Evaluation Metrics for Language Models: Perplexity and BLEU

Assumptions and Notation
This document defines evaluation metrics for language models. A language model (LM) is a probabilistic distribution P_M over sequences of tokens.
Let V be the vocabulary, a finite set of unique tokens.
Let S = (w_1, w_2, ..., w_N) be a sequence of N tokens, where w_i ∈ V for all i.
Let P_M(w_i | w_1...w_{i-1}) denote the conditional probability assigned by model M to token w_i given the preceding tokens w_1...w_{i-1}.
Let log denote the natural logarithm (base e).
Let exp denote the exponential function (e^x).
For BLEU, let C be a candidate sequence (model output) and R = {R_1, R_2, ..., R_K} be a set of K reference sequences (human translations).
Let n denote the n-gram order (e.g., n=1 for unigrams, n=2 for bigrams).
Let count(g, S) be the number of occurrences of n-gram g in sequence S.
Let count_clip(g, C, R) be the clipped count of n-gram g in candidate C with respect to reference set R.
Let L_C be the length of candidate sequence C.
Let L_R_j be the length of reference sequence R_j.
Let L_R_eff be the effective reference length.
Let w_n be the weight assigned to the n-gram precision P_n, typically w_n = 1/N_max for N_max being the maximum n-gram order.

Core Concepts and Mathematical Foundations

Perplexity
Perplexity (PPL) quantifies how well a probability distribution or language model predicts a sample. A lower perplexity indicates a better model. It is inversely related to the probability of the test set according to the language model.

Formal Definition:
Given a sequence S = (w_1, w_2, ..., w_N), the perplexity of the sequence under a language model P_M is defined as:
PPL(S) = exp( -1/N * sum_{i=1 to N} log P_M(w_i | w_1...w_{i-1}) )

Geometric or Probabilistic Interpretation:
Perplexity can be interpreted as the geometric mean of the inverse probabilities of the words in the sequence. Alternatively, it represents the weighted average number of choices a model has when predicting the next word. If a model has a perplexity of X, it means that, on average, the model is "confused" among X equally likely words at each step. A perfect model would assign a probability of 1 to each correct word, resulting in a perplexity of 1.

Dimensional Reasoning:
Probabilities P_M(w_i | w_1...w_{i-1}) are dimensionless values in the range [0, 1]. The logarithm of a dimensionless quantity is dimensionless. The sum of N dimensionless quantities is dimensionless. Dividing by N (a count, dimensionless) results in a dimensionless average. The exponential of a dimensionless quantity is dimensionless. Therefore, Perplexity is a dimensionless quantity.

BLEU (Bilingual Evaluation Understudy)
BLEU is a metric for evaluating the quality of text which has been machine-translated from one natural language to another. It measures the similarity between a machine-translated text and a set of high-quality human reference translations.

Formal Definition:
BLEU score is computed as a combination of modified n-gram precisions and a brevity penalty (BP).
BLEU = BP * exp( sum_{n=1 to N_max} w_n * log(P_n) )
where N_max is typically 4, and w_n = 1/N_max.

Modified n-gram Precision (P_n):
P_n = ( sum_{C in Corpus} sum_{g in C_n} count_clip(g, C, R) ) / ( sum_{C in Corpus} sum_{g in C_n} count(g, C) )
where C_n is the set of all n-grams in candidate C.
count_clip(g, C, R) = min( count(g, C), max_{R_j in R} count(g, R_j) )
This clipping ensures that a candidate cannot achieve a high score by simply repeating a few words that appear in the reference.

Brevity Penalty (BP):
BP = min( 1, exp(1 - L_R_eff / L_C) )
where L_C is the total length of all candidate sentences in the corpus.
L_R_eff is the effective reference length, calculated as the sum of the lengths of the reference sentences R_j that are closest to the length of the corresponding candidate sentence C. Specifically, for each candidate C, find the reference R_j whose length L_R_j is closest to L_C without being shorter than L_C. If all references are shorter, choose the shortest reference. Sum these chosen lengths over the corpus.

Geometric or Probabilistic Interpretation:
BLEU is not directly a probability. It is a weighted geometric mean of modified n-gram precisions, adjusted by a brevity penalty. The geometric mean is used because it penalizes low scores in any n-gram order more heavily than an arithmetic mean would. The brevity penalty ensures that short candidate sentences, which might achieve high precision by omitting content, are penalized.

Dimensional Reasoning:
Counts are dimensionless. Ratios of counts (P_n) are dimensionless. Logarithms of dimensionless quantities are dimensionless. The sum of dimensionless quantities is dimensionless. The exponential of a dimensionless quantity is dimensionless. Lengths are counts of tokens, hence dimensionless. Ratios of lengths are dimensionless. The brevity penalty BP is dimensionless. Therefore, BLEU is a dimensionless quantity.

Mechanism and Formal Derivation

Perplexity Derivation:
The core idea behind perplexity is to quantify how well a probabilistic model predicts a sequence of events.
1.  **Probability of a Sequence**: A language model P_M assigns a probability to a sequence S = (w_1, w_2, ..., w_N) by the chain rule of probability:
    P_M(S) = P_M(w_1) * P_M(w_2 | w_1) * ... * P_M(w_N | w_1...w_{N-1})
    P_M(S) = product_{i=1 to N} P_M(w_i | w_1...w_{i-1})
    (Note: For w_1, P_M(w_1 | w_1...w_0) simplifies to P_M(w_1)).

2.  **Negative Log-Likelihood (NLL)**: To avoid numerical underflow with very small probabilities and to convert products into sums, the negative logarithm of the sequence probability is often used:
    -log P_M(S) = -log( product_{i=1 to N} P_M(w_i | w_1...w_{i-1}) )
    -log P_M(S) = -sum_{i=1 to N} log P_M(w_i | w_1...w_{i-1})

3.  **Average Negative Log-Likelihood (Cross-Entropy Loss)**: To normalize for sequence length and obtain a per-word measure, the NLL is averaged over the number of tokens N:
    L(S) = -1/N * sum_{i=1 to N} log P_M(w_i | w_1...w_{i-1})
    This L(S) is also known as the cross-entropy loss per word for the sequence S.

4.  **Exponentiation**: Perplexity is defined as the exponentiation of this average NLL:
    PPL(S) = exp( L(S) )
    PPL(S) = exp( -1/N * sum_{i=1 to N} log P_M(w_i | w_1...w_{i-1}) )

5.  **Relationship to Geometric Mean**: By properties of logarithms and exponentials, this can be rewritten as:
    PPL(S) = ( product_{i=1 to N} P_M(w_i | w_1...w_{i-1}) )^(-1/N)
    PPL(S) = ( product_{i=1 to N} 1 / P_M(w_i | w_1...w_{i-1}) )^(1/N)
    This form explicitly shows PPL as the geometric mean of the inverse probabilities.

6.  **Interpretation and Utility**: A lower perplexity value indicates that the model assigns higher probabilities to the observed sequence, implying a better fit to the data. It provides an intuitive measure of how "surprised" the model is by the test data. For example, a perplexity of 10 means the model is, on average, as uncertain as if it had to choose uniformly from 10 words at each step.

BLEU Derivation:
BLEU aims to measure the similarity between a candidate translation and one or more reference translations by counting matching n-grams.
1.  **N-gram Precision (Initial Concept)**: For a single candidate C and a single reference R, the precision for n-grams P_n(C, R) would be:
    P_n(C, R) = (Number of n-grams in C that also appear in R) / (Total number of n-grams in C)
    This simple precision has a flaw: a candidate could achieve high precision by generating very few words, all of which are correct.

2.  **Modified N-gram Precision (Clipping)**: To address the flaw, BLEU introduces "modified precision" by clipping the count of n-grams. For an n-gram g, its count in the candidate is clipped by its maximum count in any single reference.
    count_clip(g, C, R) = min( count(g, C), max_{R_j in R} count(g, R_j) )
    This prevents overcounting of n-grams that appear many times in the candidate but only once or twice in the references.

3.  **Corpus-Level Modified N-gram Precision**: To obtain a robust score, BLEU is typically calculated over an entire corpus, not just individual sentences. The modified n-gram precision P_n for the corpus is calculated by summing the clipped n-gram counts and candidate n-gram counts over all sentences in the corpus:
    P_n = ( sum_{C in Corpus} sum_{g in C_n} count_clip(g, C, R) ) / ( sum_{C in Corpus} sum_{g in C_n} count(g, C) )
    This aggregation prevents issues with very short sentences and provides a more stable score.

4.  **Brevity Penalty (BP)**: To penalize candidates that are too short (which tend to have higher precision but omit information), a brevity penalty is introduced.
    BP = min( 1, exp(1 - L_R_eff / L_C) )
    Here, L_C is the total length of all candidate sentences. L_R_eff is the effective reference length, which is the sum of the lengths of the reference sentences that are closest to the length of their respective candidate sentences. Specifically, for each candidate C_k, find the reference R_j_k such that L_R_j_k >= L_C_k and L_R_j_k is minimized. If no such reference exists, choose the reference with the maximum length. Sum these chosen lengths over the corpus to get L_R_eff. The penalty is 1 if L_C >= L_R_eff, and less than 1 otherwise, exponentially decreasing as L_C becomes much shorter than L_R_eff.

5.  **Combined BLEU Score**: The final BLEU score combines the modified n-gram precisions using a weighted geometric mean and applies the brevity penalty.
    BLEU = BP * exp( sum_{n=1 to N_max} w_n * log(P_n) )
    Typically, N_max=4 (considering unigrams, bigrams, trigrams, and 4-grams) and w_n=1/4 for all n. The logarithm converts the product of precisions into a sum, which is then exponentiated.

6.  **Rationale**: The geometric mean ensures that if any single n-gram precision (e.g., for 4-grams) is zero, the entire product becomes zero, resulting in a zero BLEU score. This reflects that higher-order n-grams are crucial for fluency and accuracy. The brevity penalty addresses the precision bias towards short outputs.

Computational and Complexity Analysis

Perplexity:
Time Complexity:
For a single sequence of length N, computing the perplexity involves N conditional probability calculations and N logarithm operations, followed by a sum and an exponential.
Each P_M(w_i | w_1...w_{i-1}) typically involves a forward pass through the language model. For a Transformer-based decoder, this involves attention mechanisms and feed-forward networks. If the model is autoregressive and generates one token at a time, the complexity for predicting w_i is roughly O(i * d_model^2) or O(i^2 * d_model) depending on the attention implementation, where d_model is the model dimension. However, for *evaluating* a given sequence, the model can compute all P_M(w_i | w_1...w_{i-1}) in parallel up to a certain context window or using masked attention.
For a Transformer decoder, computing all N probabilities for a sequence of length N:
-   Self-attention: O(N^2 * d_model) for a single layer. With L layers, O(L * N^2 * d_model).
-   Feed-forward networks: O(N * d_model * d_ff) for a single layer. With L layers, O(L * N * d_model * d_ff).
-   Softmax over vocabulary V: O(N * V).
Thus, the dominant factor is typically O(L * N^2 * d_model + N * V). If the probabilities are pre-computed (e.g., from a logit output tensor of shape (N, V)), then the calculation of PPL is O(N) for the sum and exp.
For a corpus of M sequences, the total time complexity is M times the complexity for a single sequence.

Memory Complexity:
Storing the sequence S requires O(N) memory. Storing the probabilities P_M(w_i | w_1...w_{i-1}) or the log-probabilities requires O(N) memory. The language model itself requires O(P) memory, where P is the number of parameters. The softmax output layer requires O(V) memory.

Effect of Scaling Key Parameters:
-   Increasing N (sequence length): Quadratic impact on Transformer attention computation, linear on softmax and sum.
-   Increasing V (vocabulary size): Linear impact on softmax computation.
-   Increasing d_model (model dimension): Quadratic impact on attention, linear on FFN.
-   Increasing L (number of layers): Linear impact on overall model computation.

BLEU:
Time Complexity:
The most computationally intensive part of BLEU is n-gram counting and matching.
For a single candidate C of length L_C and K references R_j of average length L_R_avg, and maximum n-gram order N_max:
-   Generating n-grams for C: O(L_C * N_max).
-   Generating n-grams for R_j: O(L_R_avg * N_max) for each reference. Total O(K * L_R_avg * N_max).
-   Counting and clipping n-grams: This involves iterating through candidate n-grams and looking them up in reference n-gram counts. Using hash maps for n-gram counts, lookup is O(1) on average.
    -   Building candidate n-gram counts: O(L_C * N_max).
    -   Building reference n-gram counts: O(K * L_R_avg * N_max).
    -   Clipping and summing: O(L_C * N_max) for iterating through candidate n-grams.
Total time complexity for a single (C, R) pair is approximately O( (L_C + K * L_R_avg) * N_max ).
For a corpus of M candidate-reference pairs, the total time complexity is O( M * (L_C_avg + K_avg * L_R_avg) * N_max ).

Memory Complexity:
Storing n-gram counts for a candidate requires O(L_C * N_max) memory (number of unique n-grams can be up to L_C * N_max). For references, O(K * L_R_avg * N_max).

Effect of Scaling Key Parameters:
-   Increasing L_C or L_R_avg: Linear impact on n-gram counting.
-   Increasing K (number of references): Linear impact on reference n-gram processing.
-   Increasing N_max (max n-gram order): Linear impact on n-gram generation and storage.

Trade-offs:
Perplexity is computationally efficient for evaluation once model probabilities are obtained. It is a direct measure of model fit to the data distribution. BLEU is more computationally intensive due to n-gram processing, especially with many references and long sentences. Perplexity is objective and model-centric, while BLE