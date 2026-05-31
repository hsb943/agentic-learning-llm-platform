Title
Tokenization Techniques: BPE, WordPiece, and SentencePiece

Assumptions and Notation
This document assumes a foundational understanding of natural language processing concepts and basic probability theory.
- $\Sigma$: The universal character alphabet.
- $S$: The training corpus, a collection of text documents.
- $W$: The set of unique words in the training corpus $S$.
- $V_c$: The initial character vocabulary, typically $\Sigma$.
- $V_t$: The target token vocabulary size, a hyperparameter.
- $T$: The final token vocabulary, $|T| = V_t$.
- $w$: A word from the corpus, $w \in W$.
- $c$: A character, $c \in \Sigma$.
- $s$: A subword unit, $s \in T$.
- $N$: The total number of words in the training corpus $S$.
- $L_{avg}$: The average length of a word in characters.
- $L_{max}$: The maximum length of a word in characters.
- $L_{seq}$: The length of an input text sequence in characters.
- $L_{token\_avg}$: The average length of a token in characters.
- $L_{token\_max}$: The maximum length of a token in characters.
- $P(s)$: The probability of a subword $s$ occurring, used in Unigram Language Model.
- $f(x)$: Frequency of an item $x$ in the corpus.
- $f(x, y)$: Frequency of an adjacent pair $(x, y)$ in the corpus.

Core Concepts and Mathematical Foundations
Tokenization is the process of segmenting a text sequence into smaller units called tokens. Traditional tokenization operates at either the character level or the word level. Character-level tokenization results in very long sequences, increasing computational cost for sequence models. Word-level tokenization faces the "out-of-vocabulary" (OOV) problem, where unseen words cannot be represented, and leads to extremely large vocabularies for morphologically rich languages. Subword tokenization techniques address these limitations by finding a balance, decomposing words into meaningful subword units.

Formally, a tokenization function $F: \text{Text} \rightarrow (s_1, s_2, \dots, s_k)$ maps an input text string to a sequence of subword tokens, where each $s_i \in T$. The objective is to learn a vocabulary $T$ and a segmentation strategy that minimizes the sequence length $k$ while maximizing the information content per token, subject to a constraint on the vocabulary size $V_t$. This can be viewed as a form of data compression, where the goal is to represent the corpus using a compact set of units.

The core idea behind these methods is to iteratively merge frequently co-occurring units (characters or subwords) into new, longer subword units. This process continues until a predefined vocabulary size $V_t$ is reached. The resulting vocabulary $T$ contains a mix of characters, common subwords, and frequent words.

**Probabilistic Interpretation (SentencePiece Unigram Model)**:
The Unigram Language Model approach, employed by SentencePiece, frames tokenization as a maximum likelihood estimation problem. Given a sentence $X = x_1x_2\dots x_m$, the goal is to find a segmentation $X = s_1s_2\dots s_k$ (where $s_i$ are subwords from $T$) that maximizes the probability of the segmentation:
$P(X) = \sum_{s \in \text{Segmentations}(X)} P(s_1, s_2, \dots, s_k)$
Assuming independence of subwords, this simplifies to:
$P(X) = \sum_{s \in \text{Segmentations}(X)} \prod_{i=1}^k P(s_i)$
The training objective is to learn the subword probabilities $P(s)$ for all $s \in T$ such that the likelihood of the entire training corpus is maximized. During inference, the Viterbi algorithm is used to find the single most probable segmentation:
$\text{argmax}_{s \in \text{Segmentations}(X)} \prod_{i=1}^k P(s_i)$

Mechanism and Formal Derivation

**1. Byte Pair Encoding (BPE)**
BPE is a data compression algorithm adapted for text tokenization. It iteratively merges the most frequent adjacent character or subword pairs.

*   **Step 1: Initialization of Base Vocabulary.**
    The initial vocabulary $T_0$ consists of all unique characters present in the training corpus $S$. Each word $w \in W$ is represented as a sequence of characters, often with a special end-of-word token (e.g., `_`) appended to distinguish full words. For example, "low" becomes "l o w _".
*   **Step 2: Frequency Counting of Pairs.**
    Iterate through all words in the corpus. For each word, count the frequency of all adjacent pairs of units. Let $f(u_i, u_{i+1})$ be the frequency of the pair $(u_i, u_{i+1})$ where $u_i, u_{i+1} \in T_k$ (the current vocabulary).
*   **Step 3: Identification of Most Frequent Pair.**
    Identify the pair $(u_a, u_b)$ with the highest frequency $f(u_a, u_b)$. If multiple pairs have the same maximum frequency, a deterministic tie-breaking rule (e.g., lexicographical order) is applied.
*   **Step 4: Merge Operation and Vocabulary Expansion.**
    Create a new subword unit $u_{new} = u_a u_b$. Add $u_{new}$ to the vocabulary: $T_{k+1} = T_k \cup \{u_{new}\}$.
*   **Step 5: Corpus Update.**
    Replace all occurrences of the pair $(u_a, u_b)$ with the new unit $u_{new}$ in the corpus representation. For example, if "l o" is the most frequent pair, "l o w _" becomes "lo w _". This effectively shortens the sequence representation of words containing the merged pair.
*   **Step 6: Iteration until Target Vocabulary Size.**
    Repeat Steps 2-5 until the desired vocabulary size $V_t$ is reached, or no more pairs can be merged (i.e., all remaining pairs have frequency 1). The final vocabulary $T$ contains characters, subwords, and full words.

**Inference (Tokenization of New Text):**
To tokenize a new text sequence, the learned merge operations are applied greedily. The longest possible subword from $T$ that matches a prefix of the remaining text is chosen. If no subword matches, the smallest unit (character) is used. This process continues until the entire sequence is tokenized.

**2. WordPiece**
WordPiece, used in models like BERT, is similar to BPE but uses a different scoring mechanism for merging pairs. It aims to select merges that maximize the likelihood of the training corpus.

*   **Step 1: Initialization with Base Vocabulary.**
    Similar to BPE, the initial vocabulary $T_0$ consists of all unique characters. Words are typically pre-tokenized (e.g., by whitespace) and then represented as sequences of characters. A special prefix (e.g., '##' or ' ') is often used to denote subwords that are not at the beginning of a word. For example, "tokenization" might be represented as "t o k e n i z a t i o n".
*   **Step 2: Corpus Likelihood Calculation.**
    The algorithm implicitly aims to maximize the likelihood of the training corpus. The probability of a word $w$ being segmented into $s_1, s_2, \dots, s_k$ is $P(w) = \prod_{i=1}^k P(s_i)$. The probabilities $P(s)$ are estimated from the corpus.
*   **Step 3: Scoring Candidate Merges.**
    Instead of raw frequency, WordPiece evaluates candidate merges based on how much they increase the overall corpus likelihood. For a candidate merge of units $u_a$ and $u_b$ into $u_{new} = u_a u_b$, the score is often defined as:
    $\text{score}(u_a, u_b) = \frac{f(u_a u_b)}{f(u_a) \cdot f(u_b)}$
    This score represents the ratio of the frequency of the merged unit to the product of the frequencies of its constituent parts. A higher score indicates that the merge is "more surprising" or "less random" than if $u_a$ and $u_b$ appeared independently.
*   **Step 4: Identification of Best Merge.**
    Identify the pair $(u_a, u_b)$ that yields the highest $\text{score}(u_a, u_b)$.
*   **Step 5: Merge Operation and Corpus Update.**
    Create $u_{new} = u_a u_b$, add it to the vocabulary, and replace all occurrences of $(u_a, u_b)$ with $u_{new}$ in the corpus.
*   **Step 6: Iteration until Target Vocabulary Size.**
    Repeat Steps 2-5 until the desired vocabulary size $V_t$ is reached.

**Inference (Tokenization of New Text):**
Similar to BPE, WordPiece tokenization is greedy. Given a word, it attempts to find the longest possible subword from its vocabulary that matches a prefix of the word. If a match is found, that subword is emitted, and the process continues with the remainder of the word. If no subword matches, the remaining characters are typically broken down into individual characters, often prefixed with '##' to indicate they are not word-initial.

**3. SentencePiece**
SentencePiece is a language-agnostic subword tokenizer that treats the input text as a raw stream of characters, including whitespace. It does not rely on pre-tokenization and can train either a BPE or a Unigram Language Model.

*   **Step 1: Raw Text Input and Initial Vocabulary.**
    SentencePiece operates directly on raw text. Whitespace is treated as a regular character, often prefixed with ' ' (U+2581, lower-half block character) to distinguish it from internal word spaces. The initial vocabulary $T_0$ for the Unigram model typically includes all unique characters and all substrings up to a certain length found in the corpus. For BPE mode, it's just unique characters.
*   **Step 2: Model Training (Unigram Language Model).**
    For the Unigram model, the goal is to learn subword probabilities $P(s)$ for each $s \in T$. This is an iterative process:
    *   **E-step (Expectation):** Given current subword probabilities $P(s)$, for each sentence $X$ in the corpus, find the most probable segmentation $X = s_1s_2\dots s_k$ using the Viterbi algorithm. This maximizes $\prod_{i=1}^k P(s_i)$.
    *   **M-step (Maximization):** Update the subword probabilities $P(s)$ based on the segmentations found in the E-step. $P(s) = \frac{\text{count}(s)}{\sum_{s' \in T} \text{count}(s')}$.
    These steps are repeated until convergence or a fixed number of iterations.
*   **Step 3: Vocabulary Pruning (Unigram Model).**
    Starting with a large initial vocabulary, the Unigram model iteratively prunes tokens. In each iteration, a small percentage (e.g., 10-20%) of tokens are removed. The tokens chosen for removal are those whose removal would cause the *least decrease* in the overall corpus likelihood. This is often approximated by removing tokens $s$ with the smallest $\text{loss}(s) = \sum_{X \in S} \log \left( \frac{P(X | T \setminus \{s\})}{P(X | T)} \right)$. This process continues until the target vocabulary size $V_t$ is reached.
*   **Step 4: BPE Mode (Alternative to Unigram).**
    If configured for BPE mode, SentencePiece follows the standard BPE algorithm (Steps 1-6 described above), but crucially, it operates on the raw text stream including whitespace characters, without pre-tokenization.
*   **Step 5: Final Vocabulary Construction.**
    The final vocabulary $T$ consists of the learned subword units and their associated probabilities (for Unigram) or merge rules (for BPE).
*   **Step 6: Inference (Tokenization of New Text).**
    For the Unigram model, the Viterbi algorithm is used to find the single best segmentation of an input text sequence $X$ into subwords $s_1, \dots, s_k$ from $T$ that maximizes $\prod_{i=1}^k P(s_i)$. For BPE mode, the greedy longest-match strategy is used.

**Dimensional Consistency:**
While not involving matrix dimensions, the process maintains dimensional consistency in terms of sequence length. Each merge operation reduces the number of units in a word's representation by one. For example, merging $(u_a, u_b)$ into $u_{new}$ transforms a sequence $(..., u_a, u_b, ...)$ into $(..., u_{new}, ...)$, reducing the sequence length by one unit. The vocabulary size $V_t$ is a scalar constraint.

Computational and Complexity Analysis

**1. BPE and WordPiece (Training)**
*   **Time Complexity:**
    *   Naive implementation: In each of $V_t$ merge iterations, counting pairs can take $O(N \cdot L_{avg})$ time (iterating through all words and their characters). Replacing pairs can also take $O(N \cdot L_{avg})$. Total: $O(V_t \cdot N \cdot L_{avg})$.
    *   Optimized implementation: Using hash maps for pair counts and a priority queue for efficient retrieval of the most frequent pair, counting can be $O(N \cdot L_{avg})$ initially, and subsequent updates after a merge can be $O(L_{avg})$ for affected words. The total complexity can be reduced to approximately $O(V_t \cdot N \cdot \log V_t)$ or $O(V_t \cdot N)$ with highly optimized data structures.
*   **Memory Complexity:**
    *   Storing the corpus: $O(N \cdot L_{avg})$ characters.
    *   Storing pair frequencies: $O(V_c^2)$ in the worst case (all character pairs), but practically $O(N \cdot L_{avg})$ unique pairs.
    *   Storing the vocabulary: $O(V_t \cdot L_{token\_max})$ characters.

**2. SentencePiece (Unigram Model Training)**
*   **Time Complexity:**
    *   Initial vocabulary generation: $O(N \cdot L_{avg}^2)$ for all substrings.
    *   EM iterations: Each iteration involves an E-step (Viterbi) and an M-step.
        *   E-step (Viterbi): For each sentence of length $L_{seq}$, Viter