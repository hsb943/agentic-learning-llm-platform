Title
Softmax Scaling and Numerical Stability

Assumptions and Notation
Let $z$ be an input vector of unnormalized logit scores, $z \in \mathbb{R}^K$, where $K$ is the number of classes or elements.
Let $z_i$ denote the $i$-th element of the vector $z$, for $i \in \{1, \dots, K\}$.
Let $S$ be the output vector of probabilities, $S \in \mathbb{R}^K$.
Let $S_i$ denote the $i$-th element of the vector $S$, representing the probability of class $i$.
Let $C$ be a scalar scaling constant, $C \in \mathbb{R}$.
Let $\max_j(z_j)$ denote the maximum value among all elements of $z$.
Let $\exp(x)$ denote the exponential function $e^x$.
Let $\log(x)$ denote the natural logarithm function.
Let $\epsilon$ represent machine epsilon, the smallest number such that $1 + \epsilon > 1$ in floating-point arithmetic.
Floating-point numbers are assumed to conform to IEEE 754 standard.
Overflow refers to a number exceeding the maximum representable floating-point value, typically resulting in `Inf` (infinity).
Underflow refers to a number becoming too small to be represented as a normalized floating-point number, typically resulting in `0`.
`NaN` refers to "Not a Number", an undefined or unrepresentable value.

Core Concepts and Mathematical Foundations
The Softmax function is a generalization of the logistic function to multiple dimensions. It maps a vector of arbitrary real values to a probability distribution over $K$ classes. Each component of the output vector $S$ is a probability, meaning $0 < S_i < 1$ for all $i$, and the sum of all components is exactly 1, i.e., $\sum_{i=1}^K S_i = 1$.

Formal Definition:
Given an input vector $z \in \mathbb{R}^K$, the Softmax function computes the output vector $S \in \mathbb{R}^K$ where each component $S_i$ is defined as:
$S_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$

Probabilistic Interpretation:
The value $S_i$ can be interpreted as the estimated probability that the input belongs to class $i$, given the unnormalized logit scores $z$. The higher the logit score $z_i$ for a particular class $i$, the higher its corresponding probability $S_i$ will be relative to other classes. The exponential function ensures that all intermediate values $\exp(z_j)$ are positive, and thus all $S_i$ are positive. The normalization by the sum ensures that the output forms a valid probability distribution.

Geometric Interpretation:
The Softmax function maps a point in $K$-dimensional Euclidean space $\mathbb{R}^K$ to a point within the interior of the $(K-1)$-dimensional standard simplex. The standard simplex is the convex hull of the standard basis vectors in $\mathbb{R}^K$. For $K=2$, it maps to the open interval $(0,1)$ on the line segment connecting $(1,0)$ and $(0,1)$. For $K=3$, it maps to the interior of an equilateral triangle in 3D space.

Dimensional Reasoning:
Input: $z \in \mathbb{R}^K$. This is a vector of $K$ real-valued scores.
Output: $S \in \mathbb{R}^K$. This is a vector of $K$ real-valued probabilities.
Each $z_i$ is a scalar. Each $\exp(z_i)$ is a scalar. The sum $\sum_{j=1}^K \exp(z_j)$ is a scalar. The division of two scalars yields a scalar $S_i$. The operation is applied element-wise to the numerator and globally to the denominator. The dimensions are consistent.

Mechanism and Formal Derivation
The primary challenge with the standard Softmax formulation is numerical instability when the input logits $z_i$ are very large or very small. This can lead to floating-point overflow or underflow, resulting in `Inf`, `0`, or `NaN` values, which are incorrect and can halt computation. The solution involves a mathematical transformation that preserves the function's output while operating on numerically stable intermediate values.

Step 1: Standard Softmax Definition
The Softmax function for an input vector $z \in \mathbb{R}^K$ is defined as:
$S_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$
This formula is mathematically correct but computationally fragile.

Step 2: Identifying Numerical Instability
The exponential function $\exp(x)$ grows very rapidly for positive $x$ and approaches zero very rapidly for negative $x$.
-   **Overflow**: If any $z_i$ is a large positive number (e.g., $z_i > 709$ for standard 64-bit floating-point numbers), $\exp(z_i)$ will exceed the maximum representable floating-point value and result in `Inf`. If multiple $z_j$ values are large, the sum $\sum_{j=1}^K \exp(z_j)$ will also become `Inf`. A common failure mode is `Inf / Inf`, which evaluates to `NaN`.
-   **Underflow**: If all $z_i$ are very small negative numbers (e.g., $z_i < -745$ for standard 64-bit floating-point numbers), all $\exp(z_i)$ will underflow to `0`. In this case, the numerator becomes `0` and the denominator becomes `0`, leading to `0 / 0`, which evaluates to `NaN`.
-   **Mixed Overflow/Underflow**: If some $z_i$ are large positive and others are large negative, some $\exp(z_i)$ might be `Inf` while others are `0`. This can still lead to `Inf / Inf` or other problematic `NaN` forms.

Step 3: Introducing a Scaling Constant
The key insight for numerical stability is that the Softmax function is invariant to adding or subtracting a constant from all input logits $z_i$. This can be formally shown:
Let $C$ be an arbitrary scalar constant.
Consider the modified expression:
$S'_i = \frac{\exp(z_i - C)}{\sum_{j=1}^K \exp(z_j - C)}$
Using the property of exponents $\exp(a-b) = \exp(a) \exp(-b)$:
$S'_i = \frac{\exp(z_i) \exp(-C)}{\sum_{j=1}^K (\exp(z_j) \exp(-C))}$
Factor out $\exp(-C)$ from the sum in the denominator:
$S'_i = \frac{\exp(z_i) \exp(-C)}{\exp(-C) \sum_{j=1}^K \exp(z_j)}$
Since $\exp(-C)$ is a non-zero scalar, it can be cancelled from the numerator and denominator:
$S'_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$
Thus, $S'_i = S_i$. This demonstrates that subtracting any constant $C$ from all $z_i$ before applying the exponential function does not change the final Softmax output.

Step 4: Choosing the Optimal Scaling Constant for Stability
The choice of $C$ is critical for numerical stability. To prevent overflow in the numerator and denominator, we want to ensure that the arguments to the exponential function, $(z_i - C)$, are as small as possible without becoming excessively negative (which could lead to underflow). The optimal choice for $C$ is the maximum value among the input logits:
$C = \max_{j=1}^K (z_j)$
By choosing $C = \max_j(z_j)$, we guarantee that for all $i$:
$z_i - C \le 0$
This ensures that all arguments to the exponential function are non-positive. Consequently, all terms $\exp(z_i - C)$ will be in the range $(0, 1]$. Specifically, for the $k$-th element where $z_k = \max_j(z_j)$, we have $z_k - C = 0$, so $\exp(z_k - C) = \exp(0) = 1$. This prevents overflow for any individual $\exp(z_i - C)$ term.

Step 5: Formal Derivation of Stable Softmax
Using the chosen scaling constant $C = \max_j(z_j)$, the numerically stable Softmax formula becomes:
$S_i = \frac{\exp(z_i - \max_j(z_j))}{\sum_{j=1}^K \exp(z_j - \max_j(z_j))}$
Let $z'_i = z_i - \max_j(z_j)$.
Then, the stable Softmax can be written as:
$S_i = \frac{\exp(z'_i)}{\sum_{j=1}^K \exp(z'_j)}$
Here, all $z'_i \le 0$. At least one $z'_k = 0$ (for the $k$ where $z_k = \max_j(z_j)$).
Therefore, all $\exp(z'_i) \in (0, 1]$.
The sum $\sum_{j=1}^K \exp(z'_j)$ will be in the range $(0, K]$.
This formulation effectively prevents overflow in both the numerator and denominator. While individual terms $\exp(z'_i)$ might still underflow to `0` if $z'_i$ is a very large negative number, the presence of at least one term equal to `1` (from $\exp(0)$) in the sum prevents the denominator from becoming `0` unless $K=0$, which is not a valid input. This significantly mitigates the underflow problem.

Step 6: Log-Softmax and Log-Sum-Exp (LSE)
In many applications, especially with cross-entropy loss, the logarithm of the Softmax output, known as Log-Softmax, is required. Computing $\log(S_i)$ directly from the stable $S_i$ can still lead to numerical issues if $S_i$ is extremely small and underflows to `0` before the logarithm is taken, resulting in $\log(0)$ which is undefined (`-Inf`).
A numerically stable way to compute Log-Softmax is to use the Log-Sum-Exp (LSE) trick.
From Step 5, we have:
$S_i = \frac{\exp(z_i - C)}{\sum_{j=1}^K \exp(z_j - C)}$, where $C = \max_j(z_j)$.
Taking the logarithm of both sides:
$\log(S_i) = \log\left(\frac{\exp(z_i - C)}{\sum_{j=1}^K \exp(z_j - C)}\right)$
Using logarithm properties $\log(a/b) = \log(a) - \log(b)$:
$\log(S_i) = \log(\exp(z_i - C)) - \log\left(\sum_{j=1}^K \exp(z_j - C)\right)$
Using $\log(\exp(x)) = x$:
$\log(S_i) = (z_i - C) - \log\left(\sum_{j=1}^K \exp(z_j - C)\right)$
The term $\log\left(\sum_{j=1}^K \exp(z_j - C)\right)$ is the numerically stable Log-Sum-Exp function. By subtracting $C = \max_j(z_j)$ inside the sum before exponentiation, we ensure that the arguments to $\exp$ are non-positive, preventing overflow within the sum. The sum itself will be in $(0, K]$, so its logarithm will be well-defined and finite. This approach ensures that $\log(S_i)$ is computed accurately even when $S_i$ itself would underflow to zero.

Computational and Complexity Analysis
The numerical stabilization technique for Softmax introduces a small, constant-factor overhead in computation but does not change the asymptotic complexity.

Time Complexity:
Let $K$ be the number of elements in the input vector $z$.
1.  **Finding the maximum element**: Computing $C = \max_j(z_j)$ requires iterating through all $K$ elements of $z$. This operation has a time complexity of $O(K)$.
2.  **Subtracting the constant**: Computing $z'_i = z_i - C$ for all $K$ elements requires $K$ subtractions. This operation has a time complexity of $O(K)$.
3.  **Exponentiation**: Computing $\exp(z'_i)$ for all $K$ elements requires $K$ exponential operations. Assuming each $\exp$ operation takes constant time, this is $O(K)$.
4.  **Summation**: Computing $\sum_{j=1}^K \exp(z'_j)$ requires $K-1$ additions. This operation has a time complexity of $O(K)$.
5.  **Division**: Computing $S_i = \exp(z'_i) / \sum_{j=1}^K \exp(z'_j)$ for all $K$ elements requires $K$ divisions. This operation has a time complexity of $O(K)$.
Total Time Complexity: The sum of these steps is $O(K) + O(K) + O(K) + O(K) + O(K) = O(K)$.
The standard (unstable) Softmax also has $O(K)$ complexity (exponentiation, summation, division). The stable version adds a constant factor due to the initial pass for finding the maximum.

Memory Complexity:
The Softmax function requires storing the input vector $z \in \mathbb{R}^K$ and the output vector $S \in \mathbb{R}^K$. Intermediate values like $z'$ or the sum of exponentials can be computed in-place or require temporary storage proportional to $K$.
Total Memory Complexity: $O(K)$.

Effect of Scaling Key Parameters:
The primary parameter affecting complexity is $K$, the dimensionality of the input vector. As $K$ increases, both time and memory requirements scale linearly. The magnitude of the input values $z_i$ does not affect the asymptotic complexity but is the direct cause of numerical