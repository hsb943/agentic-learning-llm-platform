Quantization Techniques: INT8, 4-bit, and QLoRA

Assumptions and Notation
This document assumes familiarity with linear algebra, matrix operations, and fundamental concepts of neural networks. All operations are performed in a standard floating-point environment unless explicitly stated as quantized.

W: Full-precision weight matrix, W ∈ ℝ^(d_out × d_in).
W_q: Quantized weight matrix, W_q ∈ ℤ^(d_out × d_in).
W_dq: Dequantized weight matrix, W_dq ∈ ℝ^(d_out × d_in).
X: Input activation matrix, X ∈ ℝ^(n × d_in).
Y: Output activation matrix, Y ∈ ℝ^(n × d_out).
s: Floating-point scaling factor, s ∈ ℝ. This can be a scalar (per-tensor), a vector (per-axis/per-channel), or a matrix (per-group).
z: Integer zero-point, z ∈ ℤ. This can be a scalar, a vector, or a matrix, corresponding to the scope of s.
b: Bit-width for quantization (e.g., 8 for INT8, 4 for 4-bit).
Q_min: Minimum representable integer value for the given bit-width b. For signed integers, Q_min = -2^(b-1).
Q_max: Maximum representable integer value for the given bit-width b. For signed integers, Q_max = 2^(b-1) - 1.
R_min: Minimum observed or estimated real-valued range for a tensor or group of values.
R_max: Maximum observed or estimated real-valued range for a tensor or group of values.
d_in: Input dimension of a linear layer or matrix.
d_out: Output dimension of a linear layer or matrix.
n: Batch size.
k: Inner dimension for matrix multiplication (e.g., d_in for X @ W).
D_fp: Full-precision data type, typically FP32 or BF16.
D_q: Quantized data type, typically INT8 or INT4.
r: Rank of the low-rank approximation in LoRA, r ∈ ℕ, r << min(d_in, d_out).
A: LoRA down-projection matrix, A ∈ ℝ^(d_in × r).
B: LoRA up-projection matrix, B ∈ ℝ^(r × d_out).
ΔW: LoRA update matrix, ΔW = B A ∈ ℝ^(d_out × d_in).
W_0: Base model weight matrix in QLoRA, W_0 ∈ ℝ^(d_out × d_in).
W_0q: Quantized base model weight matrix in QLoRA, W_0q ∈ ℤ^(d_out × d_in).
s_s: Scaling factor for double quantization.
z_s: Zero-point for double quantization.
g: Group size for group-wise quantization.

Core Concepts and Mathematical Foundations
Quantization is a process of mapping a continuous or large set of discrete values (typically floating-point numbers) to a smaller set of discrete values (typically integers). This is a lossy compression technique. The primary goal is to reduce memory footprint and computational cost while maintaining acceptable model accuracy.

Formal Definitions:
Affine Quantization: The most common form of quantization, which involves a linear scaling and an integer offset (zero-point).
The quantization function Q(x) maps a real value x to an integer q:
q = round(x / s + z)
The dequantization function DQ(q) maps an integer q back to an approximate real value x_dq:
x_dq = s * (q - z)

The scaling factor s and zero-point z are crucial for mapping the full-precision range [R_min, R_max] to the integer range [Q_min, Q_max].

Symmetric Quantization: A special case where the real-valued range is symmetric around zero, i.e., [-R_max_abs, R_max_abs], and the zero-point z is typically set to 0.
s = R_max_abs / Q_max
z = 0
In this case, Q_min is often -Q_max. For INT8, Q_max = 127, so Q_min = -127.

Asymmetric Quantization: The more general case where the real-valued range [R_min, R_max] is not necessarily symmetric around zero.
s = (R_max - R_min) / (Q_max - Q_min)
z = Q_min - round(R_min / s)
The zero-point z ensures that the real value 0.0 maps to an integer value that, when dequantized, is as close to 0.0 as possible.

Geometric or Probabilistic Interpretation:
Quantization can be viewed as partitioning the real number line into Q_max - Q_min + 1 intervals, each of width s. All real numbers within an interval are mapped to the same integer value. The `round` function determines which integer value is chosen. The choice of s and z effectively shifts and scales this grid of intervals to best cover the distribution of the original floating-point values. From a probabilistic perspective, quantization introduces noise, and the goal is to minimize the mean squared error (MSE) between the original and dequantized values, or to preserve the relative order and distribution characteristics.

Dimensional Reasoning:
For a weight matrix W ∈ ℝ^(d_out × d_in), quantization can be applied at different granularities:
1. Per-tensor: A single (s, z) pair is computed for the entire matrix W. W_q ∈ ℤ^(d_out × d_in) and W_dq ∈ ℝ^(d_out × d_in).
2. Per-axis (or per-channel): A (s, z) pair is computed for each row or column of W. For example, if per-output-channel, s ∈ ℝ^(d_out × 1) and z ∈ ℤ^(d_out × 1). Then, W_q[i, j] = round(W[i, j] / s[i] + z[i]).
3. Per-group: A (s, z) pair is computed for a contiguous block of elements within W. For example, if weights are grouped into g elements, then s and z would be vectors of size (d_out * d_in) / g. This is common in 4-bit quantization.

INT8 Quantization:
Uses b=8 bits. For signed integers, Q_min = -128 and Q_max = 127. This provides 256 distinct integer values. It is widely supported by hardware (e.g., NVIDIA Tensor Cores).

4-bit Quantization:
Uses b=4 bits. For signed integers, Q_min = -8 and Q_max = 7. This provides 16 distinct integer values. This offers a 2x memory reduction compared to INT8 but introduces significantly more quantization error. To mitigate this, 4-bit quantization is almost always applied in a group-wise manner, meaning a separate (s, z) pair is used for small groups of weights (e.g., 32 or 64 elements).

QLoRA (Quantized Low-Rank Adaptation):
QLoRA is a fine-tuning technique that combines 4-bit quantization of the base model weights with Low-Rank Adaptation (LoRA). The key idea is to quantize the large, pre-trained model to 4-bit (e.g., using NormalFloat4 or NF4) and then fine-tune it by adding small, full-precision (e.g., FP16) LoRA adapter matrices.
It introduces two specific innovations:
1. Double Quantization (DQ): Quantizing the quantization constants (s and z) themselves to reduce their memory footprint.
2. Paged Optimizers: A memory optimization technique to manage memory spikes during training, especially for large models, by offloading optimizer states to CPU RAM and paging them to GPU VRAM as needed.

Mechanism and Formal Derivation
The core mechanism involves mapping floating-point values to integers and back, with specific considerations for different bit-widths and the QLoRA approach.

Step 1: Determine the Quantization Range [R_min, R_max]
For a given tensor W (or a group of elements within W), identify its minimum and maximum floating-point values.
R_min = min(W_ij for all i, j in W)
R_max = max(W_ij for all i, j in W)
This step is critical as it defines the span of values that need to be represented by the fixed integer range. For symmetric quantization, R_max_abs = max(|R_min|, |R_max|), and the range becomes [-R_max_abs, R_max_abs].

Step 2: Calculate Scaling Factor s and Zero-Point z
Using the determined range and the target integer range [Q_min, Q_max]:
For asymmetric quantization:
s = (R_max - R_min) / (Q_max - Q_min)
z = Q_min - round(R_min / s)
For symmetric quantization (assuming Q_min = -Q_max):
s = R_max_abs / Q_max
z = 0
The scaling factor s determines the precision (the size of the smallest representable step), and the zero-point z aligns the integer range with the real-valued range.
Dimensional consistency: If W ∈ ℝ^(d_out × d_in) and quantization is per-tensor, s and z are scalars. If per-axis (e.g., per-output-channel), s ∈ ℝ^(d_out × 1) and z ∈ ℤ^(d_out × 1), and the operations are broadcasted.

Step 3: Quantize the Full-Precision Weights W to W_q
Apply the quantization function element-wise:
W_q[i, j] = round(W[i, j] / s + z)
The resulting W_q ∈ ℤ^(d_out × d_in) where each element W_q[i, j] is constrained to be within [Q_min, Q_max].
Dimensional consistency: W_q has the same dimensions as W.

Step 4: Dequantize W_q to W_dq (for computation or verification)
Apply the dequantization function element-wise:
W_dq[i, j] = s * (W_q[i, j] - z)
This W_dq ∈ ℝ^(d_out × d_in) is an approximation of the original W. The difference W - W_dq represents the quantization error.
Dimensional consistency: W_dq has the same dimensions as W.

Step 5: Perform Quantized Matrix Multiplication (QGEMM)
The goal is to compute Y = X @ W_dq efficiently. While W_dq is conceptually used, in practice, specialized hardware and software kernels perform this operation directly using W_q.
Let X ∈ ℝ^(n × d_in) and W_dq ∈ ℝ^(d_in × d_out).
Y = X @ W_dq
Y[r, c] = Σ_k X[r, k] * W_dq[k, c]
Substituting W_dq:
Y[r, c] = Σ_k X[r, k] * (s_W * (W_q[k, c] - z_W))
Y[r, c] = s_W * Σ_k X[r, k] * (W_q[k, c] - z_W)
If activations X are also quantized (X_q, s_X, z_X), the computation can be further optimized to integer arithmetic:
Y[r, c] = s_X * s_W * Σ_k (X_q[r, k] - z_X) * (W_q[k, c] - z_W) + bias_terms
This sum of products of integers can be computed efficiently using integer arithmetic, often accumulating into a higher-precision integer (e.g., INT32) to prevent overflow, and then scaled back to floating-point. The actual implementation details vary by hardware and library but aim to minimize floating-point operations.

Step 6: QLoRA Specifics: 4-bit Base Model with Full-Precision LoRA Adapters and Double Quantization
For QLoRA, the base model weights W_0 are quantized to 4-bit (W_0q). The LoRA adapters ΔW = B A are kept in full precision (e.g., FP16).
During the forward pass, the effective weight matrix is W_eff = W_dq + ΔW.
Y = X @ W_eff = X @ (W_dq + B A)
Y = X @ W_dq + X @ B A
The term X @ W_dq is computed using 4-bit QGEMM (as in Step 5, but with 4-bit W_q). The term X @ B A involves full-precision matrix multiplications.
Double Quantization (DQ): The scaling factors s and zero-points z for the 4-bit base model weights are themselves floating-point numbers. To save memory, these (s, z) constants are also quantized, typically to 8-bit floating-point (e.g., FP8) or 8-bit integers.
Let s_orig be an original scaling factor.
s_s = (max(s_orig) - min(s_orig)) / (Q_max_s - Q_min_s) (scaling factor for scales)
z_s = Q_min_s - round(min(s_orig) / s_s) (zero-point for scales)
s_q = round(s_orig / s_s + z_s) (quantized scale)
s_dq = s_s * (s_q - z_s) (dequantized scale, used in computation)
This process is applied to the scaling factors and zero-points of the base model weights. This reduces the memory footprint of the quantization metadata by approximately 8x.
Paged Optimizers: During fine-tuning, optimizer states (e.g., Adam's first and second moments) can be very large. Paged Optimizers allocate these states in pages and move them between CPU RAM and GPU VRAM on demand, preventing out-of-memory errors for large models. This is a memory management technique rather than a quantization technique, but it is integral to QLoRA's ability to fine-tune large models on consumer GPUs.

Computational and Complexity Analysis
Quantization primarily impacts memory and computational efficiency.

Memory Complexity:
Let W ∈ ℝ^(d_out × d_in).
1. Full-precision (e.g., FP16): Memory = d_out * d_in * sizeof(FP16).
   Complexity: O(d_out * d_in).
2. INT8 Quantization: Memory = d_out * d_in * sizeof(INT8) + metadata_memory.
   sizeof(INT8) is 1 byte. This is an 8x reduction compared to FP16 (2 bytes).
   Metadata_memory: For per-tensor, 2 scalars (s, z). For per-axis (e.g., d_out axis), 2 * d_out scalars. This is negligible compared to the weight matrix size for large d_out, d_in.
   