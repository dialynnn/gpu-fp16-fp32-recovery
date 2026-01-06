# gpu-fp16-fp32-recovery

This repository contains experimental CUDA and CUTLASS implementations of mixed-precision matrix multiplication using **dual FP16 representations** with **FP32 accumulation**, inspired by [recent work](https://arxiv.org/pdf/2203.03341) on precision recovery for Tensor Core GEMMs. 

The goal of this project is **numerical behavior analysis**, not beating cuBLAS.

The author of this repo **IS NOT** the author of the paper nor affiliated with the authors.
---
## Overview

Modern GPUs execute FP16 Tensor Core GEMMs extremely fast, but naive FP16 inputs, when accumulated to FP32, can lead to large numerical error after computation, especially with large matrices. This project explores a technique where each FP32 input value is decomposed into:

- a high FP16 component, and
- a scaled low FP16 residual

And then computed as a combination of:

$$
\begin{align}
	A_{hi} \cdot B_{hi} \\
	A_{lo} \cdot B_{hi} \\
	A_{hi} \cdot B_{lo} \\
\end{align}
$$

Where both $A_{lo}$ and $B_{lo}$ are scaled with $2^{11}$, and final accumulation is done in FP32 (except for partial/temporary values). We then reported the accuracy numbers (absolute and relative) which are compared for analysis. Detailed breakdown can be understood in the referenced paper.

---
## Requirements

- NVIDIA GPU with Tensor Cores (tested on V100)
- CUDA 12.2
- CUTLASS 4.3.4 (coming soon, need to fix the data generator)

## Building

To compile the ```.cu``` kernels, compile with the target architecture (in our case its ``-arch=sm_70``) using nvcc.

```bash
nvcc -O3 -arch=sm_70 wmma_with_acc_recovery.cu -o wmma_with_acc_recovery
```
For CUTLASS based kernels, export the cutlass directory first and then compile (we provided example at ``compile_cutlass.sh``).

```bash
export cutlass_dir=*YOUR_CUTLASS_DIRECTORY*
```
Followed by
```bash
nvcc -O3  -std=c++17 -arch=sm_70 -I${cutlass_dir}/include cutlass_baseline.cu -o cutlass_baseline
```

---
## Results

We evaluated the proposed 2xFP16 to FP32 GEMM accumulation recovery method on an NVIDIA V100 (SM70) using WMMA-based Tensor Core kernels. The original paper used MMA for finer register controls, but we used WMMA for the ease of reproduction.

### Baseline performance (w/o precision recovery)
- **Throughput:** ~**7.5 TFLOP/s**
- **Runtime:** ~**0.5-0.6 ms**
- **Effective FLOP count:** `2 × M × N × K`

### Performance w/ precision recovery
- **Throughput:** ~**7.5 TFLOP/s**
- **Runtime:** ~**1.7 ms**
- **Effective FLOP count:** `3 × 2 × M × N × K`

Despite executing three GEMM-equivalent operations, the kernel achieves throughput comparable to a single baseline WMMA GEMM throughput wise. However, we believe further optimization to increase the throughput performance can be done.

### Accuracy
Compared against a full FP32 CPU reference:

### Baseline accuracy (w/o precision recovery)
- **Absolute error:** ~**6.35**
- **Relative error:** ~**5.5e-5**

### Accuracy w/ precision recovery
- **Absolute error:** ~**0.31**
- **Relative error:** ~**2.7e-6**

Demonstrates a significant accuracy improvement (approximately 95%) over non-corrected accumulation while maintaining almost identical throughput performance to the baseline GEMM, which are critical for HPC operations. 

---
## References
	•	https://arxiv.org/pdf/2203.03341
	•	NVIDIA CUDA Programming Guide
	•	CUTLASS documentation
