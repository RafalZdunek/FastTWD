# FastTWD: Fast Tensor Wheel Decomposition in MATLAB

FastTWD is a MATLAB implementation of a fast version of the proximal alternating minimization (PAM) algorithm for tensor wheel (TW) decomposition for multidimensional arrays. The package provides CPU and GPU solvers, benchmark scripts, and reproducible experiments for comparing the proposed FastTWD implementation with a reference TW baseline.

The software is intended for research on multilinear/tensor decompositions, tensor-network models, and scalable approximation of high-order tensors.

---

## Main features

- Fast CPU implementation of TW decomposition using matrix-free contraction-based updates.
- GPU-oriented implementation based on `gpuArray`, with optional precision and solver settings.
- PAM-based update scheme for TW ring factors and the TW core tensor.
- Avoids explicit construction of large least-squares design matrices used in the baseline implementation.
- Includes benchmark scripts for runtime, memory usage, reconstruction error, and iteration-count analysis.
- Includes synthetic benchmark generators for CP, Tucker, tensor ring, and TW tensors.
- Does not require MATLAB Tensor Toolbox for the included benchmark data generators.

---

## Tensor Wheel model

For an input tensor

```text
X in R^{I_1 x I_2 x ... x I_N},
```

FastTWD computes an approximation

```text
X = TW({G_n}_{n=1}^N; C),
```

where the TW representation consists of

```text
G_n in R^{R_n x I_n x L_n x R_{n+1}},    n = 1,...,N,
C   in R^{L_1 x L_2 x ... x L_N},
```

with the cyclic convention `R_{N+1} = R_1`. The parameters `R_n` are the outer TW ranks and `L_n` are the inner/core ranks.

The rank matrix used by the solvers is

```matlab
opts.R = [R_1 R_2 ... R_N;
          L_1 L_2 ... L_N];
```

---

## Repository structure

```text
FastTWD/
├── main.m
├── startup_fasttwd.m
├── src/
│   ├── fast_twd_cpu.m
│   ├── fast_twd_gpu.m
│   ├── cores_prod_single_tw.m
│   └── cores_prod_single_tw_gpu.m
├── experiments/
│   ├── run_quick_benchmark.m
│   ├── run_sweep_inner_rank.m
│   ├── run_sweep_outer_rank.m
│   ├── run_sweep_tensor_order.m
│   ├── run_sweep_tensor_size.m
│   └── utils/
│       ├── DataBenchmark.m
│       ├── cp_full_local.m
│       ├── tucker_full_local.m
│       └── outer_ring_prod.m
├── third_party/
│   └── Baseline_TW_TC/
│       ├── inc_TW_TC.m
│       ├── factor_dims.m
│       ├── initialization_M.m
│       ├── tensor_contraction.m
│       └── *.p
└── results/
    └── generated experiment outputs
```

### Core files

| File | Description |
|---|---|
| `main.m` | Entry point that configures the path and runs the quick benchmark. |
| `startup_fasttwd.m` | Adds the required project folders to the MATLAB path. |
| `src/fast_twd_cpu.m` | Main CPU FastTWD solver. |
| `src/fast_twd_gpu.m` | GPU/CPU FastTWD solver with GPU-specific options. |
| `src/cores_prod_single_tw.m` | Tensor reconstruction from TW factors and core. |
| `src/cores_prod_single_tw_gpu.m` | GPU-oriented reconstruction helper. |
| `experiments/run_quick_benchmark.m` | Compact benchmark comparing baseline TW, FastTWD CPU, and FastTWD GPU. |
| `experiments/run_sweep_inner_rank.m` | Monte Carlo sweep over inner/core rank `L`. |
| `experiments/run_sweep_outer_rank.m` | Monte Carlo sweep over outer rank `R`. |
| `experiments/run_sweep_tensor_order.m` | Monte Carlo sweep over tensor order `N`. |
| `experiments/run_sweep_tensor_size.m` | Monte Carlo sweep over tensor mode size `I`. |
| `experiments/utils/DataBenchmark.m` | Synthetic tensor generator used by benchmark scripts. |

---

## Requirements

### Required

- MATLAB with support for `tensorprod`, `exportgraphics`, `table`, `categorical`, and `groupsummary`.
- A standard MATLAB installation for CPU experiments.

### Optional

- Parallel Computing Toolbox for GPU execution.
- CUDA-compatible NVIDIA GPU for `fast_twd_gpu.m`.

The code was written for recent MATLAB releases:

```text
Tested with MATLAB R2025b on Windows 11 Pro.
```

---

## Installation

Clone or download the repository and start MATLAB in the repository root directory:

```matlab
cd FastTWD
startup_fasttwd(true);
```

The argument `true` also adds the baseline TW implementation located in `third_party/Baseline_TW_TC/`. To use only the FastTWD implementation, run

```matlab
startup_fasttwd(false);
```

---

## Quick start

Run the default quick benchmark:

```matlab
main
```

or equivalently:

```matlab
startup_fasttwd(true);
run(fullfile('experiments', 'run_quick_benchmark.m'));
```

The quick benchmark generates a synthetic TW tensor and compares:

1. baseline TW CPU implementation: `inc_TW_TC`,
2. FastTWD CPU: `fast_twd_cpu`,
3. FastTWD GPU: `fast_twd_gpu`, if a compatible GPU is available.

The output is printed as a compact table containing reconstruction error, runtime, sampled peak memory, and iteration count.

---

## Basic usage

### CPU solver

```matlab
startup_fasttwd(false);

% Generate a synthetic TW benchmark tensor.
F = DataBenchmark(4);

% TW ranks.
N = ndims(F);
R = 4 * ones(1, N);      % outer ranks
L = 4 * ones(1, N);      % inner/core ranks

% Solver options.
opts.R     = [R; L];
opts.maxit = 30;
opts.rho   = 1;
opts.tol   = 1e-6;

% Dense decomposition.
Omega = [];
[X, G, Core, Out] = fast_twd_cpu(F, Omega, opts);

% External relative reconstruction error.
RES = norm(F(:) - X(:)) / norm(F(:));
fprintf('Relative reconstruction error: %.3e\n', RES);
```

### GPU solver

```matlab
startup_fasttwd(false);

F = DataBenchmark(4);
N = ndims(F);
R = 4 * ones(1, N);
L = 4 * ones(1, N);

opts.R             = [R; L];
opts.maxit         = 30;
opts.rho           = 1;
opts.tol           = 1e-6;
opts.use_gpu       = true;
opts.gather_output = true;
opts.precision     = 'double';

Omega = [];
[Xg, Gg, Coreg, Outg] = fast_twd_gpu(F, Omega, opts);

RESg = norm(F(:) - Xg(:)) / norm(F(:));
fprintf('GPU relative reconstruction error: %.3e\n', RESg);
```

---

## Solver options

### Common options

| Option | Meaning | Typical value |
|---|---|---|
| `opts.R` | Required `2 x N` matrix of outer and inner TW ranks `[R; L]`. | problem-dependent |
| `opts.maxit` | Maximum number of PAM iterations. | `30`, `50`, `500` |
| `opts.rho` | Proximal regularization parameter. | `1`, `1e-3` |
| `opts.tol` | Stopping tolerance for internal relative change. | `1e-6`, `1e-8` |
| `opts.core_update_after` | Iteration after which scheduled core updates are enabled. | `3` |
| `opts.core_update_every` | Period of scheduled core updates. | `2` |

### CPU-specific option

| Option | Meaning |
|---|---|
| `opts.enforceOmega` | If `true` and `Omega` is nonempty, observed entries are projected back after each update: `X(Omega) = F(Omega)`. |

### GPU-specific options

| Option | Meaning |
|---|---|
| `opts.use_gpu` | Enables GPU execution when set to `true`. |
| `opts.gather_output` | Gathers `X`, `G`, and `Core` back to CPU before returning. |
| `opts.precision` | Numerical precision mode: `'double'`, `'single'`, or `'mixed'`. |
| `opts.enable_core` | Enables scheduled core-factor updates. |
| `opts.enable_symmetrize` | Symmetrizes Gram matrices before linear solves. |
| `opts.core_solve_on_gpu` | Solves the core linear system on GPU when possible. |
| `opts.check_every` | Period of convergence checks. |
| `opts.verbose_every` | Period of console progress messages. |

Important: in the current dense GPU implementation, `Omega` is accepted for interface compatibility but is ignored.

---

## Structure of `fast_twd_cpu`

The function `fast_twd_cpu` implements a proximal alternating minimization (PAM) solver for TW decomposition. The algorithm uses matrix-free tensor contractions to update the TW factors and the core tensor without explicitly forming large intermediate design matrices.

```text
fast_twd_cpu
├── Argument handling
│   └── Interprets the call syntax:
│       fast_twd_cpu(F, opts) or fast_twd_cpu(F, Omega, opts).
│       Sets the observed-entry constraint flag enforceOmega.
│
├── Options parsing
│   └── Reads algorithmic parameters:
│       tol, maxit, rho, max_R, core_update_after,
│       core_update_every, num_padarray.
│
├── Initialization
│   ├── factor_dims
│   │   └── Creates the dimensions of the fourth-order TW factors
│   │       G{n} of size [R_n, I_n, L_n, R_{n+1}]
│   │       from the tensor dimensions and current ranks.
│   │
│   ├── random G factors
│   │   └── Initializes all TW factors G{1},...,G{N}
│   │       with random values using the current initial ranks.
│   │
│   ├── random Core
│   │   └── Initializes the inner/core tensor Core of size
│   │       [L_1, ..., L_N].
│   │
│   └── initial residual using cores_prod_single_tw
│       └── Reconstructs the initial TW approximation
│           and computes the initial relative residual.
│
├── Main PAM loop
│   ├── Factor updates for G{1},...,G{N}
│   │   ├── unfold
│   │   │   └── Converts the current factor G{n} into matrix form
│   │   │       compatible with the local least-squares update.
│   │   │
│   │   ├── tw_factor_rhs_dense_exact
│   │   │   └── Computes the right-hand side for updating one TW factor.
│   │   │       It evaluates the equivalent of X_(n) Q_n^T by exact
│   │   │       circular tensor contractions, without explicitly constructing
│   │   │       the large environment matrix Q_n, also referred to as GCrest.
│   │   │
│   │   ├── tw_factor_gram_dense_exact
│   │   │   └── Computes the Gram matrix for the same factor update.
│   │   │       It evaluates Q_n Q_n^T by contracting double-layer
│   │   │       TW environments. The result is a symmetric
│   │   │       matrix used in the regularized normal equations.
│   │   │
│   │   ├── Cholesky/proximal solve
│   │   │   └── Solves the regularized local system using Cholesky
│   │   │       factorization when possible, with a small jitter or
│   │   │       direct solve as fallback.
│   │   │
│   │   └── fold
│   │       └── Converts the updated matrix representation back
│   │           to the tensor form of G{n}.
│   │
│   ├── Conditional core update
│   │   ├── tw_core_rhs_dense
│   │   │   └── Computes the right-hand side for the core update.
│   │   │       It contracts the current tensor X with all TW factors,
│   │   │       leaving only the inner/core indices open. The output has
│   │   │       the same size as Core, namely [L_1, ..., L_N].
│   │   │
│   │   ├── tw_core_gram
│   │   │   └── Computes the Gram matrix for the core update without
│   │   │       forming the full core-design matrix. It builds double-layer
│   │   │       transfer tensors from the TW factors and contracts the
│   │   │       outer-rank ring to obtain H = A A^T.
│   │   │
│   │   └── Cholesky/proximal solve
│   │       └── Solves the regularized core system
│   │           (H + rho I)c = b + rho c_prev,
│   │           then reshapes the solution vector back to Core.
│   │
│   ├── Tensor reconstruction/update
│   │   └── cores_prod_single_tw
│   │       └── Reconstructs the full tensor approximation Xhat
│   │           by contracting the Core tensor with all TW factors
│   │           around the closed TW ring.
│   │           The result has the same physical dimensions as F.
│   │
│   ├── Optional observed-entry projection
│   │   └── If enforceOmega is active, replaces the observed entries:
│   │       X(Omega) = F(Omega).
│   │       This preserves known entries in tensor-completion mode.
│   │
│   ├── Convergence test
│   │   └── Computes the relative step error
│   │       ||X - X_old|| / ||X_old|| and stops if it is below tol.
│   │
│   └── Adaptive rank increment
│       └── rank_inc_adaptive
│           └── Enlarges selected TW ranks by padding the factors G
│               and the Core tensor when the current approximation
│               stagnates and the maximum ranks have not yet been reached.
│
└── Output trimming
    └── Trims the recorded error history Out.RSE
        to the number of actually executed PAM iterations.
```

---

## Structure of `fast_twd_gpu`

The function `fast_twd_gpu` implements a GPU/CPU-capable proximal alternating minimization (PAM) solver for TW decomposition of an input tensor. It follows the same mathematical update logic as the fast CPU version, but adds device management, precision control, optional GPU execution, timing diagnostics, and GPU/CPU-safe generic tensor-contraction helpers.

The optional argument `Omega` is accepted for compatibility with tensor-completion-style interfaces, but in the current dense implementation it is ignored.

```text
fast_twd_gpu
├── Argument handling
│   └── Interprets the supported call syntax:
│       fast_twd_gpu(F, opts) or fast_twd_gpu(F, Omega, opts).
│       The argument Omega is accepted for interface compatibility,
│       but is ignored in the current dense implementation.
│
├── Options parsing
│   ├── parse_opts_gpu
│   │   └── Reads and assigns default values for the solver options:
│   │       tol, maxit, rho, core_update_after, core_update_every,
│   │       relax, precision, use_gpu, gather_output, enable_core,
│   │       enable_symmetrize, core_solve_on_gpu, check_every,
│   │       and verbose_every.
│   │
│   ├── get_opt
│   │   └── Small utility for reading an option field or assigning
│   │       a default value if the field is absent.
│   │
│   └── get_dtypes_from_precision
│       └── Converts the selected precision mode into MATLAB numeric
│           types. Supported modes are 'double', 'single', and 'mixed'.
│           In mixed mode, GPU working arrays are stored in single precision,
│           while selected CPU-side solves may use double precision.
│
├── Tensor-wheel dimensions
│   └── factor_dims
│       └── Creates the dimensions of the fourth-order TW factors
│           G{k} of size [R_k, I_k, L_k, R_{k+1}]
│           from the tensor dimensions and the rank matrix opts.R = [R; L].
│
├── Device and data-type preparation
│   ├── GPU availability check
│   │   └── If opts.use_gpu is true, the code checks whether a compatible GPU
│   │       and the Parallel Computing Toolbox are available.
│   │
│   ├── Xwork initialization
│   │   └── Casts the input tensor F to the selected precision and transfers
│   │       it to gpuArray when GPU execution is enabled.
│   │
│   └── rand_gpu_or_cpu
│       └── Creates random initial factors either on CPU or GPU,
│           depending on opts.use_gpu.
│
├── Initialization
│   ├── random Gwork factors
│   │   └── Initializes all TW ring factors
│   │       Gwork{1},...,Gwork{N} with sizes produced by factor_dims.
│   │
│   ├── random Cwork core
│   │   └── Initializes the inner/core tensor Cwork of size
│   │       [L_1, ..., L_N].
│   │
│   ├── diagnostic output structure
│   │   └── Preallocates Out.RSE, Out.did_core, Out.time_factor,
│   │       Out.time_core, Out.time_recon, Out.time_total,
│   │       Out.stop_iter, and Out.settings.
│   │
│   └── CPU copy of Core
│       └── Maintains Core_cpu and C_old_cpu for CPU reconstruction
│           and for detecting core-size changes.
│
├── Main PAM loop
│   ├── Factor updates for Gwork{1},...,Gwork{N}
│   │   ├── unfold_fast
│   │   │   └── Converts the current factor Gwork{n} into matrix form
│   │   │       compatible with the local least-squares subproblem.
│   │   │
│   │   ├── tw_factor_rhs_dense_exact_generic
│   │   │   └── Computes the exact right-hand side for updating one TW factor.
│   │   │       It evaluates the equivalent of X_(n) Q_n^T using labelled
│   │   │       tensor contractions. All physical modes except I_n and all
│   │   │       non-adjacent ring ranks are contracted with the remaining TW
│   │   │       factors, and the corresponding inner indices are contracted
│   │   │       with the Core tensor. The output is arranged as
│   │   │       [R_n, I_n, L_n, R_{n+1}], matching the unfold_fast convention.
│   │   │       The large environment matrix Q_n is never formed explicitly.
│   │   │
│   │   ├── tw_factor_gram_dense_exact_generic
│   │   │   └── Computes the exact Gram matrix for the same factor update.
│   │   │       It evaluates Q_n Q_n^T by building a double-layer tensor-wheel
│   │   │       environment from two copies of the factors and two copies of
│   │   │       the Core tensor. Physical indices are contracted locally,
│   │   │       outer-rank links are contracted around the ring, and the
│   │   │       remaining variables of G{n} and its primed copy are reshaped
│   │   │       into a symmetric J_n x J_n matrix.
│   │   │
│   │   ├── add_diag_inplace
│   │   │   └── Adds the proximal regularization term rho I directly
│   │   │       to the diagonal of the Gram matrix.
│   │   │
│   │   ├── optional Gram symmetrization
│   │   │   └── If opts.enable_symmetrize is true, replaces the Gram matrix
│   │   │       by 0.5 * (B + B^T) for numerical stability.
│   │   │
│   │   ├── solve_right_spd
│   │   │   └── Solves the regularized right-sided system TempA / TempB
│   │   │       using Cholesky factorization when possible. If Cholesky fails,
│   │   │       a small jitter is added to the diagonal; if necessary, the code
│   │   │       falls back to a direct solve.
│   │   │
│   │   ├── relaxation
│   │   │   └── If opts.relax is different from 1, blends the previous and
│   │   │       newly computed factor update.
│   │   │
│   │   └── fold_fast
│   │       └── Converts the updated matrix representation back to the
│   │           fourth-order tensor form of Gwork{n}.
│   │
│   ├── Conditional core update
│   │   ├── core-update decision
│   │   │   └── The core is updated only if opts.enable_core is true and
│   │   │       one of the following conditions holds:
│   │   │       k == 1, the core size has changed, or the scheduled update
│   │   │       condition based on core_update_after and core_update_every
│   │   │       is satisfied.
│   │   │
│   │   ├── tw_core_rhs_dense_exact_generic
│   │   │   └── Computes the exact right-hand side for the core update.
│   │   │       It contracts the current working tensor Xwork with all TW
│   │   │       factors over all physical indices and all circular outer-rank
│   │   │       links. The only remaining open indices are the inner/core
│   │   │       indices L_1,...,L_N, so the result has the same size as Cwork.
│   │   │
│   │   ├── tw_core_gram_dense_exact_generic
│   │   │   └── Computes the exact Gram matrix for the core update without
│   │   │       forming the full core-design matrix. For each TW factor, it
│   │   │       constructs a double-layer transfer tensor by contracting the
│   │   │       physical index between an unprimed and a primed copy. Then it
│   │   │       contracts the complete outer-rank ring and leaves only the
│   │   │       unprimed and primed inner indices. These are reshaped into
│   │   │       a prod(L) x prod(L) symmetric matrix H.
│   │   │
│   │   ├── proximal core system
│   │   │   └── Forms the regularized system
│   │   │       (H + rho I)c = b + rho c_prev,
│   │   │       where c = vec(Cwork).
│   │   │
│   │   ├── solve_left_spd
│   │   │   └── Solves the core linear system using Cholesky factorization
│   │   │       with jitter and direct-solve fallback.
│   │   │
│   │   ├── optional CPU fallback for the core solve
│   │   │   └── If opts.core_solve_on_gpu is false, the exact system is built
│   │   │       on the active device but gathered and solved on CPU, then moved
│   │   │       back to GPU if required.
│   │   │
│   │   ├── relaxation
│   │   │   └── Optionally blends the previous and newly computed core tensor.
│   │   │
│   │   └── Core_cpu update
│   │       └── Updates the CPU copy of the core tensor for reconstruction
│   │           and for later core-size comparisons.
│   │
│   ├── Reconstruction and Xwork update
│   │   ├── gather Gwork factors if needed
│   │   │   └── Creates a CPU cache of the current TW factors for dense
│   │   │       reconstruction.
│   │   │
│   │   ├── reconstruct_cpu_from_factors
│   │   │   └── Reconstructs the dense tensor approximation Xhat on CPU.
│   │   │       If the project helper cores_prod_single_tw is available on
│   │   │       the MATLAB path, it is used as the preferred reconstruction
│   │   │       routine. Otherwise, the implementation falls back to the
│   │   │       generic reconstruction helper cores_prod_single_tw_gpu.
│   │   │
│   │   ├── cast and GPU transfer
│   │   │   └── Casts Xhat to the selected precision and transfers it
│   │   │       to gpuArray when opts.use_gpu is true.
│   │   │
│   │   └── proximal/relaxed Xwork update
│   │       └── Updates the working tensor according to
│   │           Xwork = (Xhat + rho * Xwork_old) / (1 + rho).
│   │
│   ├── Convergence check and logging
│   │   ├── check_every
│   │   │   └── Controls how often the internal relative step error is computed.
│   │   │
│   │   ├── RSE computation
│   │   │   └── Computes
│   │   │       norm(Xwork - Xwork_old) / max(1e-12, norm(Xwork_old)).
│   │   │       The value is gathered to CPU when GPU execution is active.
│   │   │
│   │   ├── verbose_every
│   │   │   └── Controls how often progress information is printed.
│   │   │
│   │   └── stopping criterion
│   │       └── Stops the PAM loop when the internal RSE falls below tol.
│   │
│   └── Timing diagnostics
│       └── Records time spent in factor updates, core updates,
│           reconstruction, and the full iteration.
│
├── Finalization
│   ├── trimming diagnostic arrays
│   │   └── Trims Out.RSE, Out.did_core, Out.time_factor, Out.time_core,
│   │       Out.time_recon, and Out.time_total to the executed iterations.
│   │
│   └── output gathering
│       └── If opts.gather_output is true, gathers X, G, and Core to CPU.
│           Otherwise, GPU arrays are returned when opts.use_gpu is true.
│
└── Helper functions
    ├── Device / numerics helpers
    │   ├── rand_gpu_or_cpu
    │   ├── gather_if_needed
    │   ├── add_diag_inplace
    │   ├── solve_right_spd
    │   └── solve_left_spd
    │
    ├── Fast fold / unfold helpers
    │   ├── unfold_fast
    │   └── fold_fast
    │
    ├── Factor-update contraction helpers
    │   ├── tw_factor_rhs_dense_exact_generic
    │   └── tw_factor_gram_dense_exact_generic
    │
    ├── Core-update contraction helpers
    │   ├── tw_core_rhs_dense_exact_generic
    │   └── tw_core_gram_dense_exact_generic
    │
    ├── Labelled tensor-contraction helpers
    │   ├── labelled_contract_common
    │   ├── labelled_contract
    │   ├── normalize_tensor_to_labels
    │   ├── labelled_permute
    │   ├── label_positions
    │   ├── size_with_labels
    │   ├── expected_sizes_for_labels
    │   ├── reshape_to_label_shape
    │   ├── intersect_stable_labels
    │   ├── setdiff_stable
    │   └── symbolic label generators for I, L, R and their primed copies
    │
    ├── CPU reconstruction helper
    │   └── reconstruct_cpu_from_factors
    │
    └── Tensor-wheel dimension helper
        └── factor_dims
```

In summary, `fast_twd_gpu` performs the same main PAM steps as the CPU solver: factor updates, conditional core updates, dense reconstruction, proximal update, convergence checking, and final output preparation. The main difference is that the working arrays can be stored and updated on GPU, while the code also provides precision control, optional CPU fallback for the core solve, timing diagnostics, and generic labelled tensor-contraction routines for GPU/CPU-safe exact updates.

---


## Experiments

The `experiments/` directory contains scripts used to reproduce the main benchmark studies.

| Script | Purpose | Main sweep variable |
|---|---|---|
| `run_quick_benchmark.m` | Compact demonstration benchmark. | none |
| `run_sweep_inner_rank.m` | Scalability with respect to inner/core rank. | `L` |
| `run_sweep_outer_rank.m` | Scalability with respect to outer rank. | `R` |
| `run_sweep_tensor_order.m` | Scalability with respect to tensor order. | `N` |
| `run_sweep_tensor_size.m` | Scalability with respect to mode size. | `I` |

To run an experiment from the repository root, use for example:

```matlab
startup_fasttwd(true);
run(fullfile('experiments', 'run_sweep_outer_rank.m'));
```

Each sweep script contains user-editable controls near the top of the file, for example:

```matlab
MODE = 'runtime';      % 'runtime' | 'memory' | 'both' where supported
MEM_METHOD = 'sampled';% 'sampled' | 'proxy' | 'hybrid'
MC_runtime = 10;
MC_memory  = 10;
```

The benchmarked methods are:

```text
Baseline TW CPU     inc_TW_TC
FastTWD CPU         fast_twd_cpu
FastTWD GPU         fast_twd_gpu
```

---

## Generated outputs

Experiment outputs are written automatically to timestamped subdirectories inside `results/`. A typical output folder is

```text
results/sweep_outer_rank_memory_YYYYMMDD_HHMMSS/
├── TW_final_sweep_R_memory_YYYYMMDD_HHMMSS.mat
└── figures/
    ├── memory_vs_R.png
    ├── memory_vs_R.eps
    ├── residual_vs_R.png
    ├── residual_vs_R.eps
    ├── iterations_vs_R.png
    └── iterations_vs_R.eps
```

The `.mat` files usually contain:

- `resultsTable`: raw per-run results,
- `summaryTable`: grouped mean, standard deviation, median, and/or interquartile statistics,
- experiment settings,
- output folder paths.

The figures are saved in both PNG and EPS formats.

---

## Benchmark data

Synthetic tensors are generated by `experiments/utils/DataBenchmark.m`:

| `bench` value | Generated tensor type |
|---|---|
| `1` | CP-format tensor |
| `2` | Tucker-format tensor |
| `3` | Tensor Ring-format tensor |
| `4` | Tensor Wheel-format tensor |

For example:

```matlab
Y = DataBenchmark(4);  % synthetic Tensor Wheel tensor
```

The local helper functions `cp_full_local.m` and `tucker_full_local.m` replace Tensor Toolbox calls for CP and Tucker reconstruction in the benchmark generator.

---

## Notes on memory measurements

The experiment scripts support three memory-measurement modes:

```text
sampled    timer-based sampling of MATLAB/GPU memory
proxy      deterministic analytical proxy estimates
hybrid     combination of proxy and sampled measurements
```

CPU memory sampling relies on MATLAB memory information and can be platform-dependent. GPU memory is sampled from the active `gpuDevice` when a compatible GPU is available. Therefore, memory results should be interpreted as practical benchmark measurements rather than exact allocation traces.

---

## Third-party code

The folder

```text
third_party/Baseline_TW_TC/
```

contains a reference TW decomposition implementation used for comparison in the benchmark scripts. Some routines are distributed as MATLAB P-code files (`*.p`).

The `Baseline_TW_TC` code is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. Because this repository includes and uses that GPL-licensed baseline code, FastTWD is also intended to be released under a compatible GPL-3.0 license.

---

## Known limitations

- The current implementation targets dense tensors.
- The GPU solver currently ignores `Omega`; it is accepted only for interface compatibility.
- Runtime and memory results depend on MATLAB version, BLAS/LAPACK backend, GPU model, driver version, and operating system.
- The first Monte Carlo trial may include MATLAB JIT, cache, and GPU initialization effects; the sweep scripts preserve raw results and may exclude the first trial from selected summaries/plots.
- Very large tensors or high TW ranks may require substantial RAM or GPU memory.

---

## Citation

If you use this software in a scientific publication, please cite the associated SoftwareX paper:

```bibtex
@article{FastTWD2026,
  title   = {Fast Tensor Wheel Decomposition for Dense Multidimensional Data},
  author  = {Rafal Zdunek},
  journal = {SoftwareX},
  year    = {2026},
  doi     = {To be completed}
}
```

Please also cite the original TW baseline method if you use the comparative scripts based on `third_party/Baseline_TW_TC/`.

```bibtex
@inproceedings{wu2022tensor,
  title={Tensor Wheel Decomposition and Its Tensor Completion Application},
  author={Wu, Zhong-Cheng and Huang, Ting-Zhu and Wang, Yan-Fei and Jia, Xi-Le},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31267--31281},
  year={2022}
}
```

---

## License

This project is intended to be released under the **GNU General Public License v3.0 (GPL-3.0)**.

The repository should include a `LICENSE` file containing the full GPL-3.0 license text. The third-party baseline code in `third_party/Baseline_TW_TC/` is also distributed under GPL-3.0.

Recommended release files:

```text
LICENSE
CITATION.cff
README.md
```

---

## Contact

Maintainer: Rafał Zdunek  
Affiliation: Wrocław University of Science and Technology

For questions, bug reports, and reproducibility issues, please use the GitHub issue tracker.
