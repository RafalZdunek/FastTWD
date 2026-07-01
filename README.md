# FastTWD: Fast Tensor Wheel Decomposition in MATLAB

FastTWD is a MATLAB package implementing a matrix-free proximal alternating minimization (PAM) algorithm for tensor wheel (TW) decomposition. It provides CPU and GPU-capable solvers together with reproducible benchmarks comparing FastTWD with a bundled reference TW implementation.

The package is intended for research on tensor decompositions, tensor-network models, and scalable approximation of high-order tensors.

---

## Main features

- Matrix-free updates of TW ring factors and the central core tensor.
- CPU solver with optional observed-entry projection and adaptive rank growth.
- GPU-capable solver based on `gpuArray`, with configurable precision and numerical options.
- Quick benchmark and Monte Carlo scalability experiments for runtime, memory, reconstruction error, and iteration count.
- Synthetic CP, Tucker, tensor ring, and tensor wheel test tensors.
- No MATLAB Tensor Toolbox dependency.

---

## Tensor Wheel model

For an input tensor

```text
X in R^{I_1 x I_2 x ... x I_N},
```

FastTWD computes an approximation

```text
X_hat = TW({G_n}_{n=1}^N; C),
```

where

```text
G_n in R^{R_n x I_n x L_n x R_{n+1}},    n = 1,...,N,
C   in R^{L_1 x L_2 x ... x L_N},
```

with the cyclic convention `R_{N+1} = R_1`. The parameters `R_n` and `L_n` are the outer and inner TW ranks, respectively.

The rank matrix used by both solvers is

```matlab
opts.R = [R_1 R_2 ... R_N;
          L_1 L_2 ... L_N];
```

For the CPU solver, `opts.R` specifies the maximum ranks. For the GPU solver, it specifies fixed ranks.

---

## Repository structure

```text
FastTWD/
├── LICENSE
├── README.md
├── main.m                         quick-benchmark entry point
├── startup_fasttwd.m              MATLAB path configuration
├── src/
│   ├── fast_twd_cpu.m             CPU FastTWD solver
│   ├── fast_twd_gpu.m             GPU/CPU-capable FastTWD solver
│   ├── cores_prod_single_tw.m     TW tensor reconstruction
│   └── tensor_contraction_explicit.m
├── experiments/
│   ├── run_quick_benchmark.m
│   ├── run_sweep_inner_rank.m
│   ├── run_sweep_outer_rank.m
│   ├── run_sweep_tensor_order.m
│   ├── run_sweep_tensor_size.m
│   └── utils/                     synthetic-data generators
├── third_party/
│   └── Baseline_TW_TC/            bundled reference implementation
└── results/                        created by the sweep scripts
```

---

## Requirements

### Required

- MATLAB R2025b.

The current release was developed and tested with **MATLAB R2025b on Windows 11 Pro**. Earlier MATLAB releases have not been systematically tested.

### Optional

- Parallel Computing Toolbox for GPU execution.
- A CUDA-compatible NVIDIA GPU for `fast_twd_gpu.m` with `opts.use_gpu = true`.

The MATLAB `memory` function used for host-memory sampling is available only on Windows. On other operating systems, CPU memory results may be unavailable, while the solvers can still run.

---

## Installation

Clone or download the repository and start MATLAB in its root directory:

```matlab
cd FastTWD
startup_fasttwd(true);
```

The argument `true` adds the bundled baseline implementation from `third_party/Baseline_TW_TC/`. To use only FastTWD, run

```matlab
startup_fasttwd(false);
```

---

## Quick start

Run the default benchmark from the repository root:

```matlab
main
```

This is equivalent to

```matlab
startup_fasttwd(true);
run(fullfile('experiments', 'run_quick_benchmark.m'));
```

The default setting `bench = 4` generates a synthetic TW tensor and compares:

1. the bundled baseline TW CPU solver, `inc_TW_TC`;
2. the FastTWD CPU solver, `fast_twd_cpu`;
3. the FastTWD GPU solver, `fast_twd_gpu`, when a compatible GPU is available.

The console table reports relative reconstruction error, execution time, sampled memory, and iteration count. Timing and memory are measured in separate solver runs, so memory polling does not affect the reported execution time.

In the default benchmark, both FastTWD solvers start with `R_n = L_n = 4`. The bundled baseline starts from ranks 2 and may increase them adaptively up to 4.

The synthetic data type can be changed in `run_quick_benchmark.m`:

| `bench` | Tensor type |
|---:|---|
| `1` | CP |
| `2` | Tucker |
| `3` | Tensor Ring |
| `4` | Tensor Wheel |

---

## Basic usage

### CPU solver

```matlab
startup_fasttwd(false);

X = DataBenchmark(4);
N = 4;
R = 4 * ones(1, N);
L = 4 * ones(1, N);

opts.R            = [R; L];
opts.maxit        = 30;
opts.rho          = 1;
opts.tol          = 1e-6;
opts.seed         = 0;
opts.num_padarray = 0;

Omega = [];
[X_hat, G, Core, Out] = fast_twd_cpu(X, Omega, opts);

RSE = norm(X(:) - X_hat(:)) / norm(X(:));
fprintf('Relative reconstruction error: %.3e\n', RSE);
```

The shorter call syntax is also supported:

```matlab
[X_hat, G, Core, Out] = fast_twd_cpu(X, opts);
```

### GPU solver

```matlab
startup_fasttwd(false);

X = DataBenchmark(4);
N = 4;
R = 4 * ones(1, N);
L = 4 * ones(1, N);

opts.R             = [R; L];
opts.maxit         = 30;
opts.rho           = 1;
opts.tol           = 1e-6;
opts.seed          = 0;
opts.use_gpu       = true;
opts.gather_output = true;
opts.precision     = 'double';

Omega = [];
[X_hat, G, Core, Out] = fast_twd_gpu(X, Omega, opts);

RSE = norm(X(:) - X_hat(:)) / norm(X(:));
fprintf('GPU relative reconstruction error: %.3e\n', RSE);
```

Set `opts.use_gpu = false` to execute the GPU-solver implementation on the CPU.

---

## Solver options

### Common options

| Option | Meaning | Default |
|---|---|---:|
| `opts.R` | Required `2 x N` rank matrix `[R; L]`. Maximum ranks for the CPU solver; fixed ranks for the GPU solver. | required |
| `opts.seed` | Nonnegative integer seed for reproducible initialization. | `0` |
| `opts.maxit` | Maximum number of PAM iterations. | `500` |
| `opts.rho` | Proximal regularization parameter. | `1e-3` |
| `opts.tol` | Stopping tolerance for the internal relative change. | `1e-6` |
| `opts.core_update_after` | Threshold after which scheduled core updates are enabled. | `3` |
| `opts.core_update_every` | Period of scheduled core updates. | `2` |

### CPU-specific options

| Option | Meaning | Default |
|---|---|---:|
| `opts.num_padarray` | Reduces the initial ranks below `opts.R`; adaptive growth can subsequently increase them up to the prescribed maxima. | `0` |
| `opts.enforceOmega` | Restores observed entries after each update when `Omega` is nonempty. | `~isempty(Omega)` |

`Omega` can be empty, a logical mask with the same logical size as `X`, or a vector of valid linear indices.

### GPU-specific options

| Option | Meaning | Default |
|---|---|---:|
| `opts.use_gpu` | Execute supported computations on a GPU. | `true` |
| `opts.gather_output` | Return `X_hat`, `G`, and `Core` as CPU arrays. | `true` |
| `opts.precision` | Working precision: `'double'`, `'single'`, or `'mixed'`. | `'double'` |
| `opts.relax` | Relaxation parameter for factor and core updates. | `1` |
| `opts.enable_core` | Enable core updates. | `true` |
| `opts.enable_symmetrize` | Symmetrize Gram matrices before linear solves. | `true` |
| `opts.core_solve_on_gpu` | Solve the core linear system on the GPU when GPU execution is active. | `true` |
| `opts.check_every` | Convergence-check period. | `1` |
| `opts.verbose_every` | Console-output period. | `20` |

The current GPU solver accepts `Omega` only for interface compatibility and ignores it. It uses fixed ranks and does not implement adaptive rank growth.

---

## Algorithm overview

Both solvers apply PAM updates to the TW ring factors and core tensor. For each local subproblem, FastTWD computes the right-hand side and Gram matrix by exact tensor-network contractions rather than explicitly constructing the large least-squares design matrix used by the reference implementation. The regularized systems are solved with Cholesky factorization when possible, with numerical fallbacks when necessary.

The CPU solver additionally supports observed-entry projection and adaptive rank growth. The GPU solver provides device management, configurable precision, optional CPU execution, and timing diagnostics. In the current implementation, dense reconstruction inside `fast_twd_gpu` is performed on the CPU after gathering the factors and core.

`Out.RSE` is the internal relative change between consecutive iterates and is used as the stopping criterion. It should not be confused with the external reconstruction error

```matlab
norm(X(:) - X_hat(:)) / norm(X(:)).
```

The returned diagnostic structure includes at least:

- `Out.RSE` — internal relative-change history;
- `Out.RES_init` — initial relative residual;
- `Out.iterations` — number of executed iterations;
- `Out.converged` — convergence flag;
- `Out.final_R` — final rank matrix.

The GPU solver additionally records per-iteration timing information and core-update indicators.

---

## Experiments

The `experiments/` directory contains the scripts used for the benchmark studies:

| Script | Sweep variable |
|---|---|
| `run_quick_benchmark.m` | none |
| `run_sweep_inner_rank.m` | inner rank `L` |
| `run_sweep_outer_rank.m` | outer rank `R` |
| `run_sweep_tensor_order.m` | tensor order `N` |
| `run_sweep_tensor_size.m` | mode size `I` |

Run a sweep from the repository root, for example:

```matlab
startup_fasttwd(true);
run(fullfile('experiments', 'run_sweep_outer_rank.m'));
```

Each sweep script contains user-editable settings near its beginning, including the tested values, Monte Carlo counts, execution mode, and memory-measurement method. Depending on the script, the relevant controls include

```matlab
MODE = 'runtime';       % 'runtime' | 'memory' | 'both', where supported
MEM_METHOD = 'sampled'; % 'sampled' | 'proxy' | 'hybrid'
MC_runtime = 10;
MC_memory  = 10;
```

The scripts compare the baseline CPU, FastTWD CPU, and FastTWD GPU methods. If no compatible GPU is available, the GPU runs are skipped.

### Generated outputs

Sweep results are saved in timestamped subdirectories of `results/`, for example

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

The MAT-files contain raw per-run results (`resultsTable`), grouped statistics (`summaryTable`), experiment settings, and output paths. The sweep scripts preserve raw trials; selected scripts exclude the first Monte Carlo trial from summaries and plots to reduce initialization effects.

---

## Memory measurements

### Quick benchmark

Runtime and memory are measured in separate runs. Memory is sampled every `0.02` s and represents an approximate total footprint rather than an exact allocation peak.

- CPU methods report sampled MATLAB host memory when available.
- The GPU method reports sampled GPU-device memory.
- Host-memory and device-memory values represent different resources and are not directly comparable.

### Sweep scripts

The sweep scripts support up to three memory modes:

- `sampled` — timer-based sampling of MATLAB or GPU memory;
- `proxy` — deterministic analytical estimates;
- `hybrid` — a script-dependent combination of sampled and proxy values.

Sampled sweep results are generally expressed as the largest positive increase relative to the level measured immediately before the solver call. Memory plots use median values across the included Monte Carlo trials.

For compact reporting, the code labels binary conversions as MB and GB:

```text
1 MB = 2^20 bytes
1 GB = 2^30 bytes
```

---

## Third-party code

`third_party/Baseline_TW_TC/` contains the reference TW implementation used in the comparative benchmarks. Some routines are distributed as MATLAB P-code files (`*.p`).

The bundled baseline and this repository are distributed under the GNU General Public License v3.0 (GPL-3.0).

---

## Known limitations

- `fast_twd_gpu` targets dense decomposition and ignores `Omega`.
- Adaptive rank growth is implemented only in `fast_twd_cpu`.
- `fast_twd_cpu` requires a nonempty real double-precision input array.
- `fast_twd_gpu` accepts nonempty real single- or double-precision CPU arrays and transfers them internally when GPU execution is enabled.
- The current GPU solver reconstructs the dense tensor on the CPU during each iteration.
- Host-memory sampling through `memory` is Windows-specific.
- Runtime and memory results depend on the MATLAB release, numerical libraries, hardware, drivers, and operating system.
- Large tensors or high TW ranks may require substantial RAM or GPU memory.

---

## Citation

If you use FastTWD in a scientific publication, please cite the associated SoftwareX paper:

```bibtex
@article{FastTWD2026,
  title   = {FastTWD: A MATLAB Package for Matrix-Free CPU/GPU Tensor Wheel Decomposition},
  author  = {Rafa{\l} Zdunek},
  journal = {SoftwareX},
  year    = {2026},
  doi     = {To be completed}
}
```

Please also cite the original TW method when using the comparative scripts based on `third_party/Baseline_TW_TC/`:

```bibtex
@inproceedings{wu2022tensor,
  title     = {Tensor Wheel Decomposition and Its Tensor Completion Application},
  author    = {Wu, Zhong-Cheng and Huang, Ting-Zhu and Wang, Yan-Fei and Jia, Xi-Le},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {31267--31281},
  year      = {2022}
}
```

---

## License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. The complete license text is provided in `LICENSE`.

---

## Contact

Maintainer: Rafał Zdunek  
Affiliation: Wrocław University of Science and Technology

For questions, bug reports, and reproducibility issues, please use the GitHub issue tracker.
