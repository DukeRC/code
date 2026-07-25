#!/usr/bin/env Rscript

# Benchmark BLAS/LAPACK multithreaded speedup for common dense linear-algebra
# routines in R: matrix multiply, cross-product, Cholesky, and Cholesky solve.
#
# For each thread count in the sweep, every routine is timed `reps` times and
# the median elapsed time is recorded. Speedup is reported relative to the
# single-threaded (smallest thread count) baseline.
#
# Usage: Rscript benchmark_R_blas.R [n] [reps] [threads]
#   n        matrix dimension          (default 6000)
#   reps     timed repetitions/routine (default 3)
#   threads  comma-separated list      (default "1,2,4,8,16,32", capped at cores)
#
# Requires the RhpcBLASctl package to set thread counts from within R.

suppressPackageStartupMessages({
  have_blasctl <- requireNamespace("RhpcBLASctl", quietly = TRUE)
  if (have_blasctl) library(RhpcBLASctl)
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
n    <- if (length(args) >= 1) as.integer(args[1]) else 6000L
reps <- if (length(args) >= 2) as.integer(args[2]) else 3L

# Respect a Slurm allocation if present, else fall back to detected cores.
ncores <- {
  slurm <- Sys.getenv(c("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"))
  slurm <- as.integer(slurm[slurm != ""])
  if (length(slurm)) min(slurm) else parallel::detectCores()
}
if (length(args) >= 3) {
  threads <- as.integer(strsplit(args[3], ",")[[1]])
} else {
  threads <- c(1L, 2L, 4L, 8L, 16L, 32L)
}
threads <- sort(unique(threads[threads >= 1 & threads <= ncores]))

if (!have_blasctl && length(threads) > 1) {
  cat("Note: RhpcBLASctl not installed; cannot switch threads from R.\n",
      "      Running a single point at the backend's current thread count.\n\n", sep = "")
  threads <- NA_integer_  # single pass, don't touch thread counts
}

set.seed(123)

# Routines to benchmark: label -> quoted expression evaluated against A, B, K, b.
routines <- list(
  matmul     = list(label = "DGEMM: A %*% B",              expr = quote(A %*% B)),
  crossprod  = list(label = "DSYRK: crossprod(A)",         expr = quote(crossprod(A))),
  chol       = list(label = "Cholesky: chol(K)",           expr = quote(chol(K))),
  solve_chol = list(label = "Cholesky solve: solve(K, b)", expr = quote(solve(K, b)))
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Print the R build, its BLAS/LAPACK backend, and the active thread counts.
report_environment <- function() {
  cat("R version:  ", R.version.string, "\n", sep = "")
  cat("BLAS:       ", sessionInfo()$BLAS,   "\n", sep = "")
  cat("LAPACK:     ", sessionInfo()$LAPACK, "\n", sep = "")
  cat("Cores detected: ", ncores, "\n", sep = "")

  cat("\nEnvironment thread variables:\n")
  cat("  OMP_NUM_THREADS      =", Sys.getenv("OMP_NUM_THREADS",      "<unset>"), "\n")
  cat("  OPENBLAS_NUM_THREADS =", Sys.getenv("OPENBLAS_NUM_THREADS", "<unset>"), "\n\n")
}

# Set BLAS and OpenMP thread counts (no-op if RhpcBLASctl is unavailable).
set_threads <- function(t) {
  if (have_blasctl && !is.na(t)) {
    blas_set_num_threads(t)
    omp_set_num_threads(t)
  }
}

# Median elapsed time (seconds) for `expr` over `reps` reps, after one warm-up.
time_routine <- function(expr, reps, env) {
  invisible(eval(expr, env))                 # warm-up, untimed
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    gc()
    times[i] <- system.time(eval(expr, env))["elapsed"]
  }
  median(times)
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

report_environment()

cat(sprintf("Matrix size n = %d, reps = %d (%.2f GiB per dense matrix)\n",
            n, reps, n * n * 8 / 1024^3))
cat("Thread sweep: ",
    if (all(is.na(threads))) "backend default" else paste(threads, collapse = ", "),
    "\n\n", sep = "")

cat("Generating test matrices...\n")
A <- matrix(rnorm(n * n), n, n)
B <- matrix(rnorm(n * n), n, n)
b <- rnorm(n)

# Symmetric positive-definite matrix for Cholesky-based routines.
K <- crossprod(A)
diag(K) <- diag(K) + 1
cat("SPD matrix constructed (isSymmetric = ", isSymmetric(K), ")\n\n", sep = "")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

# median_sec[routine, thread_count] filled below.
median_sec <- matrix(
  NA_real_, nrow = length(routines), ncol = length(threads),
  dimnames = list(names(routines), as.character(threads))
)

for (t in threads) {
  set_threads(t)
  tag <- if (is.na(t)) "backend-default threads" else sprintf("%d thread(s)", t)
  cat(sprintf("== %s ==\n", tag))

  for (r in names(routines)) {
    med <- time_routine(routines[[r]]$expr, reps, environment())
    median_sec[r, as.character(t)] <- med
    cat(sprintf("  %-28s %8.3f sec\n", routines[[r]]$label, med))
  }
  cat("\n")
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

cat("Median elapsed time (seconds):\n")
print(round(median_sec, 4))

if (ncol(median_sec) > 1) {
  baseline <- median_sec[, 1]                    # smallest thread count
  speedup  <- baseline / median_sec              # per-routine speedup vs baseline
  cat(sprintf("\nSpeedup relative to %s thread(s):\n", colnames(median_sec)[1]))
  print(round(speedup, 2))

  efficiency <- sweep(speedup, 2, threads, "/")  # speedup / thread count
  cat("\nParallel efficiency (speedup / threads):\n")
  print(round(efficiency, 2))
}

cat("\nDone.\n")
