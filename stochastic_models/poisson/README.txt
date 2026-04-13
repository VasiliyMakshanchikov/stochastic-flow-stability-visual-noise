# Poisson Process Modeling for Event Streams

## Overview

In this project, we model the flow of detected objects (e.g., people in video frames) as a **Poisson process**. This provides a simple yet powerful baseline for analyzing how noise in computer vision (CV) systems affects the estimation of stochastic process parameters.

The pipeline is structured as follows:
Video → CV Model → Event Extraction → Poisson Model → Parameter Estimation

The core idea is to compare the estimated parameters **before and after noise injection**, in order to evaluate robustness.

---

## Definition of Event

An **event** is defined as:

> The appearance of a new object (person) in the frame.

After running the CV model, we obtain a sequence of event timestamps:
t = [t₁, t₂, t₃, ..., tₙ]


where:
- `tᵢ` is the time (or frame index converted to time) when the i-th object appears.

---

## Poisson Process: Concept

A Poisson process is a stochastic process that models **random, independent events occurring over time** with a constant average rate.

Key assumptions:
- Events occur **independently**
- The process is **stationary** (constant rate)
- No memory (past events do not affect future ones)

---

## Mathematical Formulation

### 1. Event Count Distribution

The number of events in a time interval of length `t` follows a Poisson distribution:

\[
P(N(t)=k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}
\]

where:
- `N(t)` — number of events in time interval `t`
- `λ` — intensity (average rate of events)

---

### 2. Interarrival Times (Key Representation)

More importantly for practical implementation:

\[
\Delta t_i = t_{i+1} - t_i \sim \text{Exponential}(\lambda)
\]

This means:
- Time intervals between consecutive events follow an **exponential distribution**
- This property is used for parameter estimation

---

## Input to the Model

The model takes as input:
event_times = [t₁, t₂, ..., tₙ]
or equivalently:
nterarrival_times = [Δt₁, Δt₂, ..., Δtₙ₋₁]
These are obtained from the CV detection pipeline.

---

## Output of the Model

The Poisson process is fully described by a single parameter:

### Intensity (λ)

\[
\lambda = \frac{1}{\mathbb{E}[\Delta t]}
\]

In practice:
λ = number_of_events / total_time

---

## Experimental Setup

The goal of the experiment is to evaluate how noise affects the estimated parameter `λ`.

### 1. Baseline (No Noise)
Original Video
↓
CV Model
↓
event_times_true
↓
λ_true
---

### 2. With Noise
Noisy Video
↓
CV Model
↓
event_times_noisy
↓
λ_noisy

---

## Evaluation Metric

The main metric is the deviation in estimated intensity:

\[
\Delta \lambda = |\lambda_{noisy} - \lambda_{true}|
\]

Interpretation:
- Small Δλ → model is robust to noise
- Large Δλ → strong sensitivity to noise

---

## Effect of Noise

Different types of noise affect the process differently:

### 1. False Negatives (Missed Detections)
- Events are removed
- Interarrival times increase
- λ is **underestimated**

---

### 2. False Positives (Spurious Detections)
- Extra events are added
- Interarrival times decrease
- λ is **overestimated**

---

### 3. Image Noise (e.g., Gaussian, Poisson)
- Affects detection quality indirectly
- Leads to both false positives and false negatives

---

## Model Validation

To verify whether the Poisson model is appropriate, the following checks should be performed:

1. **Exponentiality of interarrival times**
2. **Independence of events**
3. **Stationarity of intensity**

Typical tools:
- Histogram of interarrival times
- Kolmogorov–Smirnov test
- Autocorrelation analysis

---

## Summary

| Component        | Description                          |
|------------------|--------------------------------------|
| Event            | Appearance of a new object           |
| Input            | Event timestamps                     |
| Model            | Poisson process                      |
| Parameter        | λ (event intensity)                  |
| Baseline Output  | λ_true                               |
| Noisy Output     | λ_noisy                              |
| Goal             | Analyze robustness to noise          |

---

## Key Insight

This setup allows us to study:

> How errors in computer vision detection propagate into stochastic process parameter estimation.

This is essential for understanding the reliability of real-world systems that rely on noisy visual data.