# Benchmark Suite

The Pseudo-Mamba benchmark suite consists of 8 vectorized memory tasks designed to stress-test long-horizon capabilities.

## Tasks

### 1. Delayed Cue (`delayed_cue`)
*   **Goal**: Remember a cue presented at t=0 after a long delay.
*   **Difficulty**: Linear with delay length.
*   **Metric**: Accuracy at query step.

### 2. Copy Memory (`copy_memory`)
*   **Goal**: Reproduce a sequence of tokens after a delay.
*   **Difficulty**: Depends on sequence length and delay.
*   **Metric**: Token-level accuracy.

### 3. Associative Recall (`assoc_recall`)
*   **Goal**: Given a key, recall the associated value from a previously presented pair.
*   **Difficulty**: Number of pairs and delay.
*   **Metric**: Accuracy.

### 4. N-Back (`n_back`)
*   **Goal**: Output the token presented N steps ago.
*   **Difficulty**: N.
*   **Metric**: Accuracy.

### 5. Multi-Cue Delay (`multi_cue_delay`)
*   **Goal**: Remember multiple cues presented at random times.
*   **Difficulty**: Number of cues and temporal sparsity.
*   **Metric**: Accuracy.

### 6. Permuted Copy (`permuted_copy`)
*   **Goal**: Reproduce a sequence in a permuted order.
*   **Difficulty**: Requires random access memory.
*   **Metric**: Token-level accuracy.

### 7. Pattern Binding (`pattern_binding`)
*   **Goal**: Bind two sequences (A -> B) and recall B given A.
*   **Difficulty**: Sequence length and delay.
*   **Metric**: Token-level accuracy.

### 8. Distractor Navigation (`distractor_nav`)
*   **Goal**: Navigate to a target location shown only at t=0, ignoring distractors.
*   **Difficulty**: Horizon length and distractor intensity.
*   **Metric**: Final distance to target.
