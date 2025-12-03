# State Management in Pseudo-Mamba

## Overview

Correct state management is critical for training recurrent models with PPO, especially for long-horizon tasks where BPTT must span the entire episode.

## Unified Interface

All controllers implement the `BaseController` interface:

### 1. Initialization
```python
state = controller.init_state(batch_size, device)
```
*   **GRU**: Returns a Tensor `[B, H]`.
*   **Mamba**: Returns a tuple `(conv_state, ssm_state)`.
*   **Pseudo-Mamba**: Returns a Tensor `[B, H]`.

### 2. Forward Step
```python
features, new_state = controller.forward_step(input, state)
```
*   **Input**: `[B, input_dim]`
*   **Features**: `[B, feature_dim]` (Hidden representation)
*   **New State**: Updated state object.

### 3. Masking (Reset)
```python
masked_state = controller.reset_mask(state, done_mask)
```
*   **Done Mask**: Boolean tensor `[B]`. True indicates the episode finished at the *previous* step.
*   **Behavior**: Zeros out the state for finished environments, effectively resetting them for the next episode.

## PPO Integration

The `PPO` engine in `pseudo_mamba.rlh` handles state management automatically:
1.  **Rollout**: Stores the initial state of the rollout chunk.
2.  **Update**: Re-runs the forward pass on the entire chunk (or minibatch), starting from the stored initial state.
3.  **BPTT**: Gradients flow through the re-computed graph, ensuring correct credit assignment.
