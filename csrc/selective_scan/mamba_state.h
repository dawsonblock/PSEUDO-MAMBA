/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Enhanced with explicit state management for RL/streaming inference.
 ******************************************************************************/

#pragma once

#include <torch/extension.h>

/**
 * @brief Encapsulates Mamba layer state for manual persistence/restoration.
 * 
 * This struct enables:
 * - Saving/loading checkpoints mid-generation
 * - Cross-device state transfer
 * - Explicit state management in RL rollouts
 * 
 * State tensors:
 * - conv_state: [batch, d_inner, d_conv-1] - Causal conv sliding window
 * - ssm_state:  [batch, d_inner, d_state] - SSM recurrent hidden state
 */
struct MambaState {
    at::Tensor conv_state;  // [B, D, d_conv-1]
    at::Tensor ssm_state;   // [B, D, d_state]
    
    /**
     * @brief Move state tensors to specified device/dtype
     */
    MambaState to(c10::optional<at::Device> device = c10::nullopt,
                  c10::optional<at::ScalarType> dtype = c10::nullopt) const {
        return MambaState{
            conv_state.to(device, dtype.has_value() ? dtype.value() : conv_state.scalar_type()),
            ssm_state.to(device, dtype.has_value() ? dtype.value() : ssm_state.scalar_type())
        };
    }
};
