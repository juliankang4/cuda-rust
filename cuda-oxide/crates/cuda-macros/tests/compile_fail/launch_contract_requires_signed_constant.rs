// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `requires` relations are evaluated in u64, so a constant they name must be
// an unsigned integer, as a scalar parameter must; a signed one is rejected
// by name at the relation.

const OFFSET: i32 = 4;

#[cuda_macros::cuda_module]
mod kernels {
    use super::*;

    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n + OFFSET),
    )]
    pub fn shifted(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, input, &mut output);
    }
}

fn main() {}
