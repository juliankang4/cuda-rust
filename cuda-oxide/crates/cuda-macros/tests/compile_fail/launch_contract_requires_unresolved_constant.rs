// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// An upper-case name in `requires` is a constant, resolved where the checks
// are generated: a misspelt one is a compile error at the relation, not a
// check against some other value.

#[cuda_macros::cuda_module]
mod kernels {
    const TILE: usize = 36;

    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n * TIEL),
    )]
    pub fn tiled(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, input, &mut output);
    }
}

fn main() {}
