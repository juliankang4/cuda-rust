// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A standalone #[launch_contract] (no #[cuda_module]) has no launcher to
// evaluate its relations, but the constants they name are still resolved and
// typed at the kernel: a signed one is rejected by name.

const OFFSET: i32 = 4;

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n + OFFSET),
)]
pub fn shifted(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
    let _ = (n, input, &mut output);
}

fn main() {}
