// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A `requires` relation may name unsigned integer constants: in upper case
// (`TILE`, a const generic `N`) or by path (`super::shapes::ROWS`,
// `P::WIDTH`). The launchers of a #[cuda_module] evaluate them, and a
// standalone kernel still has them resolved and typed next to it.

const TILE: usize = 36;

mod shapes {
    pub const ROWS: u32 = 8;
}

pub trait Policy {
    const WIDTH: u64;
}

enum Wide {}

impl Policy for Wide {
    const WIDTH: u64 = 4;
}

#[cuda_macros::cuda_module]
mod kernels {
    use super::*;

    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n * TILE, output.len() >= n * super::shapes::ROWS),
    )]
    pub fn tiled(n: u32, input: &[f32], mut output: cuda_device::DisjointSlice<f32>) {
        let _ = (n, input, &mut output);
    }

    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (input.len() >= n * P::WIDTH),
    )]
    pub fn policy<P: Policy>(n: u32, input: &[f32]) {
        let _ = (n, input);
    }

    #[cuda_macros::kernel]
    #[cuda_macros::launch_contract(domain = 1, block = (64, 1, 1), requires = (input.len() >= N))]
    pub fn chunked<const N: usize>(input: &[f32]) {
        let _ = input;
    }
}

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n * TILE),
)]
pub fn standalone(n: u32, input: &[f32]) {
    let _ = (n, input);
}

#[cuda_macros::kernel]
#[cuda_macros::launch_contract(
    domain = 1,
    block = (64, 1, 1),
    requires = (input.len() >= n * P::WIDTH),
)]
pub fn standalone_policy<P: Policy>(n: u32, input: &[f32]) {
    let _ = (n, input);
}

fn launchers(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    input: &cuda_core::DeviceBuffer<f32>,
    output: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), cuda_core::LaunchContractError> {
    let tiled = module.prepare_tiled(cuda_core::LaunchConfig1D::new(1, 64, 0))?;
    module.tiled(stream, &tiled, 1, input, output)?;
    let policy = module.prepare_policy::<Wide>(cuda_core::LaunchConfig1D::new(1, 64, 0))?;
    module.policy::<Wide>(stream, &policy, 1, input)?;
    let chunked = module.prepare_chunked::<4>(cuda_core::LaunchConfig1D::new(1, 64, 0))?;
    module.chunked::<4>(stream, &chunked, input)
}

fn main() {
    let _ = launchers;
}
