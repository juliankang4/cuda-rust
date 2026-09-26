// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A kernel that a `macro_rules!` invocation declares inside a #[cuda_module]
// expands after the module macro runs, so like an `include!`d kernel it gets
// no launcher: calling one is a compile error, never a launch of another entry.

use cuda_core::CudaStream;
use cuda_core::simt::LaunchConfig;
use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    pub fn root(value: u32) {
        let _ = value;
    }

    macro_rules! declare_kernel {
        ($name:ident) => {
            #[cuda_macros::kernel]
            pub fn $name(value: u32) {
                let _ = value;
            }
        };
    }

    declare_kernel!(from_macro);
}

fn undiscovered(module: &kernels::LoadedModule, stream: &CudaStream, config: LaunchConfig) {
    let _ = module.from_macro(stream, config, 1u32);
}

fn main() {}
