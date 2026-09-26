// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// When every kernel of a #[cuda_module] comes from a macro invocation, the
// module macro sees none of them; its error names the invocation.

use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
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

fn main() {}
