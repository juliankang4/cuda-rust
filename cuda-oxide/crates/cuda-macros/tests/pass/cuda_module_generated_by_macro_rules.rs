// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A macro that generates kernels expands to the whole #[cuda_module] module:
// the module macro then sees every kernel inline and generates its launcher.

macro_rules! kernel_module {
    ($module:ident { $($name:ident = $value:expr;)* }) => {
        #[cuda_macros::cuda_module]
        mod $module {
            $(
                #[cuda_macros::kernel]
                pub fn $name(value: u32) {
                    let _ = (value, $value);
                }
            )*
        }
    };
}

kernel_module!(generated {
    first = 1u32;
    second = 2u32;
});

fn launch(
    module: &generated::LoadedModule,
    stream: &cuda_core::CudaStream,
    config: cuda_core::simt::LaunchConfig,
) {
    unsafe {
        module.first(stream, config, 1).unwrap();
        module.second(stream, config, 2).unwrap();
    }
}

fn main() {
    let _ = launch;
}
