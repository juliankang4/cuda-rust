# Dynamic shared memory passed to a device helper

This example checks the raw-pointer pattern:

```rust
let smem = DynamicSharedArray::<u8, 1024>::get_raw();
helper(smem, offset);

unsafe fn helper(smem: *mut u8, offset: usize) -> u64 {
    let full = smem.add(1024).cast::<Barrier>();
    // Initialize, arrive at, and wait on the shared-memory barrier.
    // Read and update a separate shared-memory data slot.
}
```

The no-feature build also runs the same probe through `SharedPtr`, a
CTA-shared pointer typed as such (the feature builds fail MIR verification
before any probe runs):

```rust
let smem = DynamicSharedArray::<u32, 1024>::shared_ptr();
shared_ptr_probe(Region { base: smem, offset });

#[inline(never)]
unsafe fn shared_ptr_probe<T>(region: Region<T>) -> u64 {
    let full = region.base.cast::<u8>().byte_add(1024).cast::<Barrier>().as_ptr();
    // ...
}
```

`SharedPtr<T>` lowers to an `addrspace(3)` pointer by its type, so the
generic helper's parameter, nested in a struct, is one too, and the call
needs no address-space adaptation.

Two more shapes run after that probe, in `shared_ptr_shapes_probe`: a struct
with two `SharedPtr<f32>` fields passed by value to the non-inlined
`write_pair`, which writes through both, and a `SharedPtr<f32>` returned from
the non-inlined `select_slot` and used as a store target. Each of 64 lanes
writes its own slots, then overwrites one slot of its partner lane through the
returned pointer, so the final values prove both stores landed.

`verify-code-shape.sh` checks the LLVM IR and PTX the build leaves next to
this file:

| Helper | Required code shape |
| --- | --- |
| `shared_ptr_probe` | first parameter is `ptr addrspace(3)`, and its call site passes one |
| `write_pair` | both `Pair` fields arrive as `ptr addrspace(3)` parameters |
| `select_slot` | return type is `ptr addrspace(3)` |
| `shared_ptr_probe` in PTX | a `.func` that the kernel calls once, with `mbarrier.init.shared` inside its body |
| whole module | no `cvta.to.shared` and no `isspacep` in the PTX |

The PTX half reads the unoptimized build. `scripts/smoketest.sh` lists this
example in `NO_OPT_SHAPE_EXAMPLES`, so after the optimized build decides the
verdict it rebuilds once with `CUDA_OXIDE_NO_OPT=1` and runs the script on
those artifacts. To run the script by hand, build the same way first:

```bash
CUDA_OXIDE_NO_OPT=1 cargo oxide build shared_memory_helper
bash crates/rustc-codegen-cuda/examples/shared_memory_helper/verify-code-shape.sh
```

Without the surviving `.func`, a module-wide search for `cvta.to.shared`
proves little: an optimized build inlines the helper, and the generic-pointer
control compiles to the same instructions. The call into the `.func` is where
a generic pointer would have needed a conversion.

The launch supplies 2048 dynamic shared bytes. Four independent CTAs each use
one thread. Nine launches cover three data offsets and three input values,
including wrapping arithmetic. Every CTA checks both a volatile shared-memory
data round trip and a barrier transition from pending to complete. Barrier
polling is bounded so a broken barrier reports failure instead of hanging.

The operation body is a macro shared by three variants, so only the function
boundary changes:

| Features | Placement |
| --- | --- |
| None | Directly in the kernel; control case |
| `local-helper` | Helper in the kernel's crate |
| `cross-crate-helper` | Helper in the `helper-lib` dependency crate |

Helpers use `#[inline(never)]` by default. Add `force-inline` to request
`#[inline(always)]`, or `heuristic-inline` to omit an inline attribute. These
source attributes alone are not evidence that a call survives or disappears
in generated GPU code.

## Reproduce

Run from the repository root with the repository's Rust and CUDA toolchains.
The barrier probe requires SM90 or newer; the commands below target an RTX
5090 (`sm_120a`).

Build and run the direct control first. `set -e` prevents execution of an old
binary if this build fails.

```bash
set -e
cargo oxide build shared_memory_helper --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  crates/rustc-codegen-cuda/examples/shared_memory_helper/target/release/shared_memory_helper
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  crates/rustc-codegen-cuda/examples/shared_memory_helper/target/release/shared_memory_helper
```

Build each helper variant separately, then run its binary with the same
sanitizer commands. Stop on a failed build to avoid running a stale binary.

```bash
cargo oxide build shared_memory_helper \
  --arch sm_120a --features local-helper
cargo oxide build shared_memory_helper \
  --arch sm_120a --features cross-crate-helper
```

Repeat with `--features local-helper,force-inline`,
`cross-crate-helper,force-inline`, `local-helper,heuristic-inline`, or
`cross-crate-helper,heuristic-inline` to compare inlining policies.
Set `CUDA_OXIDE_DUMP_MIR=1` to inspect the imported helper definition and call.

## Expected results

The direct control and both helper variants must build and report a `PASS`
marker after checking the data and barrier results. The `examples-compile`
workflow builds both helper variants on `main` and `bloomery` with their
default `#[inline(never)]` policy.

The imported argument has address space 3 (shared memory), while the helper
parameter has address space 0 (generic memory). The importer adapts that
address-space difference before call verification. Pointee, mutability and
pointer kind must still match.

With no features, the `SharedPtr` probe runs after the direct control and
reports `PASS shared-ptr: 9 launches, 36 CTA data/barrier checks`, and the
shapes probe reports `PASS shared-ptr-shapes: 3 launches, 384 slot checks`,
on the LLVM NVPTX route and with `--emit-nvvm-ir --arch sm_120`.

The dependency variant imports available MIR across crates; it does not test
an opaque, separately linked device binary. The runtime checks one-thread
barrier behavior, not synchronization between multiple threads or TMA traffic.
