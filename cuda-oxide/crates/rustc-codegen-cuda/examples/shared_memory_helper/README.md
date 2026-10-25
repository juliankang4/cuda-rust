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

The dependency variant imports available MIR across crates; it does not test
an opaque, separately linked device binary. The runtime checks one-thread
barrier behavior, not synchronization between multiple threads or TMA traffic.
