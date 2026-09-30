# Enum niche payload leaf slots

Reproducer for the rms_norm/rope local-depot defect: an enum whose niche
carrier shares its bytes with an aggregate payload used to leave the
payload's non-pointer leaves in `[N x i8]` filler. Construction spilled the
value byte by byte, payload reads loaded it back as one wide value, and LLVM
SROA cannot reassemble mixed-type slices — so the spill slot survived
`opt -O2` as a `.local` depot (measured: rms_norm 16 B, rope 104 B in
bloomery's kernels).

The fix (`build_enum_slot_map` leaf decomposition) gives every payload leaf
its own typed slot at its rustc byte offset, and construct/payload reads
rebuild the aggregate in SSA. Memory layout, size, alignment, and the niche
encoding are unchanged — only the SSA shape changes.

## The three kernels

| kernel | shape | depot before | depot after |
|---|---|---|---|
| `smooth_triplets` | `Option<(f32, f32, &mut f32)>` driven by `while let` (the rms_norm `next_cell` shape; floats share filler with the pointer niche) | yes | **none** |
| `lease_borrow` | `Option<Pair>`, `Pair { src: &mut Stepper, base: usize }` where `Stepper` is kernel-local (the rope `own` shape) | yes | **yes — the rule's witness** |
| `lease_raw` | same lease, `Pair { src: *mut Stepper, base: usize, _borrow: PhantomData }` | yes | **none** |

`lease_borrow` keeps its depot on purpose: the payload pointer is a
kernel-local alloca's address flowing through the `None`/`Some` merge. That
is a property of the borrowed code shape (a returned struct holding `&mut`
to a kernel-local struct), not an enum-lowering defect, and no enum slot
rule can undo it. `lease_raw` shows the same lease with the pointer copied:
identical enum bytes `{ptr, usize}`, no reference crossing the merge, and no
depot after the fix.

## PASS/FAIL criterion

```bash
cargo oxide build enum_niche_payload --verbose
```

- FAIL (parent `c76f1e17`): the build prints a local-memory warning for all
  three kernels, and each entry's PTX contains a `.local` depot.
- PASS (with the fix): no local-memory warning for `smooth_triplets` and
  `lease_raw`; their PTX entries contain no `.local`. `lease_borrow` keeps
  its warning and depot.

Machine-checked form (after `cargo oxide build enum_niche_payload`):

```bash
crates/rustc-codegen-cuda/examples/enum_niche_payload/target/release/enum_niche_payload --verify-ptx
```

checks the three PTX entries' `.local` presence against the table above.

The value checks (`cargo oxide run enum_niche_payload`) must pass both
before and after the fix: the change is layout-neutral, so a wrong result on
either side is a miscompile, not a codegen-shape difference.
