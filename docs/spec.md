# `tensor` library spec

`tensor` (`github.com/pacolang/numerics/tensor`) is a dense, row-major
tensor type for Paco. Dimensions are part of the type; a `Dyn` dimension is
resolved at run time.

## `Tensor<T: Numeric, const D: int...>`

`T` is any `Numeric` type (the integer types, `float`/`f32`/`f64`, `bf16`,
`f16`; the FP8 types `f8e4m3`/`f8e5m2` are `Numeric` but see "FP8" below).
`D` is a list of dimensions, each either a static extent (`2`, `784`, ...)
or `Dyn`. A `Dyn` dimension's actual extent is stored per-value and
supplied at construction time; a static dimension is never stored, only
checked.

### Construction

- `Tensor::zeros() -> Self` — every element zero, every `Dyn` dimension `0`.
- `Tensor::zeros_with(dyn_dims: []i64) -> Result<Self, ShapeError>` — zeros,
  with one run-time extent per `Dyn` dimension, in declaration order.
- `Tensor::from_slice(data: []T, dyn_dims: []i64) -> Result<Self, ShapeError>`
  — takes ownership of a row-major buffer; its length must match the
  resulting shape.
- `Tensor::cast<S: Numeric>(source: &Tensor<S, D...>) -> Self` — converts
  every element with `as`.

Each of these returns `Err(ShapeError)` when the caller supplied the wrong
number of `Dyn` extents, or (`from_slice`) a buffer of the wrong length.

### Query and element access

- `rank(&self) -> i64`, `shape(&self) -> []i64` (full shape, static and
  `Dyn` dimensions together), `len(&self) -> i64` (element count),
  `dyn_dims(&self) -> &[]i64`.
- `data(&self) -> &[]T`, `data_mut(&mut self) -> &mut []T` — the row-major
  buffer.
- `at(&self, i: i64) -> T`, `set_at(&mut self, i: i64, value: T)` — flat
  index.
- `offset(&self, index: &[]i64) -> i64` — row-major offset of a multi-index.
- `get(&self, index: []i64) -> T`, `set(&mut self, index: []i64, value: T)`
  — multi-index access.

## Elementwise arithmetic (`T: Numeric + Add + Sub + Mul`)

Two forms of each operator, both element-by-element:

- `add`/`sub`/`mul(&self, other: &Self) -> Self` — the `+`/`-`/`*`
  operators (Paco dispatches `a + b` to a method literally named `add` on
  `a`'s type, structurally, no trait impl required). **Panics** if the
  shapes disagree at run time — the same trade this compiler makes for
  indexing: a bounds/shape violation is a programmer error, not a value to
  thread through every call site.
- `checked_add`/`checked_sub`/`checked_mul<const E: int...>(&self, other: &Tensor<T, E...>) -> Result<Self, ShapeError>`
  — the same computation, returning `Err(ShapeError)` on a shape mismatch
  instead of panicking. Use these when the operand shapes are not proved
  equal by the type (typically because they came from different `Dyn`
  bindings — the compiler statically requires the `checked_*` form in that
  case, since nothing proves the two `Dyn` extents equal at the call site;
  see `cannot prove these dimensions are equal` in that case, with a
  fix-it pointing here).

`ShapeError { axis: i64, expected: i64, found: i64 }` — `axis` is the
disagreeing dimension, or `-1` when the element count or number of `Dyn`
extents was wrong. `ShapeError::display(&self) -> string` renders it as a
one-line message; the panicking operator forms use it as their panic
message.

Also:

- `scale(&self, k: T) -> Self` — multiply every element by a scalar.
- `sum(&self) -> T` — sum of all elements.

## `Dyn`-storage operations

- `concat<const S: int...>(&self, other: &Tensor<T, Dyn, S...>) -> Result<Self, ShapeError>`
  (on `Tensor<T, Dyn, R...>`) — stacks two tensors along their leading
  `Dyn` axis; their other extents must agree at run time.

## Matrix multiply (`T: Numeric + Add + Mul`)

On `Tensor<T, M, K>`:

- `matmul<const N: int>(&self, other: &Tensor<T, K, N>) -> Tensor<T, M, N>`
  — `self` (M×K) times `other` (K×N); the inner dimension is proved equal
  by the type. **Panics** if `M`, `K` or `N` is `Dyn` and the run-time
  extents disagree — the same panicking/checked split as `add`/`sub`/`mul`
  above (changed from a `Result`-returning form so it can carry a
  `#[derivative]`, which needs a plain return type); use `checked_matmul`
  for the `Result` form.
- `checked_matmul<const J: int, const N: int>(&self, other: &Tensor<T, J, N>) -> Result<Tensor<T, M, N>, ShapeError>`
  — `self` (M×K) times `other` (J×N) where `K` and `J` are not proved equal
  either; both are compared at run time.

## `Dyn` semantics

A `Dyn` dimension's extent lives on the value (`dyn_dims`), not the type.
Two tensors of the same generic shape (e.g. both `Tensor<f32, Dyn, 3>`) can
disagree at run time if their `Dyn` extents came from different bindings.
The `checked_*` methods and `matmul`/`concat` compare full shapes
(`check_shape`) at run time and return `Err` on a mismatch; the plain
operator forms (`add`/`sub`/`mul`) panic instead, via `ShapeError::display`.

## `std::autodiff::Differentiable`

`Tensor<T, D...>` satisfies `Differentiable` with `Tangent = Self`:
`zero_tangent(&self) -> Self` (same shape, all zero) and
`move_by(&mut self, offset: &Self)` (elementwise `+=`).

`add`, `sub`, `mul`, `scale`, `sum` and `matmul` each carry a hand-written
`#[derivative(of = ...)]`, so `std::autodiff::grad` differentiates through
them without needing to unfold their loop bodies. `scale`'s gradient covers
only its `Tensor` operand — the scale factor `k: T` is a bare generic
parameter (`Type::Generic`, not `Type::Float`) at `scale`'s own generic
signature, and the compiler's `#[derivative]` check (`is_differentiable_type`,
`paco-types`) only ever treats a generic parameter as differentiable when a
concrete float type has already been substituted for it, which never
happens at a generic registration's own (unsubstituted) signature; `k`
therefore gets no gradient, by construction, regardless of what `T` turns
out to be at any call site.

**Known compiler gap (not this library's code):** task 3.2a fixed calling a
method on a `grad`-returned gradient for a generic differentiable struct
with a *single* `#[derivative(of = ...)]` registration, or with several
registrations used from the module that declares the struct. `Tensor` has
six registrations (`add`, `sub`, `mul`, `scale`, `sum`, `matmul`) and, being
a library, is always used from a different module than the one that
declares it — and that combination is still broken: calling *any* method on
a `grad`-returned `Tensor` gradient, including a plain struct-body one like
`at`, panics the compiler outright (`paco-driver/src/lowering.rs:141`, "no
declaration found for generic method `Tensor::at`") rather than failing to
type-check. `grad` itself runs and returns correctly (confirmed: printing
the scalar loss value works); only touching the *gradient*'s contents
panics. This reproduces independent of `Tensor`, in the simplest
single-input case, with any number of static or `Dyn` dimensions — see
task 3.2d in the extraction change's tasks.md, which this blocks, for the
minimal repro. Until that is fixed, `grad` over `Tensor` cannot be
exercised end to end (FD-checked or otherwise).

## FP8 has no arithmetic

`f8e4m3` and `f8e5m2` satisfy `Numeric` (so `Tensor<f8e4m3, ...>` is a
valid type — see `cast`) but not `Add`/`Sub`/`Mul`. This is a compiler-level
bound on those types, not something this library implements or restricts
further, but it means an FP8 tensor has none of the elementwise
arithmetic, `scale`, `sum` or `matmul` methods above: they simply do not
exist for `T = f8e4m3` or `T = f8e5m2`, and the method-not-found error names
the missing bound.

# `math` library spec

`math` (`github.com/pacolang/numerics/math`) builds two-dimensional and
tabular types on top of `tensor`, using it the way any two files in one
project use each other (plain `tensor::` paths, no separate module
dependency).

## `Matrix<T: Numeric, const R: int, const C: int>`

A dense row-major matrix whose dimensions are part of its type
(`Matrix<float, 2, 3>`), wrapping a `tensor::Tensor<T, R, C>`.

### Construction

- `Matrix::zeros() -> Self` — all zeros; a `Dyn` dimension gets extent `0`.
- `Matrix::with_shape(rows: i64, cols: i64) -> Result<Self, tensor::ShapeError>`
  — all zeros with the given extents, which must match every static
  dimension; they fill in the `Dyn` ones. `Err(ShapeError)` on a mismatch.
- `Matrix::identity() -> Self` (`T: Numeric + Add + Sub + Mul`) — the
  identity matrix (`1` on the diagonal, `0` elsewhere).

### Query and element access

- `rows(&self) -> i64`, `cols(&self) -> i64`.
- `get(&self, r: i64, c: i64) -> T`, `set(&mut self, r: i64, c: i64, value: T)`.
- `data(&self) -> &[]T`, `data_mut(&mut self) -> &mut []T` — the row-major
  buffer.
- `tensor(&self) -> &tensor::Tensor<T, R, C>` — the underlying tensor.
- `transpose(&self) -> Matrix<T, C, R>`.

### Arithmetic (`T: Numeric + Add + Sub + Mul`)

Unlike `Tensor`'s `add`/`sub`/`mul`/`matmul`, `Matrix` has not yet had the
checked/panicking split applied (out of scope for this extraction — see the
extraction change's tasks.md task 3.3): `add`/`sub`/`mul` each still return
`Result`, delegating directly to `checked_add`/`checked_sub`/`checked_mul`,
and `a + b`/`a - b`/`a * b` (Paco dispatches an operator to the
identically-named method, structurally) each therefore evaluate to a
`Result` too, not a bare `Matrix`.

- `add`/`sub(&self, other: &Self) -> Result<Self, tensor::ShapeError>` —
  element-wise, `Err(ShapeError)` on a shape mismatch.
- `checked_add`/`checked_sub<const R2: int, const C2: int>(&self, other: &Matrix<T, R2, C2>) -> Result<Self, tensor::ShapeError>`
  — currently identical to `add`/`sub` (kept as separate names so a future
  panicking split does not have to invent them later).
- `mul<const N: int>(&self, other: &Matrix<T, C, N>) -> Result<Matrix<T, R, N>, tensor::ShapeError>`
  — `self` (R×C) times `other` (C×N); the inner dimension is proved equal
  by the type.
- `checked_mul<const J: int, const N: int>(&self, other: &Matrix<T, J, N>) -> Result<Matrix<T, R, N>, tensor::ShapeError>`
  — `self` (R×C) times `other` (J×N) where `C` and `J` are not proved
  equal; compared at run time.
- `scale(&self, k: T) -> Self` — multiply every element by a scalar.

## `DataFrame<S>`

A growable column of rows of any type `S`.

- `DataFrame::new() -> Self`.
- `push(&mut self, row: S)`.
- `len(&self) -> i64`.
- `row(&self, i: i64) -> Option<S>` — `None` when `i` is out of range.

## `Schema` and `#[derive(Schema)]`

`#[derivable(derive_schema)] pub trait Schema` lets any struct opt in with
`#[derive(Schema)]`. For each field `f: T` of the struct, `derive_schema`
generates a column accessor `S::f(frame: &math::DataFrame<S>) -> Vec<T>`
that reads that field out of every row, in order — a comptime-generated
column projection over a `DataFrame` of that row type.
