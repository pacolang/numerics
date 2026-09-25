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

- `matmul<const N: int>(&self, other: &Tensor<T, K, N>) -> Result<Tensor<T, M, N>, ShapeError>`
  — `self` (M×K) times `other` (K×N); the inner dimension is proved equal
  by the type, so this only fails when `M`, `K` or `N` is `Dyn` and the
  run-time extents disagree.
- `checked_matmul<const J: int, const N: int>(&self, other: &Tensor<T, J, N>) -> Result<Tensor<T, M, N>, ShapeError>`
  — `self` (M×K) times `other` (J×N) where `K` and `J` are not proved equal
  either; both are compared at run time.

`matmul`/`checked_matmul` are not part of the operator-overload pair above:
they take an extra output dimension (`N`) and already return a `Result` in
both forms, so there is no separate panicking variant.

## `Dyn` semantics

A `Dyn` dimension's extent lives on the value (`dyn_dims`), not the type.
Two tensors of the same generic shape (e.g. both `Tensor<f32, Dyn, 3>`) can
disagree at run time if their `Dyn` extents came from different bindings.
The `checked_*` methods and `matmul`/`concat` compare full shapes
(`check_shape`) at run time and return `Err` on a mismatch; the plain
operator forms (`add`/`sub`/`mul`) panic instead, via `ShapeError::display`.

## FP8 has no arithmetic

`f8e4m3` and `f8e5m2` satisfy `Numeric` (so `Tensor<f8e4m3, ...>` is a
valid type — see `cast`) but not `Add`/`Sub`/`Mul`. This is a compiler-level
bound on those types, not something this library implements or restricts
further, but it means an FP8 tensor has none of the elementwise
arithmetic, `scale`, `sum` or `matmul` methods above: they simply do not
exist for `T = f8e4m3` or `T = f8e5m2`, and the method-not-found error names
the missing bound.
