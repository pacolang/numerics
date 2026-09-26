<p align="center">
  <a href="https://github.com/pacolang/numerics"><img alt="License" src="https://img.shields.io/github/license/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/issues"><img alt="Issues" src="https://img.shields.io/github/issues/pacolang/numerics"></a>
</p>

<h1 align="center">Numerics</h1>

<p align="center">
  Dense tensors, compile-time shapes, and the <code>Matrix</code>/<code>DataFrame</code> types built on top of them, for <a href="https://github.com/pacolang/paco">Paco</a>.
  <br />
  <a href="docs/spec.md"><strong>Explore the spec »</strong></a>
  <br />
  <br />
  <a href="https://github.com/pacolang/paco">View the Compiler</a>
  ·
  <a href="https://github.com/pacolang/numerics/issues/new">Report a Bug</a>
  ·
  <a href="https://github.com/pacolang/rfcs">Propose an RFC</a>
</p>

**Read this in:** **English** · [Português](README.pt-BR.md) · [Español](README.es.md)

> **Status:** two importable modules, `tensor` (dense N-dimensional arrays with compile-time shapes) and `math` (`Matrix`, `DataFrame`, built on `tensor`). No BLAS binding lives in this repository — see [Ecosystem](#ecosystem).

## Table of Contents

- [About](#about)
- [Ecosystem](#ecosystem)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

`numerics` is Paco's numerics library: `tensor::Tensor<T: Numeric, const D: int...>`,
a dense, row-major tensor whose rank and dimensions are part of its type —
each dimension is either a static extent or `Dyn`, resolved at run time —
with elementwise arithmetic (panicking and `checked_*` forms), `matmul`,
`concat` along a leading `Dyn` axis, and a `std::autodiff::Differentiable`
implementation with hand-written derivatives for `add`/`sub`/`mul`/`scale`/
`sum`/`matmul`. `math::Matrix<T: Numeric, const R: int, const C: int>` wraps
a `Tensor<T, R, C>` with the usual 2D operations (`transpose`, `identity`,
`add`/`sub`/`mul`), and `math::DataFrame<S>` is a growable column of rows of
any type `S`, with `#[derive(Schema)]` generating a typed column accessor
per field. See [`docs/spec.md`](docs/spec.md) for the full API.

The name is historical: `numerics` is what this repository was extracted
from — `std::numerics` and `std::math` (`Tensor`, `Matrix`, `DataFrame`) —
before either the eventual `tensor`/`math` split, or a `std::blas` binding
to the system BLAS, existed as separate repositories. Neither `math` nor a
BLAS binding is part of this repository's own module tree today: `math` is
one of the two modules this repository already exports, side by side with
`tensor`, and no `blas` module exists here at all yet. [RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md)
records why domain libraries live outside `pacolang/paco`'s `stdlib` — and
explicitly leaves this repository's own internal shape, including whether
`tensor` and `math` eventually split further, as a later, separate
decision.

## Ecosystem

One organization, `github.com/pacolang`, one repository per project.
[`pacolang/paco`](https://github.com/pacolang/paco) is the core: compiler,
runtime and `stdlib`, one version. `pacolang/numerics` is an official
library next to it, versioned independently, fetched with
`paco get github.com/pacolang/numerics@<version>` the way any dependency
is. Design decisions for the language and its ecosystem are recorded as
RFCs in [`pacolang/rfcs`](https://github.com/pacolang/rfcs), never as ADRs.

## Getting Started

### Prerequisites

- A `paco` toolchain — see [`pacolang/paco`](https://github.com/pacolang/paco)
  for building or installing one.
- A `paco.mod` in your project (`paco mod init` if you don't have one yet).

### Installation

```bash
paco get github.com/pacolang/numerics@<version>
paco mod tidy
```

## Usage

```paco
use tensor;
use math;

fn main() {
    // A static-shape tensor: 2x3, both dimensions known at compile time.
    let a = tensor::Tensor<f32, 2, 3>::zeros();
    let b = tensor::Tensor<f32, 2, 3>::zeros();
    let c = a.add(&b); // panics on a shape mismatch; use checked_add otherwise

    // Matrix multiply: the inner dimension is proved equal by the type.
    let m = math::Matrix<f32, 2, 3>::zeros();
    let n = math::Matrix<f32, 3, 4>::zeros();
    let product = m.mul(&n); // Result<Matrix<f32, 2, 4>, tensor::ShapeError>
}
```

See [`docs/spec.md`](docs/spec.md) for `Dyn` dimensions, `checked_*`
operators, `Differentiable`/`grad`, and `DataFrame`/`Schema`.

## Roadmap

- `grad` over `Tensor` cannot yet be exercised end to end — a compiler gap
  in method calls on a `grad`-returned gradient for a multi-registration
  generic type used from outside its declaring module. Tracked in
  [`docs/spec.md`](docs/spec.md), under `std::autodiff::Differentiable`; not this library's own bug.
- `Matrix`'s `add`/`sub`/`mul` still return `Result` rather than having had
  `Tensor`'s panicking/`checked_*` split applied to them.
- Whether `tensor` and `math` ever split into separate repositories is an
  open question left to a later RFC — see [RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md).

See this repository's [issues](https://github.com/pacolang/numerics/issues)
and [milestones](https://github.com/pacolang/numerics/milestones) for
day-to-day tracking.

## Contributing

Contributions are what make the open-source community such an amazing
place to learn and create. Any contribution you make is **greatly
appreciated**.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feat/my-feature`).
3. Run `paco test` before opening a pull request.
4. Commit your changes and open a pull request.

For a language or ecosystem design change, open an RFC in
[`pacolang/rfcs`](https://github.com/pacolang/rfcs) first.

## License

Distributed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE)
for more information.

## Contact

Project Link: [https://github.com/pacolang/numerics](https://github.com/pacolang/numerics)
