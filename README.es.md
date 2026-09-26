<p align="center">
  <a href="https://github.com/pacolang/numerics"><img alt="License" src="https://img.shields.io/github/license/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/issues"><img alt="Issues" src="https://img.shields.io/github/issues/pacolang/numerics"></a>
</p>

<h1 align="center">Numerics</h1>

<p align="center">
  Tensores densos, formas verificadas en tiempo de compilación, y los tipos <code>Matrix</code>/<code>DataFrame</code> construidos sobre ellos, para <a href="https://github.com/pacolang/paco">Paco</a>.
  <br />
  <a href="docs/spec.md"><strong>Ver la especificación »</strong></a>
  <br />
  <br />
  <a href="https://github.com/pacolang/paco">Ver el Compilador</a>
  ·
  <a href="https://github.com/pacolang/numerics/issues/new">Reportar un Error</a>
  ·
  <a href="https://github.com/pacolang/rfcs">Proponer una RFC</a>
</p>

**Leer en:** [English](README.md) · [Português](README.pt-BR.md) · **Español**

> **Estado:** dos módulos importables, `tensor` (arreglos densos N-dimensionales con formas verificadas en tiempo de compilación) y `math` (`Matrix`, `DataFrame`, construidos sobre `tensor`). Ningún binding de BLAS vive en este repositorio — ver [Ecosistema](#ecosistema).

## Índice

- [Acerca de](#acerca-de)
- [Ecosistema](#ecosistema)
- [Primeros Pasos](#primeros-pasos)
- [Uso](#uso)
- [Hoja de Ruta](#hoja-de-ruta)
- [Contribuir](#contribuir)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Acerca de

`numerics` es la biblioteca numérica de Paco: `tensor::Tensor<T: Numeric, const D: int...>`,
un tensor denso, row-major, cuyo rank y dimensiones forman parte de su
tipo — cada dimensión es una extensión estática o `Dyn`, resuelta en tiempo
de ejecución — con aritmética elemento a elemento (formas con panic y
formas `checked_*`), `matmul`, `concat` a lo largo de un eje `Dyn`
principal, y una implementación de `std::autodiff::Differentiable` con
derivadas escritas a mano para `add`/`sub`/`mul`/`scale`/`sum`/`matmul`.
`math::Matrix<T: Numeric, const R: int, const C: int>` envuelve un
`Tensor<T, R, C>` con las operaciones 2D habituales (`transpose`,
`identity`, `add`/`sub`/`mul`), y `math::DataFrame<S>` es una columna
creciente de filas de cualquier tipo `S`, con `#[derive(Schema)]` generando
un accesor de columna tipado por campo. Ver [`docs/spec.md`](docs/spec.md)
para la API completa.

El nombre es histórico: `numerics` es de donde se extrajo este
repositorio — `std::numerics` y `std::math` (`Tensor`, `Matrix`,
`DataFrame`) — antes de que existieran, como repositorios separados, tanto
la eventual división `tensor`/`math` como un binding `std::blas` para el
BLAS del sistema. Ni `math` ni un binding de BLAS forman parte del árbol de
módulos de este repositorio hoy de la forma que el nombre sugiere: `math`
es uno de los dos módulos que este repositorio ya exporta, junto a
`tensor`, y ningún módulo `blas` existe aquí todavía. La
[RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md)
registra por qué las bibliotecas de dominio viven fuera del `stdlib` de
`pacolang/paco` — y deja explícitamente la forma interna de este
repositorio, incluyendo si `tensor` y `math` eventualmente se separan aún
más, como una decisión futura y separada.

## Ecosistema

Una organización, `github.com/pacolang`, un repositorio por proyecto.
[`pacolang/paco`](https://github.com/pacolang/paco) es el núcleo:
compilador, runtime y `stdlib`, una sola versión. `pacolang/numerics` es
una biblioteca oficial junto a él, versionada de forma independiente,
obtenida con `paco get github.com/pacolang/numerics@<versión>` de la misma
forma que cualquier dependencia. Las decisiones de diseño del lenguaje y de
su ecosistema se registran como RFCs en
[`pacolang/rfcs`](https://github.com/pacolang/rfcs), nunca como ADRs.

## Primeros Pasos

### Requisitos previos

- Un toolchain de `paco` — ver [`pacolang/paco`](https://github.com/pacolang/paco)
  para compilar o instalar uno.
- Un `paco.mod` en tu proyecto (`paco mod init` si todavía no tienes uno).

### Instalación

```bash
paco get github.com/pacolang/numerics@<versión>
paco mod tidy
```

## Uso

```paco
use tensor;
use math;

fn main() {
    // Un tensor de forma estática: 2x3, ambas dimensiones conocidas en
    // tiempo de compilación.
    let a = tensor::Tensor<f32, 2, 3>::zeros();
    let b = tensor::Tensor<f32, 2, 3>::zeros();
    let c = a.add(&b); // falla (panic) si las formas no coinciden; usa checked_add si no

    // Multiplicación de matrices: la dimensión interna es probada igual por el tipo.
    let m = math::Matrix<f32, 2, 3>::zeros();
    let n = math::Matrix<f32, 3, 4>::zeros();
    let product = m.mul(&n); // Result<Matrix<f32, 2, 4>, tensor::ShapeError>
}
```

Ver [`docs/spec.md`](docs/spec.md) para dimensiones `Dyn`, operadores
`checked_*`, `Differentiable`/`grad`, y `DataFrame`/`Schema`.

## Hoja de Ruta

- `grad` sobre `Tensor` todavía no se puede ejercitar de extremo a extremo —
  una brecha del compilador en las llamadas a métodos sobre un gradiente
  devuelto por `grad`, para un tipo genérico con múltiples registros de
  derivada usado fuera del módulo que lo declara. Registrado en
  [`docs/spec.md`](docs/spec.md), en la sección `std::autodiff::Differentiable`; no es un error de esta biblioteca.
- `add`/`sub`/`mul` de `Matrix` todavía devuelven `Result` en lugar de haber
  recibido la misma división panic/`checked_*` que ya tiene `Tensor`.
- Si `tensor` y `math` alguna vez se separan en repositorios distintos es
  una pregunta abierta, dejada para una RFC futura — ver la
  [RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md).

Ver los [issues](https://github.com/pacolang/numerics/issues) y los
[milestones](https://github.com/pacolang/numerics/milestones) de este
repositorio para el seguimiento del día a día.

## Contribuir

Las contribuciones son lo que hace de la comunidad open-source un lugar
increíble para aprender y crear. Cualquier contribución tuya es **muy
bienvenida**.

1. Haz un fork del repositorio.
2. Crea tu rama de feature (`git checkout -b feat/mi-feature`).
3. Ejecuta `paco test` antes de abrir un pull request.
4. Haz commit de tus cambios y abre un pull request.

Para un cambio de diseño del lenguaje o del ecosistema, abre una RFC en
[`pacolang/rfcs`](https://github.com/pacolang/rfcs) primero.

## Licencia

Distribuido bajo la Apache License, Version 2.0. Ver [`LICENSE`](LICENSE)
para más información.

## Contacto

Enlace del proyecto: [https://github.com/pacolang/numerics](https://github.com/pacolang/numerics)
