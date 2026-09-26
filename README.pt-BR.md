<p align="center">
  <a href="https://github.com/pacolang/numerics"><img alt="License" src="https://img.shields.io/github/license/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/pacolang/numerics"></a>
  <a href="https://github.com/pacolang/numerics/issues"><img alt="Issues" src="https://img.shields.io/github/issues/pacolang/numerics"></a>
</p>

<h1 align="center">Numerics</h1>

<p align="center">
  Tensores densos, formas verificadas em tempo de compilação, e os tipos <code>Matrix</code>/<code>DataFrame</code> construídos sobre eles, para o <a href="https://github.com/pacolang/paco">Paco</a>.
  <br />
  <a href="docs/spec.md"><strong>Veja a especificação »</strong></a>
  <br />
  <br />
  <a href="https://github.com/pacolang/paco">Ver o Compilador</a>
  ·
  <a href="https://github.com/pacolang/numerics/issues/new">Reportar um Bug</a>
  ·
  <a href="https://github.com/pacolang/rfcs">Propor uma RFC</a>
</p>

**Leia em:** [English](README.md) · **Português** · [Español](README.es.md)

> **Status:** dois módulos importáveis, `tensor` (arrays densos N-dimensionais com formas verificadas em tempo de compilação) e `math` (`Matrix`, `DataFrame`, construídos sobre `tensor`). Nenhum binding de BLAS vive neste repositório — veja [Ecossistema](#ecossistema).

## Sumário

- [Sobre](#sobre)
- [Ecossistema](#ecossistema)
- [Primeiros Passos](#primeiros-passos)
- [Uso](#uso)
- [Roteiro](#roteiro)
- [Contribuindo](#contribuindo)
- [Licença](#licença)
- [Contato](#contato)

## Sobre

`numerics` é a biblioteca de numérica do Paco: `tensor::Tensor<T: Numeric, const D: int...>`,
um tensor denso, row-major, cujo rank e dimensões fazem parte do seu tipo —
cada dimensão é uma extensão estática ou `Dyn`, resolvida em tempo de
execução — com aritmética elemento a elemento (formas com panic e formas
`checked_*`), `matmul`, `concat` ao longo de um eixo `Dyn` líder, e uma
implementação de `std::autodiff::Differentiable` com derivadas escritas à
mão para `add`/`sub`/`mul`/`scale`/`sum`/`matmul`.
`math::Matrix<T: Numeric, const R: int, const C: int>` envolve um
`Tensor<T, R, C>` com as operações 2D usuais (`transpose`, `identity`,
`add`/`sub`/`mul`), e `math::DataFrame<S>` é uma coluna crescente de linhas
de qualquer tipo `S`, com `#[derive(Schema)]` gerando um acessor de coluna
tipado por campo. Veja [`docs/spec.md`](docs/spec.md) para a API completa.

O nome é histórico: `numerics` é de onde este repositório foi extraído —
`std::numerics` e `std::math` (`Tensor`, `Matrix`, `DataFrame`) — antes de
existirem, como repositórios separados, tanto a eventual divisão
`tensor`/`math` quanto um binding `std::blas` para o BLAS do sistema. Nem
`math` nem um binding de BLAS fazem parte da árvore de módulos deste
repositório hoje da forma que o nome sugere: `math` é um dos dois módulos
que este repositório já exporta, lado a lado com `tensor`, e nenhum módulo
`blas` existe aqui ainda. A [RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md)
registra por que as bibliotecas de domínio vivem fora do `stdlib` de
`pacolang/paco` — e deixa explicitamente a forma interna deste repositório,
incluindo se `tensor` e `math` eventualmente se separam ainda mais, como
uma decisão futura e separada.

## Ecossistema

Uma organização, `github.com/pacolang`, um repositório por projeto.
[`pacolang/paco`](https://github.com/pacolang/paco) é o núcleo: compilador,
runtime e `stdlib`, uma única versão. `pacolang/numerics` é uma biblioteca
oficial ao lado dele, versionada de forma independente, obtida com
`paco get github.com/pacolang/numerics@<versão>` da mesma forma que
qualquer dependência. Decisões de design da linguagem e do seu ecossistema
são registradas como RFCs em [`pacolang/rfcs`](https://github.com/pacolang/rfcs),
nunca como ADRs.

## Primeiros Passos

### Pré-requisitos

- Um toolchain `paco` — veja [`pacolang/paco`](https://github.com/pacolang/paco)
  para construir ou instalar um.
- Um `paco.mod` no seu projeto (`paco mod init` se ainda não tiver um).

### Instalação

```bash
paco get github.com/pacolang/numerics@<versão>
paco mod tidy
```

## Uso

```paco
use tensor;
use math;

fn main() {
    // Um tensor de forma estática: 2x3, ambas as dimensões conhecidas em
    // tempo de compilação.
    let a = tensor::Tensor<f32, 2, 3>::zeros();
    let b = tensor::Tensor<f32, 2, 3>::zeros();
    let c = a.add(&b); // dá panic em caso de mismatch de forma; use checked_add caso contrário

    // Multiplicação de matrizes: a dimensão interna é provada igual pelo tipo.
    let m = math::Matrix<f32, 2, 3>::zeros();
    let n = math::Matrix<f32, 3, 4>::zeros();
    let product = m.mul(&n); // Result<Matrix<f32, 2, 4>, tensor::ShapeError>
}
```

Veja [`docs/spec.md`](docs/spec.md) para dimensões `Dyn`, operadores
`checked_*`, `Differentiable`/`grad`, e `DataFrame`/`Schema`.

## Roteiro

- `grad` sobre `Tensor` ainda não pode ser exercitado de ponta a ponta —
  uma lacuna do compilador em chamadas de método sobre um gradiente
  retornado por `grad`, para um tipo genérico com múltiplos registros de
  derivada usado fora do módulo que o declara. Registrado em
  [`docs/spec.md`](docs/spec.md), na seção `std::autodiff::Differentiable`; não é um bug desta biblioteca.
- `add`/`sub`/`mul` de `Matrix` ainda retornam `Result` em vez de terem
  recebido a mesma divisão panic/`checked_*` que `Tensor` já tem.
- Se `tensor` e `math` algum dia se separam em repositórios distintos é uma
  questão em aberto, deixada para uma RFC futura — veja a
  [RFC 0030](https://github.com/pacolang/rfcs/blob/main/text/0030-repository-organization-and-stdlib-scope.md).

Veja as [issues](https://github.com/pacolang/numerics/issues) e os
[milestones](https://github.com/pacolang/numerics/milestones) deste
repositório para o acompanhamento do dia a dia.

## Contribuindo

Contribuições são o que fazem da comunidade open-source um lugar incrível
para aprender e criar. Qualquer contribuição sua é **muito bem-vinda**.

1. Faça um fork do repositório.
2. Crie sua branch de feature (`git checkout -b feat/minha-feature`).
3. Rode `paco test` antes de abrir um pull request.
4. Faça commit das suas mudanças e abra um pull request.

Para uma mudança de design de linguagem ou de ecossistema, abra uma RFC em
[`pacolang/rfcs`](https://github.com/pacolang/rfcs) primeiro.

## Licença

Distribuído sob a Apache License, Version 2.0. Veja [`LICENSE`](LICENSE)
para mais informações.

## Contato

Link do projeto: [https://github.com/pacolang/numerics](https://github.com/pacolang/numerics)
