# sparse-interp

Basic polynomial arithmetic, multi-point evaluation, and sparse interpolation.

This crate is **very limiteda** so far in its functionality and under **active development**.
The current functionality isi mostly geared towards
sparse interpolation with a known set of possible exponents.
Expect frequent breaking changes as things get started.

The [`Poly`] type is used to represent dense polynomials along with traits for
algorithm choices. The [`ClassicalPoly`] type alias specifies classical arithmetic
algorithms via the [`ClassicalTraits`] trait.

```rust
use sparse_interp::ClassicalPoly;

// f represents 4 + 3x^2 - x^3
let f = ClassicalPoly::new(vec![4, 0, 3, -1]);

// g prepresents 2x
let g = ClassicalPoly::new(vec![0, 2]);

// basic arithmetic is supported
let h = f + g;
assert_eq!(h, ClassicalPoly::new(vec![4, 2, 3, -1]));
```

## Evaluation

Single-point and multi-point evaluation work as follows.

```rust
let h = ClassicalPoly::new(vec![4, 2, 3, -1]);
assert_eq!(h.eval(&0), 4);
assert_eq!(h.eval(&1), 8);
assert_eq!(h.eval(&-1), 6);
assert_eq!(h.mp_eval([0,1,-1].iter()), [4,8,6]);
```

If the same evaluation points are used for multiple polynomials,
they can be preprocessed with [`Poly::mp_eval_prep()`], and then
replacing [`Poly::mp_eval()`] with [`Poly::mp_eval_post()`] will
be more efficient overall.

## Sparse interpolation

Sparse interpolation should work over any type supporting
*field operations* of addition, subtration, multiplication,
and division.

For a polynomial *f* with at most *t* terms, sparse interpolation requires
eactly 2*t* evaluations at consecutive powers of some value θ, starting
with θ<sup>0</sup> = 1.

This value θ must have sufficiently high order in the underlying field;
that is, all powers of θ up to the degree of the polynomial must be distinct.

Calling [`Poly::sparse_interp()`] returns on success a vector of (exponent, coefficient)
pairs, sorted by exponent, corresponding to the nonzero terms of the
evaluated polynomial.

```rust
let f = ClassicalPoly::new(vec![0., -2.5, 0., 0., 0., 7.1]);
let t = 2;
let theta = 1.8f64;
let eval_pts = [1., theta, theta.powi(2), theta.powi(3)];
let evals = f.mp_eval(eval_pts.iter());
let error = 0.001;
let mut result = ClassicalPoly::sparse_interp(
    &theta,    // evaluation base point
    t,         // upper bound on nonzero terms
    0..8,      // iteration over possible exponents
    &evals,    // evaluations at powers of theta
    &RelativeParams::<f64>::new(Some(error), Some(error))
               // needed for approximate types like f64
).unwrap();

// round the coefficients to nearest 0.1
for (_,c) in result.iter_mut() {
    *c = (*c * 10.).round() / 10.;
}

assert_eq!(result, [(1, -2.5), (5, 7.1)]);
```

Current version: 0.0.1

## License

This software was written by [Daniel S. Roche](https://www.usna.edu/cs/roche/)
in 2021, as part of their job as a U.S. Government employee.
The source code therefore belongs in the
public domain in the United States and is not copyrightable.

[0BSD]: https://opensource.org/licenses/0BSD

Otherwise, the [0-clause BSD license](0BSD) applies.
