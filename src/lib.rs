#![warn(missing_docs)]
#![warn(missing_crate_level_docs)]
#![warn(macro_use_extern_crate)]
#![warn(invalid_html_tags)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(unreachable_pub)]
#![warn(unused_extern_crates)]
#![warn(unused_lifetimes)]

//! Basic polynomial arithmetic, multi-point evaluation, and sparse interpolation.
//!
//! This crate is **very limited** so far in its functionality and under **active development**.
//! The current functionality isi mostly geared towards
//! sparse interpolation with a known set of possible exponents.
//! Expect frequent breaking changes as things get started.
//!
//! The [`Poly`] type is used to represent dense polynomials along with traits for
//! algorithm choices. The [`ClassicalPoly`] type alias specifies classical arithmetic
//! algorithms via the [`ClassicalTraits`] trait.
//!
//! ```
//! use sparse_interp::ClassicalPoly;
//!
//! // f represents 4 + 3x^2 - x^3
//! let f = ClassicalPoly::<f32>::new(vec![4., 0., 3., -1.]);
//!
//! // g prepresents 2x
//! let g = ClassicalPoly::<f32>::new(vec![0., 2.]);
//!
//! // basic arithmetic is supported
//! let h = f + g;
//! assert_eq!(h, ClassicalPoly::new(vec![4., 2., 3., -1.]));
//! ```
//!
//! # Evaluation
//!
//! Single-point and multi-point evaluation work as follows.
//!
//! ```
//! # use sparse_interp::ClassicalPoly;
//! type CP = ClassicalPoly<f32>;
//! let h = CP::new(vec![4., 2., 3., -1.]);
//! assert_eq!(h.eval(&0.), Ok(4.));
//! assert_eq!(h.eval(&1.), Ok(8.));
//! assert_eq!(h.eval(&-1.), Ok(6.));
//! let eval_info = CP::mp_eval_prep([0., 1., -1.].iter().copied());
//! assert_eq!(h.mp_eval(&eval_info).unwrap(), [4.,8.,6.]);
//! ```
//!
//! # Sparse interpolation
//!
//! Sparse interpolation should work over any type supporting
//! *field operations* of addition, subtration, multiplication,
//! and division.
//!
//! For a polynomial *f* with at most *t* terms, sparse interpolation requires
//! eactly 2*t* evaluations at consecutive powers of some value θ, starting
//! with θ<sup>0</sup> = 1.
//!
//! This value θ must have sufficiently high order in the underlying field;
//! that is, all powers of θ up to the degree of the polynomial must be distinct.
//!
//! Calling [`Poly::sparse_interp()`] returns on success a vector of (exponent, coefficient)
//! pairs, sorted by exponent, corresponding to the nonzero terms of the
//! evaluated polynomial.
//!
//! ```
//! # use sparse_interp::{ClassicalPoly, RelativeParams};
//! type CP = ClassicalPoly<f64>;
//! let f = CP::new(vec![0., -2.5, 0., 0., 0., 7.1]);
//! let t = 2;
//! let (eval_info, interp_info) = ClassicalPoly::sparse_interp_prep(
//!     t,          // upper bound on nonzero terms
//!     0..8,       // iteration over possible exponents
//!     &f64::MAX,  // upper bound on coefficient magnitude
//! );
//! let evals = f.mp_eval(&eval_info).unwrap();
//! let mut result = CP::sparse_interp(&evals, &interp_info).unwrap();
//!
//! // round the coefficients to nearest 0.1
//! for (_,c) in result.iter_mut() {
//!     *c = (*c * 10.).round() / 10.;
//! }
//!
//! assert_eq!(result, [(1, -2.5), (5, 7.1)]);
//! ```

use core::{
    ops::{
        Add,
        Mul,
        AddAssign,
        SubAssign,
        MulAssign,
    },
    borrow::{
        Borrow,
        BorrowMut,
    },
    iter::{
        self,
        FromIterator,
        IntoIterator,
        ExactSizeIterator,
        Extend,
    },
    marker::{
        PhantomData,
    },
    convert::{
        self,
    },
};
use num_traits::{
    Zero,
    One,
    Inv,
};
use num_complex::{
    Complex,
    Complex32,
    Complex64,
};
use custom_error::custom_error;

custom_error!{
    /// Errors that arise in polynomial arithmetic or sparse interpolation.
    #[derive(PartialEq)]
    #[allow(missing_docs)]
    pub Error
        CoeffConv{cause: String}
            = "Could not convert coefficient: {cause}",
        Singular = "Encountered singular matrix when expecting nonsingular; perhaps sparsity was set too low",
        MissingExponents = "The exponent of a non-zero term was missing from the list",
}

/// A specialized [core::result::Result] type for sparse interpolation.
pub type Result<T> = core::result::Result<T, Error>;

/// A trait for 1-way conversions between numeric types that may fail.
///
/// Provides basically the same functionality as [`std::convert::TryFrom`]
/// from `std`, but a new trait was needed in order to implement it for
/// conversions such as `f64` to `u32`, or
/// [`num_complex::Complex32`] to `f32`.
///
/// See also the [conv crate](https://crates.io/crates/conv), which was
/// also considered but cannot be used here because it is not compatible
/// with [`num_complex::Complex32`].
pub trait OneWay {
    /// The type being converted from.
    type Source;
    /// The type being converted to.
    type Dest;
    /// Conversion from Source to Dest that may fail.
    fn one_way(src: Self::Source) -> Result<Self::Dest>;
}

/// A trait for 2-way conversions that may fail.
///
/// Of course, for any `x: Source`, it should be the case that
/// `other_way(one_way(x)?)? == x` when possible, but this is not
/// required.
pub trait TwoWay: OneWay {
    /// Conversion from Dest to Source that may fail.
    fn other_way(src: <Self as OneWay>::Dest) -> Result<<Self as OneWay>::Source>;
}

/// The default conversion from S to D, if it exists.
#[derive(Debug)]
pub struct DefConv<S,D>(PhantomData<(S,D)>);

impl<S,D> TwoWay for DefConv<S,D>
where Self: OneWay<Source=S, Dest=D>,
      DefConv<D,S>: OneWay<Source=D, Dest=S>,
{
    fn other_way(src: D) -> Result<S> {
        <DefConv<D,S> as OneWay>::one_way(src)
    }
}

impl<S> OneWay for DefConv<S,S>
{
    type Source = S;
    type Dest = S;

    #[inline(always)]
    fn one_way(src: S) -> Result<S> {
        Ok(src)
    }
}

macro_rules! one_way_as {
    ($s:ty, $($ds:ty),+) => {
        $(
            impl OneWay for DefConv<$s,$ds> {
                type Source = $s;
                type Dest = $ds;
                fn one_way(src: $s) -> Result<$ds> {
                    Ok(src as $ds)
                }
            }
        )+
    };
}

macro_rules! one_way_as_check {
    ($s:ty, $($ds:ty),+) => {
        $(
            impl OneWay for DefConv<$s,$ds> {
                type Source = $s;
                type Dest = $ds;
                fn one_way(src: $s) -> Result<$ds> {
                    let dest = src as $ds;
                    if (dest as $s) == src {
                        Ok(dest)
                    } else {
                        Err(Error::CoeffConv{cause: format!("{} not representable as {}", src, stringify!($ds))})
                    }
                }
            }
        )+
    }
}

macro_rules! one_way_round {
    ($s:ty, $($ds:ty),+) => {
        $(
            impl OneWay for DefConv<$s,$ds> {
                type Source = $s;
                type Dest = $ds;
                fn one_way(src: $s) -> Result<$ds> {
                    let rounded = src.round();
                    let dest = rounded as $ds;
                    if (dest as $s) == rounded {
                        Ok(dest)
                    } else {
                        Err(Error::CoeffConv{cause: format!("{} not representable as {}", src, stringify!($ds))})
                    }
                }
            }
        )+
    }
}

macro_rules! one_way_try {
    ($s:ty, $($ds:ty),+) => {
        $(
            impl OneWay for DefConv<$s,$ds> {
                type Source = $s;
                type Dest = $ds;
                fn one_way(src: $s) -> Result<$ds> {
                    convert::TryInto::<$ds>::try_into(src).map_err(|e| Error::CoeffConv{cause: e.to_string()})
                }
            }
        )+
    }
}

//one_way_as!(i8, i8);
one_way_try!(i8, u8);
one_way_as!(i8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);

one_way_try!(u8, i8);
one_way_as!(u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);

one_way_try!(i16, i8, u8, u16);
one_way_as!(i16, i32, u32, i64, u64, i128, u128, f32, f64);

one_way_try!(u16, i8, u8, i16);
one_way_as!(u16, i32, u32, i64, u64, i128, u128, f32, f64);

one_way_try!(i32, i8, u8, i16, u16, u32);
one_way_as!(i32, i64, u64, i128, u128, f64);
one_way_as_check!(i32, f32);

one_way_try!(u32, i8, u8, i16, u16, i32);
one_way_as!(u32, i64, u64, i128, u128, f64);
one_way_as_check!(u32, f32);

one_way_try!(i64, i8, u8, i16, u16, i32, u32, u64);
one_way_as!(i64, i128, u128);
one_way_as_check!(i64, f32, f64);

one_way_try!(u64, i8, u8, i16, u16, i32, u32, i64);
one_way_as!(u64, i128, u128);
one_way_as_check!(u64, f32, f64);

one_way_try!(i128, i8, u8, i16, u16, i32, u32, i64, u64, u128);
one_way_as_check!(i128, f32, f64);

one_way_try!(u128, i8, u8, i16, u16, i32, u32, i64, u64, i128);
one_way_as_check!(u128, f32, f64);

one_way_round!(f32, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);
one_way_as!(f32, f64);

one_way_round!(f64, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl OneWay for DefConv<f64, f32> {
    type Source = f64;
    type Dest = f32;

    fn one_way(src: f64) -> Result<f32> {
        let dest = src as f32;
        if dest.log2().round() == (src.log2().round() as f32) {
            Ok(dest)
        } else {
            Err(Error::CoeffConv{cause: format!("f64->f32 exponent overflow on {}", src)})
        }
    }
}

macro_rules! two_way_complex {
    ($t:ty, $($us:ty),+) => {
        $(
            impl OneWay for DefConv<$us, Complex<$t>> {
                type Source = $us;
                type Dest = Complex<$t>;

                #[inline(always)]
                fn one_way(src: Self::Source) -> Result<Self::Dest> {
                    Ok(Complex::from(DefConv::<$us,$t>::one_way(src)?))
                }
            }

            impl OneWay for DefConv<Complex<$t>, $us> {
                type Source = Complex<$t>;
                type Dest = $us;

                #[inline(always)]
                fn one_way(src: Self::Source) -> Result<Self::Dest> {
                    // TODO maybe return Err if imaginary part is too large?
                    DefConv::<$t,$us>::one_way(src.re)
                }
            }
        )+
    };
}

two_way_complex!(f32, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);
two_way_complex!(f64, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64);

/// A possibly-stateful comparison for exact or approximate types.
///
/// Implementors of this trait can be used to compare `Item`s for
/// "closeness". The idea is that closeness should encompass absolute
/// equality as well as approximate equality.
///
/// For exact types, use the [CloseToEq] struct to just fall back to direct
/// comparison. For inexact types, use the [RelativeParams] struct to specify
/// the acceptable (relative) error.
pub trait CloseTo {
    /// The type of thing that can be compared.
    type Item;

    /// Returns `true` iff `x` is approximatey equal to `y`.
    fn close_to(&self, x: &Self::Item, y: &Self::Item) -> bool;

    /// Indicates `true` if `x` is approximately zero.
    fn close_to_zero(&self, x: &Self::Item) -> bool;

    /// Checks closeness over an iteration.
    fn close_to_iter<'a, Iter1, Iter2>(&'a self, x: Iter1, y: Iter2) -> bool
    where Iter1: Iterator<Item=&'a Self::Item>,
          Iter2: Iterator<Item=&'a Self::Item>,
    {
        x.zip(y).all(|(xi, yi)| self.close_to(xi, yi))
    }
}

/// A struct to use for exact equality in the [`CloseTo`] trait.
///
/// ```
/// # use sparse_interp::*;
/// let test = CloseToEq::new();
/// assert!(test.close_to(&15, &15));
/// assert!(! test.close_to(&15, &16));
/// assert!(test.close_to_zero(&0));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloseToEq<T>{
    /// The zero element (additive identity) for type `T`.
    pub zero: T,
}

impl<T: Zero> CloseToEq<T> {
    /// Creates a new struct for exact equality, using [`num_traits::Zero::zero`] to get the zero element.
    pub fn new() -> Self {
        Self {
            zero: T::zero(),
        }
    }
}

impl<T: PartialEq> CloseTo for CloseToEq<T> {
    type Item = T;

    #[inline(always)]
    fn close_to(&self, x: &Self::Item, y: &Self::Item) -> bool {
        x.eq(y)
    }

    #[inline(always)]
    fn close_to_zero(&self, x: &Self::Item) -> bool {
        x.eq(&self.zero)
    }
}

/// A struct to use for approximate equality.
///
/// ```
/// # use sparse_interp::*;
/// let test = RelativeParams::<f32>::new(Some(0.05), Some(0.001));
/// assert!(test.close_to(&1.0, &1.001));
/// assert!(test.close_to(&1001., &1000.));
/// assert!(test.close_to_zero(&-0.05));
/// assert!(! test.close_to(&0.1, &0.11));
/// assert!(! test.close_to_zero(&0.06));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeParams<T, E=T>
{
    /// Below this threshold in absolute value, values are considered close to zero.
    pub zero_thresh: E,

    /// Values not close to zero are considered close to each other if their relative
    /// error is at most this large.
    pub max_relative: E,

    phantom: PhantomData<T>,
}

macro_rules! impl_relative_params {
    ($t:ident, $e:ident, $norm:ident) => {
        impl RelativeParams<$t,$e> {
            /// Create a new closeness tester with the given parameters.
            ///
            /// For both arguments, `zero_thresh` and `max_relative`,
            /// either the given bound is used, or machine epsilon if the argument
            /// is `None`.
            #[inline(always)]
            pub fn new(zero_thresh: Option<$e>, max_relative: Option<$e>) -> Self {
                Self {
                    zero_thresh: zero_thresh.unwrap_or($e::EPSILON),
                    max_relative: max_relative.unwrap_or($e::EPSILON),
                    phantom: PhantomData,
                }
            }
        }

        impl Default for RelativeParams<$t,$e> {
            /// Create a closeness tester with machine epsilon precision.
            #[inline(always)]
            fn default() -> Self {
                Self::new(None, None)
            }
        }

        impl CloseTo for RelativeParams<$t,$e> {
            type Item = $t;

            #[inline]
            fn close_to(&self, x: &Self::Item, y: &Self::Item) -> bool {
                if x == y {
                    return true
                };
                if $t::is_infinite(*x) || $t::is_infinite(*y) {
                    return false
                };

                let absx = $t::$norm(*x);
                let absy = $t::$norm(*y);

                let largest = if absx <= absy {
                    absy
                } else {
                    absx
                };

                if largest <= self.zero_thresh {
                    true
                } else {
                    $t::$norm(x - y) / largest <= self.max_relative
                }
            }

            #[inline(always)]
            fn close_to_zero(&self, x: &Self::Item) -> bool {
                $t::$norm(*x) <= self.zero_thresh
            }
        }
    };
}

impl_relative_params!(f32, f32, abs);
impl_relative_params!(f64, f64, abs);
impl_relative_params!(Complex32, f32, norm);
impl_relative_params!(Complex64, f64, norm);

/// Algorithms to enable polynomial arithmetic.
///
/// Generally, `PolyTraits` methods should not be used directly, but only
/// within the various method impls for [`Poly`].
///
/// This is implemented as a separate, possibly stateless traits object
/// in order to allow selecting different underlying algorithms separately
/// from the overall representation.
///
/// The methods here generally work on slice references so as to be
/// representation-agnostic.
///
/// So far the only implementation is [`ClassicalTraits`].
pub trait PolyTraits {
    /// The type of polynomial coefficients.
    type Coeff;

    /// The evaluation needed for sparse interpolation.
    type SparseInterpEval: EvalTypes<Coeff=Self::Coeff>;

    /// An opaque type returned by the pre-processing method [`Self::sparse_interp_prep()`].
    type SparseInterpInfo;

    /// Multiplies two polynomails (represented by slices) and stores the result in another slice.
    ///
    /// Implementations may assume that all slices are non-empty, and that
    /// `out.len() == lhs.len() + rhs.len() - 1`.
    ///
    /// ```
    /// # use sparse_interp::*;
    /// # type TraitImpl = ClassicalTraits<f32>;
    /// let a = [1., 2., 3.];
    /// let b = [4., 5.];
    /// let mut c = [0.; 4];
    /// TraitImpl::slice_mul(&mut c[..], &a[..], &b[..]);
    /// assert_eq!(c, [1.*4., 1.*5. + 2.*4., 2.*5. + 3.*4., 3.*5.]);
    /// ```
    fn slice_mul(out: &mut [Self::Coeff], lhs: &[Self::Coeff], rhs: &[Self::Coeff]);

    /// Pre-processing for multi-point evaluation.
    ///
    /// This method must be called to specify the evaluation points prior to
    /// calling [`Self::mp_eval_slice()`].
    ///
    /// The same pre-processed output can be used repeatedly to
    /// evaluate possibly different polynomials at the same points.
    ///
    /// The default implementation should be used; it relies on the [`EvalTypes::prep()`]
    /// trait method specialized for the coefficient and evaluation types.
    #[inline(always)]
    fn mp_eval_prep<U>(pts: impl Iterator<Item=U>) -> <EvalTrait<Self,U> as EvalTypes>::EvalInfo
    where EvalTrait<Self,U>: EvalTypes<Coeff=Self::Coeff, Eval=U>
    {
        <EvalTrait<Self,U> as EvalTypes>::prep(pts)
    }


    /// Multi-point evaluation.
    ///
    /// Evaluates the polynomial (given by a slice of coefficients) at all points
    /// specified in a previous call to [`Self::mp_eval_prep()`].
    ///
    /// ```
    /// # use sparse_interp::*;
    /// # type TraitImpl = ClassicalTraits<f32>;
    /// let pts = [10., -5.];
    /// let preprocess = TraitImpl::mp_eval_prep(pts.iter().copied());
    ///
    /// let f = [1., 2., 3.];
    /// let mut evals = Vec::new();
    /// TraitImpl::mp_eval_slice(&mut evals, &f[..], &preprocess);
    /// assert_eq!(evals, vec![321., 66.]);
    ///
    /// let g = [4., 5., 6., 7.];
    /// TraitImpl::mp_eval_slice(&mut evals, &g[..], &preprocess);
    /// assert_eq!(evals, vec![321., 66., 7654., 4. - 5.*5. + 6.*25. - 7.*125.]);
    /// ```
    ///
    /// The provided implementation should generally be used; it relies on the
    /// [`EvalTypes::post()`] trait method specialized for the coefficient and
    /// evaluation types.
    #[inline(always)]
    fn mp_eval_slice<U>(out: &mut impl Extend<U>,
                        coeffs: &[Self::Coeff],
                        info: &<EvalTrait<Self,U> as EvalTypes>::EvalInfo) -> Result<()>
    where EvalTrait<Self,U>: EvalTypes<Coeff=Self::Coeff, Eval=U>
    {
        <EvalTrait<Self,U> as EvalTypes>::post(out, coeffs, info)
    }

    /// Pre-processing for sparse interpolation.
    ///
    /// This method must be called prior to
    /// calling [`Self::sparse_interp_slice()`].
    ///
    /// A later call to sparse interpolation is guaranteed to succeed only when, for some
    /// unknown polynomial `f`, the following are all true:
    /// *   `f` has at most `sparsity` non-zero terms
    /// *   The exponents of all non-zero terms of `f` appear in `expons`
    /// *   The coefficients of `f` are bounded by `max_coeff` in magnitude.
    ///
    /// The list `expons` must be sorted in ascending order.
    ///
    /// The same pre-processed output can be used repeatedly to
    /// interpolate possibly different polynomials under the same settings.
    fn sparse_interp_prep(sparsity: usize, expons: impl Iterator<Item=usize>, max_coeff: &Self::Coeff)
        -> (<Self::SparseInterpEval as EvalTypes>::EvalInfo, Self::SparseInterpInfo)
    ;

    /// Sparse interpolation following evaluation.
    ///
    /// The evaluations in `eval` should correspond to what was specified by
    /// [`Self::sparse_interp_prep()`].
    ///
    /// If those requirements are met, the function will return Some(..) containing
    /// a list of exponent-coefficient pairs, sorted in ascending order of exponents.
    ///
    /// Otherwise, for example if the evaluated function has more non-zero terms
    /// than the pre-specified limit, this function *may* return None or may return
    /// Some(..) with incorrect values.
    fn sparse_interp_slice(evals: &[<Self::SparseInterpEval as EvalTypes>::Eval],
                           info: &Self::SparseInterpInfo)
        -> Result<Vec<(usize, Self::Coeff)>>;
}

/// A trait struct used for multi-point evaluation of polynomials.
///
/// Typically, `T` will be a type which implements `PolyTraits` and
/// `U` will be the type of evaluation point. If the combination of `T`
/// and `U` is meaningful, then `EvalTrait<T,U>` should implement
/// `EvalTypes`; this indicates that polynomials under trait `T`
/// can be evaluated at points of type `U`.
///
/// The purpose of this is to allow a polynomial with coefficients of
/// one type (say, integers) to be evaluated at a point with another type
/// (say, complex numbers, or integers modulo a prime).
///
/// ```
/// # use sparse_interp::*;
/// type Eval = EvalTrait<ClassicalTraits<i32>, f32>;
/// let coeffs = [1, 0, -2];
/// let pts = [0.5, -1.5];
/// let info = Eval::prep(pts.iter().copied());
/// let mut result = Vec::new();
/// Eval::post(&mut result, &coeffs, &info);
/// assert_eq!(result, vec![0.5, -3.5]);
/// ```
///
/// This is essentially a work-around for a problem better solved by
/// [GATs](https://github.com/rust-lang/rust/issues/44265).
/// (At the time of this writing, GAT is merged in nightly rust but not
/// stable.)
#[derive(Debug)]
pub struct EvalTrait<T: ?Sized, U>(PhantomData<T>,PhantomData<U>);

/// Trait for evaluating polynomials over (possibly) a different domain.
///
/// Evaluation is divided into pre-processing [`EvalTypes::prep`] stage, which depends only on the
/// evaluation points, and the actual evaluation phase [`EvalTypes::post`].
///
/// Typically the provided blanket implementations for [`EvalTrait`] should be sufficient,
/// unless you care creating a new polynomial type.
pub trait EvalTypes {
    /// Coefficient type of the polynomial being evaluated.
    type Coeff;
    /// Type of the evaluation point(s) and the output(s) of the polynomial evaluation.
    type Eval;
    /// Opaque type to hold pre-processing results.
    type EvalInfo;

    /// Pre-processing for multi-point evaluation.
    ///
    /// Takes a list of evaluation points and prepares an `EvalInfo` opaque
    /// object which can be re-used to evaluate multiple polynomials over
    /// the same evaluation points.
    fn prep(pts: impl Iterator<Item=Self::Eval>) -> Self::EvalInfo;

    /// Multi-point evaluation after pre-processing.
    fn post(out: &mut impl Extend<Self::Eval>, coeffs: &[Self::Coeff], info: &Self::EvalInfo) -> Result<()>;
}

impl<C,U> EvalTypes for EvalTrait<ClassicalTraits<C>, U>
where C: Clone,
      U: Clone + Zero + MulAssign + AddAssign,
      DefConv<C,U>: OneWay<Source=C, Dest=U>,
{
    type Coeff = C;
    type Eval = U;
    type EvalInfo = Vec<U>;

    #[inline(always)]
    fn prep(pts: impl Iterator<Item=Self::Eval>) -> Self::EvalInfo {
        pts.collect()
    }

    fn post(out: &mut impl Extend<Self::Eval>, coeffs: &[Self::Coeff], info: &Self::EvalInfo) -> Result<()> {
        //out.extend(coeffs.iter().zip(info.iter()).map(|(c,x)| { let mut out = x.clone(); out *= c; out }));
		ErrorIter::new(info.iter().map(|x| horner_eval(coeffs.iter(), x)))
            .map(|evals| out.extend(evals))
        //out.extend(info.iter().map(|x| horner_eval(coeffs.iter(), x)));
    }
}

/// PolyTraits implementation for classical (slow) algorithms.
///
/// The underlying algorithms are neither fast nor numerically stable.
/// They should be used only for very small sizes, or for debugging.
///
/// *   [`ClassicalTraits::slice_mul()`] uses a classical quadratic-time algorithm
/// *   [`ClassicalTraits::mp_eval_slice()`] simply evaluates each point one at a time, using
///     Horner's rule. O(num_points * poly_degree) time.
/// *   [`ClassicalTraits::sparse_interp_slice()`] uses unstructured linear algebra and classical
///     root finding, with complexity O(sparsity^3).
#[derive(Debug,Default,Clone,Copy,PartialEq,Eq)]
pub struct ClassicalTraits<C>(PhantomData<C>);

// FIXME remove
fn at_a_loss(x: &Vec<Complex64>, y: &Complex64) -> Result<Complex64> {
    horner_eval(x.iter(), y)
}

impl<C> PolyTraits for ClassicalTraits<C>
where C: Clone + Mul<Output=C> + AddAssign,
      DefConv<C, Complex64>: TwoWay<Source=C, Dest=Complex64>,
{
    type Coeff = C;
    type SparseInterpEval = EvalTrait<Self, Complex64>;
    type SparseInterpInfo = (usize, Vec<(usize, Complex64)>, RelativeParams<Complex64, f64>);

    #[inline(always)]
    fn slice_mul(out: &mut [Self::Coeff], lhs: &[Self::Coeff], rhs: &[Self::Coeff]) {
        classical_slice_mul(out, lhs, rhs);
    }

    fn sparse_interp_prep(sparsity: usize, expons: impl Iterator<Item=usize>, _max_coeff: &Self::Coeff)
        -> (<Self::SparseInterpEval as EvalTypes>::EvalInfo, Self::SparseInterpInfo)
    {
        let mut theta_pows: Vec<_> = expons.map(|d| (d, Complex64::default())).collect();
        let theta = match theta_pows.iter().map(|pair| pair.0).max() {
            Some(max_pow) => Complex64::from_polar(1., 2.*core::f64::consts::PI / (max_pow as f64 + 1.)),
            None => Complex64::default(),
        };
        for (ref expon, ref mut power) in theta_pows.iter_mut() {
            *power = theta.powu(*expon as u32);
        }
        (EvalTrait::<Self, Complex64>::prep((0..2*sparsity).map(|e| theta.powu(e as u32))),
         (sparsity, theta_pows, RelativeParams::<Complex64,f64>::new(Some(1e-10), Some(1e-10))) //TODO how to choose approx params
        )
    }

    fn sparse_interp_slice(
        evals: &[Complex64],
        info: &Self::SparseInterpInfo,
        ) -> Result<Vec<(usize, Self::Coeff)>>
    {
        let (sparsity, pows, close) = info;
        assert_eq!(evals.len(), 2*sparsity);
        let mut lambda = bad_berlekamp_massey(evals, close)?;
        lambda.push(Complex64::from(-1.));
        let (degs, roots): (Vec<usize>, Vec<Complex64>) = pows.iter().filter_map(
            |(deg, rootpow)| {
            // FIXME why helper function is needed?
            //horner_eval(lambda.iter(), rootpow).ok().and_then(
            at_a_loss(&lambda, rootpow).ok().and_then(
                |eval| match close.close_to_zero(&eval) {
                    true => Some((deg, rootpow)),
                    false => None,
                }
            )}
        ).unzip();
        if degs.len() != lambda.len() - 1 {
            Err(Error::MissingExponents)
        } else {
            let evslice = &evals[..degs.len()];
            Ok(degs.into_iter().zip(
                bad_trans_vand_solve(&roots, evslice, close)?.into_iter()
                    .map(|c| DefConv::<C,Complex64>::other_way(c).unwrap()) // FIXME avoid unwrap
            ).collect())
        }
    }
}

/// Generic struct to hold a polynomial and traits for operations.
///
/// The idea here is to combine some representation of the polynomial
/// (say, a vector of coefficients) with an implementation of [`PolyTraits`],
/// to allow implementing all the standard arithmetic and other user-facing
/// polynomial operations.
///
/// ```
/// # use sparse_interp::*;
/// let f = ClassicalPoly::<f64>::new(vec![1., 2., 3.]);
/// let g = ClassicalPoly::<f64>::new(vec![4., 5.]);
///
/// assert_eq!(&f * &g, ClassicalPoly::new(vec![4., 13., 22., 15.]));
/// assert_eq!(&f + &g, ClassicalPoly::new(vec![5., 7., 3.]));
/// ```
///
/// Type aliases are provided for various combinations that will work well.
/// So far the only alias is [`ClassicalPoly`].
#[derive(Debug)]
#[repr(transparent)]
pub struct Poly<T,U> {
    rep: T,
    traits: PhantomData<U>,
}

/// Univeriate polynomial representation using classical arithmetic algorithms.
///
/// Objects of this type implement many standard numeric operations like +, -,
/// usually on *references* to the type.
///
/// Multi-point evaluation and sparse interpolation routines are also supported.
pub type ClassicalPoly<C> = Poly<Vec<C>, ClassicalTraits<C>>;

impl<T,U,V,W> PartialEq<Poly<V,W>> for Poly<T,U>
where U: PolyTraits,
      W: PolyTraits,
      T: Borrow<[U::Coeff]>,
      V: Borrow<[W::Coeff]>,
      U::Coeff: PartialEq<W::Coeff>,
{
    fn eq(&self, other: &Poly<V,W>) -> bool {
        self.rep.borrow() == other.rep.borrow()
    }
}

impl<T,U> Eq for Poly<T,U>
where U: PolyTraits,
      T: Borrow<[U::Coeff]>,
      U::Coeff: Eq,
{ }

impl<U> Poly<Vec<U::Coeff>,U>
where U: PolyTraits,
      U::Coeff: Zero,
{
    /// Creates a new `Poly` from a vector of coefficients.
    pub fn new(rep: Vec<U::Coeff>) -> Self {
        let mut out = Self {
            rep,
            traits: PhantomData,
        };
        out.normalize();
        out
    }

    /// Ensures nonzero leading coefficient (or empty).
    fn normalize(&mut self) {
        while self.rep.last().map_or(false, Zero::is_zero) {
            self.rep.pop();
        }
    }
}

impl<T,U> Poly<T,U>
where U: PolyTraits,
      T: Borrow<[U::Coeff]>,
      U::Coeff: Zero,
{
    /// Creates a new `Poly` from the underlying representation which has nonzero
    /// leading coefficient.
    ///
    /// # Panics
    /// Panics if `rep` if not empty and has a leading coefficient which is zero.
    pub fn new_norm(rep: T) -> Self {
        assert!(rep.borrow().last().map_or(true, |x| !x.is_zero()));
        Self {
            rep,
            traits: PhantomData,
        }
    }
}

impl<T,U> Default for Poly<T,U>
where T: Default,
{
    #[inline(always)]
    fn default() -> Self {
        Self {
            rep: T::default(),
            traits: PhantomData,
        }
    }
}

impl<T,U> FromIterator<U::Coeff> for Poly<T,U>
where U: PolyTraits,
      T: FromIterator<U::Coeff> + Borrow<[U::Coeff]>,
      U::Coeff: Zero,
{
    fn from_iter<V>(iter: V) -> Self
    where V: IntoIterator<Item=U::Coeff>,
    {
        struct Trimmed<I: Iterator>(I, Option<(usize, I::Item)>);
        impl<I: Iterator> Iterator for Trimmed<I>
        where I::Item: Zero,
        {
            type Item = I::Item;
            fn next(&mut self) -> Option<Self::Item> {
                match self.1.take() {
                    Some((count, nzcoeff)) => Some(
                        if count == 0 {
                            nzcoeff
                        } else {
                            self.1 = Some((count - 1, nzcoeff));
                            Zero::zero()
                        }
                    ),
                    None => self.0.next().and_then(|coeff| {
                        if coeff.is_zero() {
                            let mut count = 1;
                            while let Some(coeff) = self.0.next() {
                                if ! coeff.is_zero() {
                                    self.1 = Some((count-1, coeff));
                                    return Some(Zero::zero());
                                }
                                count += 1;
                            }
                            None
                        } else {
                            Some(coeff)
                        }
                    }),
                }
            }
        }
        Self::new_norm(Trimmed(iter.into_iter(), None).collect())
    }
}

impl<T,U> Poly<T,U>
where U: PolyTraits,
      T: Borrow<[U::Coeff]>,
      U::Coeff: Clone + Zero + AddAssign + MulAssign,
{
    /// Evaluate this polynomial at the given point.
    ///
    /// Uses Horner's rule to perform the evaluation using exactly d multiplications
    /// and d additions, where d is the degree of self.
    #[inline(always)]
    pub fn eval<V>(&self, x: &V) -> Result<V>
    where DefConv<U::Coeff, V>: OneWay<Source=U::Coeff, Dest=V>,
          V: Clone + Zero + AddAssign + MulAssign,
    {
        horner_eval(self.rep.borrow().iter(), x)
    }

    /// Perform pre-processing for multi-point evaluation.
    ///
    /// `pts` is an iterator over the desired evaluation points.
    ///
    /// See [`PolyTraits::mp_eval_prep()`] for more details.
    #[inline(always)]
    pub fn mp_eval_prep<V>(pts: impl Iterator<Item=V>) -> <EvalTrait<U,V> as EvalTypes>::EvalInfo
    where EvalTrait<U, V>: EvalTypes<Coeff=U::Coeff, Eval=V>,
    {
        U::mp_eval_prep(pts)
    }

    /// Evaluate this polynomial at all of the given points.
    ///
    /// Performs multi-point evaluation using the underlying trait algorithms.
    /// In general, this can be much more efficient than repeatedly calling
    /// [`self.eval()`].
    ///
    /// `info` should be the result of calling [`Self::mp_eval_prep()`].
    #[inline]
    pub fn mp_eval<V>(&self, info: &<EvalTrait<U,V> as EvalTypes>::EvalInfo) -> Result<Vec<V>>
    where EvalTrait<U, V>: EvalTypes<Coeff=U::Coeff, Eval=V>,
    {
        let mut out = Vec::new();
        U::mp_eval_slice(&mut out, self.rep.borrow(), info)?;
        Ok(out)
    }
}

impl<T,U> Poly<T,U>
where U: PolyTraits,
{
    /// Perform pre-processing for sparse interpolation.
    ///
    /// *   `sparsity` is an upper bound on the number of nonzero terms in the evaluated
    ///     polynomial.
    /// *   `expons` is an iteration over the possible exponents which may appear in nonzero
    ///     terms.
    /// *   `max_coeff` is an upper bound on the magnitude of any coefficient.
    ///
    /// See [`PolyTraits::sparse_interp_prep()`] for more details.
    #[inline(always)]
    pub fn sparse_interp_prep(sparsity: usize, expons: impl Iterator<Item=usize>, max_coeff: &U::Coeff)
        -> (<U::SparseInterpEval as EvalTypes>::EvalInfo, U::SparseInterpInfo)
    {
        U::sparse_interp_prep(sparsity, expons, max_coeff)
    }

    /// Perform sparse interpolation after evaluation.
    ///
    /// Beforehand, [`Self::sparse_interp_prep()`] must be called to generate two info
    /// structs for evaluation and interpolation. These can be used repeatedly to
    /// evaluate and interpolate many polynomials with the same sparsity and degree
    /// bounds.
    ///
    /// *   `evals` should correspond evaluations according to the `EvalInfo` struct
    ///     returned from preprocessing.
    ///     at consecutive powers of the `theta` used in preprocessing.
    /// *   `info` should come from a previous call to [`Self::sparse_interp_prep()`].
    ///
    /// On success, a vector of (exponent, nonzero coefficient) pairs is returned,
    /// sorted by increasing exponent values.
    #[inline(always)]
    pub fn sparse_interp(evals: &[<U::SparseInterpEval as EvalTypes>::Eval], info: &U::SparseInterpInfo)
        -> Result<Vec<(usize, U::Coeff)>>
    {
        U::sparse_interp_slice(evals, info)
    }
}

#[must_use]
struct LongZip<A,B,F,G,H>(A,B,F,G,H);

impl<A,B,F,G,H,O> Iterator for LongZip<A,B,F,G,H>
where A: Iterator,
      B: Iterator,
      F: FnMut(A::Item, B::Item) -> O,
      G: FnMut(A::Item) -> O,
      H: FnMut(B::Item) -> O,
{
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(x) => Some(match self.1.next() {
                Some(y) => self.2(x,y),
                None => self.3(x),
            }),
            None => self.1.next().map(|y| self.4(y)),
        }
    }
}

impl<'a,'b,T,U,V,W> Add<&'b Poly<V,W>> for &'a Poly<T,U>
where U: PolyTraits,
      W: PolyTraits,
      &'a T: IntoIterator<Item=&'a U::Coeff>,
      &'b V: IntoIterator<Item=&'b W::Coeff>,
      &'a U::Coeff: Add<&'b W::Coeff, Output=U::Coeff>,
      U::Coeff: AddAssign<&'b W::Coeff>,
      U::Coeff: Clone + Zero
{
    type Output = Poly<Vec<U::Coeff>, U>;

    fn add(self, rhs: &'b Poly<V,W>) -> Self::Output {
        Poly::from_iter(LongZip(
            self.rep.into_iter(),
            rhs.rep.into_iter(),
            Add::add,
            Clone::clone,
            |x| {
                let mut sum = U::Coeff::zero();
                sum += x;
                sum
            },
        ))
    }
}

impl<T,U,V,W> Add<Poly<V,W>> for Poly<T,U>
where U: PolyTraits,
      W: PolyTraits,
      T: IntoIterator<Item=U::Coeff>,
      V: IntoIterator<Item=W::Coeff>,
      U::Coeff: Zero + From<W::Coeff> + Add<W::Coeff, Output=U::Coeff>,
{
    type Output = Poly<Vec<U::Coeff>, U>;

    fn add(self, rhs: Poly<V,W>) -> Self::Output {
        Poly::from_iter(LongZip(
            self.rep.into_iter(),
            rhs.rep.into_iter(),
            Add::add,
            convert::identity,
            From::from,
        ))
    }
}

impl<'a,'b,T,U,V> Mul<&'b Poly<V,U>> for &'a Poly<T,U>
where U: PolyTraits,
      T: Borrow<[U::Coeff]>,
      V: Borrow<[U::Coeff]>,
      U::Coeff: Zero,
{
    type Output = Poly<Box<[U::Coeff]>, U>;

    fn mul(self, rhs: &'b Poly<V,U>) -> Self::Output {
        let aslice = self.rep.borrow();
        let bslice = rhs.rep.borrow();
        if aslice.is_empty() || bslice.is_empty() {
            return Poly::new_norm(Box::new([]));
        }

        let clen = aslice.len() + bslice.len() - 1;
        let mut outbox = {
            let mut vec = Vec::with_capacity(clen);
            vec.resize_with(clen, U::Coeff::zero);
            vec.into_boxed_slice()
        };

        U::slice_mul(outbox.borrow_mut(), aslice, bslice);

        Poly::new_norm(outbox)
    }
}

fn dot_product<'a,'b,T,U>(lhs: impl Iterator<Item=&'a T>, rhs: impl Iterator<Item=&'b U>) -> T
where T: 'a + Clone + Mul<Output=T> + AddAssign,
      U: 'b + Clone + Into<T>,
{
    let mut zipit = lhs.cloned().zip(rhs.cloned());
    let (a,b) = zipit.next().expect("dot product must be with non-empty iterators");
    let mut out = a * b.into();
    for (a,b) in zipit {
        out += a * b.into();
    }
    out
}

fn classical_slice_mul<T,U>(output: &mut [T], lhs: &[T], rhs: &[U])
where T: Clone + Mul<Output=T> + AddAssign,
      U: Clone + Into<T>,
{
    if lhs.len() == 0 || rhs.len() == 0 {
        assert_eq!(output.len(), 0);
        return;
    }

    assert_eq!(lhs.len() + rhs.len(), output.len() + 1);

    let mut i = 0;

    while i < lhs.len() && i < rhs.len() {
        output[i] = dot_product(lhs[..i+1].iter(), rhs[..i+1].iter().rev());
        i = i + 1;
    }

    let mut j = 1;
    while i < lhs.len() {
        output[i] = dot_product(lhs[j..i+1].iter(), rhs.iter().rev());
        i = i + 1;
        j = j + 1;
    }

    let mut k = 1;
    while i < rhs.len() {
        output[i] = dot_product(lhs.iter(), rhs[k..i+1].iter().rev());
        i = i + 1;
        k = k + 1;
    }

    while i < output.len() {
        output[i] = dot_product(lhs[j..].iter(), rhs[k..].iter().rev());
        i = i + 1;
        j = j + 1;
        k = k + 1;
    }
}

fn horner_eval<'a,'b,T,U>(mut coeffs: impl DoubleEndedIterator<Item=&'a T>, x: &'b U)
    -> Result<U>
where T: 'a + Clone,
      U: Clone + Zero + MulAssign + AddAssign,
      DefConv<T,U>: OneWay<Source=T, Dest=U>,
{
    if let Some(leading) = coeffs.next_back() {
        let mut out = DefConv::<T,U>::one_way(leading.clone())?;
        for coeff in coeffs.rev() {
            out *= x.clone();
            out += DefConv::<T,U>::one_way(coeff.clone())?;
        }
        Ok(out)
    } else {
        Ok(U::zero())
    }
}

/// Computes base^exp, where the base is a reference and not an owned value.
///
/// The number of clones is kept to a minimum. This is useful when objects of type T
/// are large and copying them is expensive.
///
/// ```
/// # use sparse_interp::refpow;
/// let x = 10u128;
/// assert_eq!(refpow(&x, 11), 100000000000u128)
/// ```
///
/// see also: <https://docs.rs/num-traits/0.2.14/src/num_traits/pow.rs.html#189-211>
#[inline]
pub fn refpow<T>(base: &T, exp: usize) -> T
where T: Clone + One + Mul<Output=T> + MulAssign,
{
    match exp {
        0 => T::one(),
        1 => base.clone(),
        _ => {
            let mut acc = base.clone() * base.clone();
            let mut curexp = exp >> 1;

            // invariant: curexp > 0 and base^exp = acc^curexp * base^(exp mod 2)
            while curexp & 1 == 0 {
                acc *= acc.clone();
                curexp >>= 1;
            }
            // now: curexp positive odd, base^exp = acc^curexp * base^(exp mod 2)

            if curexp > 1 {
                let mut basepow = acc.clone() * acc.clone();
                curexp >>= 1;

                // invariant: curexp > 0 and base^exp = acc * basepow^curexp * base^(exp mod 2)
                while curexp > 1 {
                    if curexp & 1 == 1 {
                        acc *= basepow.clone();
                    }
                    basepow *= basepow.clone();
                    curexp >>= 1;
                }
                // now: curexp == 1 and base^exp = acc * basepow * base^(exp mod 2)

                acc *= basepow.clone();
            }
            // now: curexp == 1 and base^exp = acc * base^(exp mod 2)

            if exp & 1 == 1 {
                acc *= base.clone();
            }
            acc
        }
    }
}


// --- BAD LINEAR SOLVER, TODO REPLACE WITH BETTER "SUPERFAST" SOLVERS ---

fn bad_linear_solve<'a,'b,M,T>(
    matrix: M,
    rhs: impl IntoIterator<Item=&'b T>,
    close: &impl CloseTo<Item=T>,
    ) -> Result<Vec<T>>
where M: IntoIterator,
      M::Item: IntoIterator<Item=&'a T>,
      T: 'a + 'b + Clone + One + Mul<Output=T> + SubAssign + MulAssign + Inv<Output=T>,
{
    let mut workmat: Vec<Vec<_>> =
        matrix .into_iter()
            .map(|row| row.into_iter().map(Clone::clone).collect())
            .collect();
    let mut sol: Vec<_> = rhs.into_iter().map(Clone::clone).collect();

    let n = workmat.len();
    assert!(workmat.iter().all(|row| row.len() == n));
    assert_eq!(sol.len(), n);

    for i in 0..n {
        { // find pivot
            let mut j = i;
            while close.close_to_zero(&workmat[j][i]) {
                j += 1;
                if j == n {
                    return Err(Error::Singular);
                }
            }
            if i != j {
                workmat.swap(i, j);
                sol.swap(i, j);
            }
        }
        // normalize pivot row
        {
            let pivot_inverse = workmat[i][i].clone().inv();
            for x in workmat[i].split_at_mut(i+1).1 {
                *x *= pivot_inverse.clone();
            }
            sol[i] *= pivot_inverse;
            workmat[i][i].set_one();
        }
        // cancel
        {
            let (top, midbot) = workmat.split_at_mut(i);
            let (pivrow, bottom) = midbot.split_first_mut().expect("row index i must exist");
            let (soltop, solmidbot) = sol.split_at_mut(i);
            let (solpiv, solbot) = solmidbot.split_first_mut().expect("row index imust exist");
            for (row, solx) in top.iter_mut().chain(bottom.iter_mut())
                               .zip(soltop.iter_mut().chain(solbot.iter_mut()))
            {
                if !close.close_to_zero(&row[i]) {
                    let (left, right) = row.split_at_mut(i+1);
                    for (j, x) in (i+1..n).zip(right) {
                        *x -= left[i].clone() * pivrow[j].clone();
                    }
                    *solx -= left[i].clone() * solpiv.clone();
                }
            }
        }
    }
    Ok(sol)
}

fn bad_berlekamp_massey<T>(seq: &[T], close: &impl CloseTo<Item=T>) -> Result<Vec<T>>
where T: Clone + One + Mul<Output=T> + SubAssign + MulAssign + Inv<Output=T>,
{
    assert_eq!(seq.len() % 2, 0);
    let n = seq.len() / 2;
    bad_linear_solve(
        (0..n).map(|i| &seq[i..i+n]),
        &seq[n..2*n],
        close)
}

fn bad_trans_vand_solve<'a,'b,T,U>(
    roots: U,
    rhs: impl IntoIterator<Item=&'b T>,
    close: &impl CloseTo<Item=T>)
    -> Result<Vec<T>>
where T: 'a + 'b + Clone + One + Mul<Output=T> + SubAssign + MulAssign + Inv<Output=T>,
      U: Copy + IntoIterator<Item=&'a T>,
      U::IntoIter: ExactSizeIterator,
{
    let n = roots.into_iter().len();
    let mat: Vec<_> = iter::successors(
        Some(iter::repeat_with(One::one).take(n).collect::<Vec<T>>()),
        |row| Some(row.iter().cloned().zip(roots.into_iter().cloned()).map(|(x,y)| x * y).collect())
    ).take(n).collect();
    bad_linear_solve(&mat, rhs, close)
}

struct ErrorIter<I,E> {
    iter: I,
    err: Option<E>,
}

impl<'a,I,E,T> Iterator for &'a mut ErrorIter<I,E>
where
    I: 'a + Iterator<Item=core::result::Result<T,E>>,
    T: 'a,
    E: 'a,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.err {
            Some(_) => None,
            None => {
                match self.iter.next() {
                    Some(Ok(x)) => Some(x),
                    Some(Err(e)) => {
                        self.err = Some(e);
                        None
                    }
                    None => None,
                }
            }
        }
    }
}

impl<I,E> ErrorIter<I,E> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            err: None,
        }
    }

    fn map<F,T>(mut self, f: F) -> core::result::Result<T,E>
    where F: FnOnce(&mut Self) -> T
    {
        let x = f(&mut self);
        match self.err {
            Some(e) => Err(e),
            None => Ok(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_linear_algebra() {
        let very_close = RelativeParams::<f64>::default();
        {
            let mat :Vec<Vec<f64>> = vec![
                vec![2., 2., 2.],
                vec![-3., -3., -1.],
                vec![5., 3., -3.],
            ];
            let rhs = vec![12., -12., 10.];
            assert!(very_close.close_to_iter(
                bad_linear_solve(&mat, &rhs, &very_close).unwrap().iter(),
                [5., -2., 3.].iter()));
        }

        {
            let mat :Vec<Vec<f64>> = vec![
                vec![2., 8., 10., 52.],
                vec![1., 4., 5., 24.],
                vec![-2., -6., -4., -40.],
                vec![3., 11., 12., 77.],
            ];
            let rhs = vec![1., 2., 3., 4.];
            assert_eq!(bad_linear_solve(&mat, &rhs, &very_close), Err(Error::Singular));
        }

        {
            let seq = vec![1., 0., 5., -2., 12., -1.];
            assert!(very_close.close_to_iter(
                bad_berlekamp_massey(&seq, &very_close).unwrap().iter(),
                [3., 2., -1.].iter()));
        }

        {
            let roots = vec![2., -1., 3.];
            let rhs = vec![8.5, 29., 70.];
            assert!(very_close.close_to_iter(
                bad_trans_vand_solve(&roots, &rhs, &very_close).unwrap().iter(),
                [4.5,-2.,6.].iter()));
        }
    }

    #[test]
    fn add() {
        let a = vec![10, 20, 30, 40];
        let b = vec![3, 4, 5];
        let c = vec![13, 24, 35, 40];

        let ap: ClassicalPoly<_> = a.iter().copied().collect();
        let bp: ClassicalPoly<_> = b.iter().copied().collect();
        let cp: ClassicalPoly<_> = c.iter().copied().collect();

        assert_eq!(ap + bp, cp);
    }

    #[test]
    fn mul() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![300, 4, 50000];
        let c = vec![300, 604, 50908, 101212, 151516, 200020, 250000];

        let ap: ClassicalPoly<_> = a.iter().copied().collect();
        let bp: ClassicalPoly<_> = b.iter().copied().collect();
        let cp: ClassicalPoly<_> = c.iter().copied().collect();

        assert_eq!(&ap * &bp, cp);
        assert_eq!(&bp * &ap, cp);
    }

    #[test]
    fn eval() {
        type CP = ClassicalPoly<i16>;
        let f = CP::new(vec![-5,3,-1,2]);

        assert_eq!(f.eval(&-3.0f32),
            Ok((-5 + 3*-3 + -1*-3*-3 + 2*-3*-3*-3) as f32));
        {
            let pts = vec![-2,0,7];
            let prep = CP::mp_eval_prep(pts.iter().copied());
            assert!(f.mp_eval(&prep).unwrap().into_iter().eq(
                pts.iter().map(|x| f.eval(x).unwrap())));
        }
    }

    #[test]
    fn pow() {
        assert_eq!(refpow(&3u64, 0), 1);
        assert_eq!(refpow(&25u64, 1), 25);
        assert_eq!(refpow(&4u64, 2), 16);
        assert_eq!(refpow(&-5i32, 3), -125);
        assert_eq!(refpow(&2u16, 8), 256);
        let mut pow3 = 1u64;
        for d in 1..41 {
            pow3 *= 3u64;
            assert_eq!(refpow(&3u64, d), pow3);
        }
    }

    #[test]
    fn classical_sparse_interp() {
        type CP = ClassicalPoly<i32>;
        let f = CP::new(vec![3, 0, -2, 0, 0, 0, -1]);
        let t = 3;
        let (eval_info, interp_info) = CP::sparse_interp_prep(t, 0..10, &100);
        let evals = f.mp_eval(&eval_info).unwrap();
        let expected_sparse: Vec<_> = f.rep.iter().enumerate().filter(|(_,c)| **c != 0).map(|(e,c)| (e,*c)).collect();
        let sparse_f = CP::sparse_interp(&evals, &interp_info).unwrap();
        assert_eq!(expected_sparse, sparse_f);
    }
}
