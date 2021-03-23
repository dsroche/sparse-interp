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
//! let h = ClassicalPoly::<f32>::new(vec![4., 2., 3., -1.]);
//! assert_eq!(h.eval(&0.), 4.);
//! assert_eq!(h.eval(&1.), 8.);
//! assert_eq!(h.eval(&-1.), 6.);
//! assert_eq!(h.mp_eval([0.,1.,-1.].iter()), [4.,8.,6.]);
//! ```
//!
//! If the same evaluation points are used for multiple polynomials,
//! they can be preprocessed with [`Poly::mp_eval_prep()`], and then
//! replacing [`Poly::mp_eval()`] with [`Poly::mp_eval_post()`] will
//! be more efficient overall.
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
//! let f = ClassicalPoly::new(vec![0., -2.5, 0., 0., 0., 7.1]);
//! let t = 2;
//! let theta = 1.8f64;
//! let eval_pts = [1., theta, theta.powi(2), theta.powi(3)];
//! let evals = f.mp_eval(eval_pts.iter());
//! let error = 0.001;
//! let mut result = ClassicalPoly::sparse_interp(
//!     &theta,    // evaluation base point
//!     t,         // upper bound on nonzero terms
//!     0..8,      // iteration over possible exponents
//!     &evals,    // evaluations at powers of theta
//!     &RelativeParams::<f64>::new(Some(error), Some(error))
//!                // needed for approximate types like f64
//! ).unwrap();
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
        Neg,
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
    convert,
    slice,
};
use num_traits::{
    Zero,
    One,
    Inv,
};

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
    ($T:ident, $E:ident) => {
        impl RelativeParams<$T,$E> {
            /// Create a new closeness tester with the given parameters.
            ///
            /// For both arguments, `zero_thresh` and `max_relative`,
            /// either the given bound is used, or machine epsilon if the argument
            /// is `None`.
            #[inline(always)]
            pub fn new(zero_thresh: Option<$E>, max_relative: Option<$E>) -> Self {
                Self {
                    zero_thresh: zero_thresh.unwrap_or($E::EPSILON),
                    max_relative: max_relative.unwrap_or($E::EPSILON),
                    phantom: PhantomData,
                }
            }
        }

        impl Default for RelativeParams<$T,$E> {
            /// Create a closeness tester with machine epsilon precision.
            #[inline(always)]
            fn default() -> Self {
                Self::new(None, None)
            }
        }

        impl CloseTo for RelativeParams<$T,$E> {
            type Item = $T;

            #[inline]
            fn close_to(&self, x: &Self::Item, y: &Self::Item) -> bool {
                if x == y {
                    return true
                };
                if $T::is_infinite(*x) || $T::is_infinite(*y) {
                    return false
                };

                let absx = $T::abs(*x);
                let absy = $T::abs(*y);

                let largest = if absx <= absy {
                    absy
                } else {
                    absx
                };

                if largest <= self.zero_thresh {
                    true
                } else {
                    $T::abs(x - y) / largest <= self.max_relative
                }
            }

            #[inline(always)]
            fn close_to_zero(&self, x: &Self::Item) -> bool {
                $T::abs(*x) <= self.zero_thresh
            }
        }
    };
}

impl_relative_params!(f32, f32);
impl_relative_params!(f64, f64);

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

    /// An opaque type returned by the pre-processing method [`Self::mp_eval_prep()`].
    type EvalInfo;

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
    fn mp_eval_prep<'a>(pts: impl Iterator<Item=&'a Self::Coeff>) -> Self::EvalInfo
    where Self::Coeff: 'a;

    /// Multi-point evaluation.
    ///
    /// Evaluates the polynomial (given by a slice of coefficients) at all points
    /// specified in a previous call to [`Self::mp_eval_prep()`].
    ///
    /// ```
    /// # use sparse_interp::*;
    /// # type TraitImpl = ClassicalTraits<f32>;
    /// let pts = [10., -5.];
    /// let preprocess = TraitImpl::mp_eval_prep(pts.iter());
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
    fn mp_eval_slice(out: &mut impl Extend<Self::Coeff>, coeffs: &[Self::Coeff], info: &Self::EvalInfo);

    /// Pre-processing for sparse interpolation.
    ///
    /// This method must be called prior to
    /// calling [`Self::mp_eval_slice()`].
    ///
    /// A later call to sparse interpolation is guaranteed to succeed only when, for some
    /// unknown polynomial `f`, the following are all true:
    /// *   `f` has at most `sparsity` non-zero terms
    /// *   The exponents of all non-zero terms of `f` appear in `expons`
    /// *   `f` is evaluated at points `pow(theta, i)` for all `i` in `[0..2*sparsity]`.
    ///
    /// The list `expons` must be sorted in ascending order.
    ///
    /// The same pre-processed output can be used repeatedly to
    /// interpolate possibly different polynomials under the same settings.
    fn sparse_interp_prep(theta: &Self::Coeff, sparsity: usize, expons: impl Iterator<Item=usize>)
        -> Self::SparseInterpInfo;

    /// Sparse interpolation from special evaluation points.
    ///
    /// The points in `eval` should correspond to the requirements in
    /// [`Self::sparse_interp_prep()`].
    ///
    /// In addition the provided [`CloseTo`] impl works for the underlying sparse interpolation
    /// algorithm. For exact fields, [`CloseToEq`] should always work. For inexact
    /// fields such as [`f64`], [`RelativeParams`] should work, but some understanding
    /// of the trait's sparse interpolation algorithm may be required to ensure accuracy.
    ///
    /// If those requirements are met, the function will return Some(..) containing
    /// a list of exponent-coefficient pairs, sorted in ascending order of exponents.
    ///
    /// Otherwise, for example if the evaluated function has more non-zero terms
    /// than the pre-specified limit, this function *may* return None or may return
    /// Some(..) with incorrect values.
    ///
    /// ```
    /// # use sparse_interp::*;
    /// # type TraitImpl = ClassicalTraits<f32>;
    /// let f :[f32; 8] = [0., 0., -18.5, 0., 0., 0., 0., 31.7];
    /// let theta: f32 = 2.1;
    /// let eval_pts :Vec<_> = (0..4).map(|d| theta.powi(d)).collect();
    /// let evals :Vec<_> = eval_pts.iter().map(|x| f[2] * x.powi(2) + f[7] * x.powi(7)).collect();
    /// let close_check = RelativeParams::<f32>::new(Some(0.1), Some(0.1));
    ///
    /// let sp_info = TraitImpl::sparse_interp_prep(&theta, 2, 0..10);
    /// let sp_result = TraitImpl::sparse_interp_slice(&evals, &sp_info, &close_check).unwrap();
    /// assert_eq!(sp_result.len(), 2);
    /// assert_eq!(sp_result[0].0, 2);
    /// assert!(close_check.close_to(&sp_result[0].1, &f[2]));
    /// assert_eq!(sp_result[1].0, 7);
    /// assert!(close_check.close_to(&sp_result[1].1, &f[7]));
    /// ```
    fn sparse_interp_slice(evals: &[Self::Coeff], info: &Self::SparseInterpInfo, close: &impl CloseTo<Item=Self::Coeff>)
        -> Option<Vec<(usize, Self::Coeff)>>;
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

impl<C> PolyTraits for ClassicalTraits<C>
where C: Clone + Zero + One + Neg<Output=C> + AddAssign + SubAssign
        + for<'a> MulAssign<&'a C> + for<'a> AddAssign<&'a C> + for<'a> MulAssign<&'a C>,
      for<'a> &'a C: Mul<Output=C> + Inv<Output=C>,
{
    type Coeff = C;
    type EvalInfo = Vec<C>;
    type SparseInterpInfo = (usize, Vec<(usize, C)>);

    #[inline(always)]
    fn slice_mul(out: &mut [Self::Coeff], lhs: &[Self::Coeff], rhs: &[Self::Coeff]) {
        classical_slice_mul(out, lhs, rhs);
    }

    #[inline]
    fn mp_eval_prep<'a>(pts: impl Iterator<Item=&'a Self::Coeff>) -> Self::EvalInfo
    where Self::Coeff: 'a
    {
        pts.map(Clone::clone).collect()
    }

    #[inline]
    fn mp_eval_slice(out: &mut impl Extend<Self::Coeff>, coeffs: &[Self::Coeff], info: &Self::EvalInfo) {
        out.extend(info.iter().map(|x| horner_slice(coeffs, x)));
    }

    fn sparse_interp_prep(theta: &Self::Coeff, sparsity: usize, expons: impl Iterator<Item=usize>)
        -> Self::SparseInterpInfo
    {
        let theta_pows: Vec<_> = expons.map(|d| (d, refpow(theta, d))).collect();
        assert!(sparsity <= theta_pows.len());
        (sparsity, theta_pows)
    }

    fn sparse_interp_slice(evals: &[Self::Coeff], info: &Self::SparseInterpInfo, close: &impl CloseTo<Item=Self::Coeff>)
        -> Option<Vec<(usize, Self::Coeff)>>
    {
        assert_eq!(evals.len(), 2*info.0);
        for k in (1..=info.0).rev() {
            if let Some(mut lambda) = bad_berlekamp_massey(&evals[..2*k]) {
                lambda.push(-C::one());
                let (degs, roots): (Vec<usize>, Vec<&C>) = info.1.iter().filter_map(
                    |(deg, rootpow)| match close.close_to_zero(&horner_slice(&lambda, rootpow)) {
                        true => Some((deg, rootpow)),
                        false => None,
                    }).unzip();
                if degs.len() == lambda.len() - 1 {
                    // Note, seems dumb I had to do this just to turn &&C into &C, but alas...
                    struct IterHolder<'a,T>(Vec<&'a T>);
                    impl<'a,T> IntoIterator for &'a IterHolder<'a, T> {
                        type Item = &'a T;
                        type IntoIter = iter::Copied<slice::Iter<'a, &'a T>>;
                        fn into_iter(self) -> Self::IntoIter {
                            self.0.iter().copied()
                        }
                    }
                    if let Some(coeffs) = bad_trans_vand_solve(&IterHolder(roots), &evals[..degs.len()]) {
                        return Some(degs.into_iter().zip(coeffs.into_iter()).collect());
                    }
                }
            }
        }
        None
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
      U::Coeff: Clone + Zero + for<'c> MulAssign<&'c U::Coeff> + for<'c> AddAssign<&'c U::Coeff>,
{
    /// Evaluate this polynomial at the given point.
    ///
    /// Uses Horner's rule to perform the evaluation using exactly d multiplications
    /// and d additions, where d is the degree of self.
    #[inline(always)]
    pub fn eval(&self, x: &U::Coeff) -> U::Coeff {
        horner_slice(self.rep.borrow(), x)
    }

    /// Perform pre-processing for multi-point evaluation.
    ///
    /// `pts` is an iterator over the desired evaluation points.
    ///
    /// See [`PolyTraits::mp_eval_prep()`] for more details.
    #[inline(always)]
    pub fn mp_eval_prep<'a>(pts: impl Iterator<Item=&'a U::Coeff>) -> U::EvalInfo
    where U::Coeff: 'a,
    {
        U::mp_eval_prep(pts)
    }

    /// Perform multi-point evaluation after pre-processing.
    ///
    /// `info` should be the result of calling [`Self::mp_eval_prep()`].
    #[inline]
    pub fn mp_eval_post(&self, info: &U::EvalInfo) -> Vec<U::Coeff> {
        let mut out = Vec::new();
        U::mp_eval_slice(&mut out, self.rep.borrow(), info);
        out
    }

    /// Evaluate this polynomial at all of the given points.
    ///
    /// Performs multi-point evaluation using the underlying trait algorithms.
    /// In general, this can be much more efficient than repeatedly calling
    /// [`self.eval()`].
    #[inline(always)]
    pub fn mp_eval<'a>(&self, pts: impl Iterator<Item=&'a U::Coeff>) -> Vec<U::Coeff>
    where U::Coeff: 'a,
    {
        self.mp_eval_post(&Self::mp_eval_prep(pts))
    }
}

impl<T,U> Poly<T,U>
where U: PolyTraits,
{
    /// Perform pre-processing for sparse interpolation.
    ///
    /// *   Evaluations will be at consecutive powers of `theta`.
    /// *   `sparsity` is an upper bound on the number of nonzero terms in the evaluated
    ///     polynomial.
    /// *   `expons` is an iteration over the possible exponents which may appear in nonzero
    ///     terms.
    ///
    /// See [`PolyTraits::sparse_interp_prep()`] for more details.
    #[inline(always)]
    pub fn sparse_interp_prep(theta: &U::Coeff, sparsity: usize, expons: impl Iterator<Item=usize>)
        -> U::SparseInterpInfo
    {
        U::sparse_interp_prep(theta, sparsity, expons)
    }

    /// Perform sparse interpolation after pre-processing.
    ///
    /// *   `evals` should correspond to evaluations of some unknown power
    ///     at consecutive powers of the `theta` used in preprocessing.
    /// *   `info` should be the result of calling [`Self::sparse_interp_prep()`].
    /// *   The parameters for `close` depend on the underlying field and the trait's interpolation
    ///     algorithm.
    ///
    /// On success, a vector of (exponent, nonzero coefficient) pairs is returned,
    /// sorted by increasing exponent values.
    #[inline(always)]
    pub fn sparse_interp_post(evals: &[U::Coeff], info: &U::SparseInterpInfo, close: &impl CloseTo<Item=U::Coeff>)
        -> Option<Vec<(usize, U::Coeff)>>
    {
        U::sparse_interp_slice(evals, info, close)
    }

    /// Perform sparse interpolation.
    ///
    /// *   Evaluations will be at consecutive powers of `theta`.
    /// *   `sparsity` is an upper bound on the number of nonzero terms in the evaluated
    ///     polynomial.
    /// *   `expons` is an iteration over the possible exponents which may appear in nonzero
    ///     terms.
    /// *   `evals` should correspond to evaluations of some unknown power
    ///     at consecutive powers of the `theta` used in preprocessing.
    /// *   The parameters for `close` depend on the underlying field and the trait's interpolation
    ///     algorithm.
    ///
    /// On success, a vector of (exponent, nonzero coefficient) pairs is returned,
    /// sorted by increasing exponent values.
    /// #[inline(always)]
    pub fn sparse_interp(
        theta: &U::Coeff,
        sparsity: usize,
        expons: impl Iterator<Item=usize>,
        evals: &[U::Coeff],
        close: &impl CloseTo<Item=U::Coeff>)
        -> Option<Vec<(usize, U::Coeff)>>
    {
        Self::sparse_interp_post(evals, &U::sparse_interp_prep(theta, sparsity, expons), close)
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

fn dot_product_in<T,U,V>(result: &mut T, lhs: U, rhs: V)
where T: AddAssign,
      U: Iterator,
      V: Iterator,
      U::Item: Mul<V::Item, Output=T>,
{
    let mut zipit = lhs.zip(rhs);
    let (a,b) = zipit.next().expect("dot product must be with non-empty iterators");
    *result = a * b;
    for (a,b) in zipit {
        *result += a * b;
    }
}

fn classical_slice_mul<T>(output: &mut [T], lhs: &[T], rhs: &[T])
where T: AddAssign,
      for<'a> &'a T: Mul<Output=T>,
{
    if lhs.len() == 0 || rhs.len() == 0 {
        assert_eq!(output.len(), 0);
        return;
    }

    assert_eq!(lhs.len() + rhs.len(), output.len() + 1);

    let mut i = 0;

    while i < lhs.len() && i < rhs.len() {
        dot_product_in(&mut output[i], lhs[..i+1].iter(), rhs[..i+1].iter().rev());
        i = i + 1;
    }

    let mut j = 1;
    while i < lhs.len() {
        dot_product_in(&mut output[i], lhs[j..i+1].iter(), rhs.iter().rev());
        i = i + 1;
        j = j + 1;
    }

    let mut k = 1;
    while i < rhs.len() {
        dot_product_in(&mut output[i], lhs.iter(), rhs[k..i+1].iter().rev());
        i = i + 1;
        k = k + 1;
    }

    while i < output.len() {
        dot_product_in(&mut output[i], lhs[j..].iter(), rhs[k..].iter().rev());
        i = i + 1;
        j = j + 1;
        k = k + 1;
    }
}

fn horner_slice<T>(coeffs: &[T], x: &T) -> T
where T: Clone + Zero + for<'c> MulAssign<&'c T> + for<'c> AddAssign<&'c T>,
{
    let mut coeffs = coeffs.iter().rev();
    match coeffs.next() {
        Some(leading) => {
            let mut out = leading.clone();
            for coeff in coeffs {
                out *= x;
                out += &coeff;
            }
            out
        },
        None => T::zero(),
    }
}

// see also: https://docs.rs/num-traits/0.2.14/src/num_traits/pow.rs.html#189-211
#[inline]
fn refpow<T>(base: &T, exp: usize) -> T
where T: Clone + One + for<'a> MulAssign<&'a T>,
      for<'a> &'a T: Mul<Output=T>,
{
    match exp {
        0 => T::one(),
        1 => base.clone(),
        _ => {
            let mut acc = base * base;
            let mut curexp = exp >> 1;

            // invariant: curexp > 0 and base^exp = acc^curexp * base^(exp mod 2)
            while curexp & 1 == 0 {
                acc = &acc * &acc;
                curexp >>= 1;
            }
            // now: curexp positive odd, base^exp = acc^curexp * base^(exp mod 2)

            if curexp > 1 {
                let mut basepow = &acc * &acc;
                curexp >>= 1;

                // invariant: curexp > 0 and base^exp = acc * basepow^curexp * base^(exp mod 2)
                while curexp > 1 {
                    if curexp & 1 == 1 {
                        acc *= &basepow;
                    }
                    basepow = &basepow * &basepow;
                    curexp >>= 1;
                }
                // now: curexp == 1 and base^exp = acc * basepow * base^(exp mod 2)

                acc *= &basepow;
            }
            // now: curexp == 1 and base^exp = acc * base^(exp mod 2)

            if exp & 1 == 1 {
                acc *= &base;
            }
            acc
        }
    }
}


// --- BAD LINEAR SOLVER, TODO REPLACE WITH BETTER "SUPERFAST" SOLVERS ---

fn bad_linear_solve<'a,'b,M,T>(matrix: M, rhs: impl IntoIterator<Item=&'b T>) -> Option<Vec<T>>
where M: IntoIterator,
      M::Item: IntoIterator<Item=&'a T>,
      T: 'a + 'b + Clone + Zero + One + SubAssign + for<'x> MulAssign<&'x T>,
      for<'x> &'x T: Mul<Output=T> + Inv<Output=T>,
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
            while workmat[j][i].is_zero() {
                j += 1;
                if j == n {
                    return None;
                }
            }
            if i != j {
                workmat.swap(i, j);
                sol.swap(i, j);
            }
        }
        // normalize pivot row
        {
            let pivot_inverse = (&workmat[i][i]).inv();
            sol[i] *= &pivot_inverse;
            for x in workmat[i].split_at_mut(i+1).1 {
                *x *= &pivot_inverse;
            }
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
                if !row[i].is_zero() {
                    let (left, right) = row.split_at_mut(i+1);
                    for (j, x) in (i+1..n).zip(right) {
                        *x -= &left[i] * &pivrow[j];
                    }
                    *solx -= &left[i] * solpiv;
                }
            }
        }
    }
    Some(sol)
}

fn bad_berlekamp_massey<T>(seq: &[T]) -> Option<Vec<T>>
where T: Clone + Zero + One + SubAssign + for<'b> MulAssign<&'b T>,
      for<'b> &'b T: Mul<Output=T> + Inv<Output=T>,
{
    assert_eq!(seq.len() % 2, 0);
    let n = seq.len() / 2;
    bad_linear_solve(
        (0..n).map(|i| &seq[i..i+n]),
        &seq[n..2*n])
}

fn bad_trans_vand_solve<'a,'b,T,U>(roots: U, rhs: impl IntoIterator<Item=&'b T>)
    -> Option<Vec<T>>
where T: 'a + 'b + Clone + Zero + One + SubAssign + for<'c> MulAssign<&'c T>,
      for<'c> &'c T: Mul<Output=T> + Inv<Output=T>,
      U: Copy + IntoIterator<Item=&'a T>,
      U::IntoIter: ExactSizeIterator,
{
    let n = roots.into_iter().len();
    let mat: Vec<_> = iter::successors(
        Some(iter::repeat_with(One::one).take(n).collect::<Vec<T>>()),
        |row| Some(row.iter().zip(roots.into_iter()).map(|(x,y)| x*y).collect())
    ).take(n).collect();
    bad_linear_solve(&mat, rhs)
}

#[cfg(test)]
mod tests {
    use num_rational::{
        Rational32,
    };
    use super::*;

    #[test]
    fn bad_linear_algebra() {
        let close = RelativeParams::<f64>::default();
        {
            let mat :Vec<Vec<f64>> = vec![
                vec![2., 2., 2.],
                vec![-3., -3., -1.],
                vec![5., 3., -3.],
            ];
            let rhs = vec![12., -12., 10.];
            assert!(close.close_to_iter(
                bad_linear_solve(&mat, &rhs).unwrap().iter(),
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
            assert_eq!(bad_linear_solve(&mat, &rhs), None);
        }

        {
            let seq = vec![1., 0., 5., -2., 12., -1.];
            assert!(close.close_to_iter(
                bad_berlekamp_massey(&seq).unwrap().iter(),
                [3., 2., -1.].iter()));
        }

        {
            let roots = vec![2., -1., 3.];
            let rhs = vec![8.5, 29., 70.];
            assert!(close.close_to_iter(
                bad_trans_vand_solve(&roots, &rhs).unwrap().iter(),
                [4.5,-2.,6.].iter()));
        }
    }

    #[test]
    fn add() {
        let a: Vec<_> = [10, 20, 30, 40]
            .iter().copied().map(Rational32::from_integer).collect();
        let b: Vec<_> = [3, 4, 5]
            .iter().copied().map(Rational32::from_integer).collect();
        let c: Vec<_> = [13, 24, 35, 40]
            .iter().copied().map(Rational32::from_integer).collect();

        let ap: ClassicalPoly<_> = a.iter().copied().collect();
        let bp: ClassicalPoly<_> = b.iter().copied().collect();
        let cp: ClassicalPoly<_> = c.iter().copied().collect();

        assert_eq!(ap + bp, cp);
    }

    #[test]
    fn mul() {
        let a: Vec<_> = [1, 2, 3, 4, 5]
            .iter().copied().map(Rational32::from_integer).collect();
        let b: Vec<_> = [300, 4, 50000]
            .iter().copied().map(Rational32::from_integer).collect();
        let c: Vec<_> = [300, 604, 50908, 101212, 151516, 200020, 250000]
            .iter().copied().map(Rational32::from_integer).collect();

        let ap: ClassicalPoly<_> = a.iter().copied().collect();
        let bp: ClassicalPoly<_> = b.iter().copied().collect();
        let cp: ClassicalPoly<_> = c.iter().copied().collect();

        assert_eq!(&ap * &bp, cp);
        assert_eq!(&bp * &ap, cp);
    }

    #[test]
    fn eval() {
        let f: ClassicalPoly<_> = [-5,3,-1,2].iter().copied().map(Rational32::from_integer).collect();

        assert_eq!(f.eval(&Rational32::from_integer(-3)),
            Rational32::from_integer(-5 + 3*-3 + -1*-3*-3 + 2*-3*-3*-3));
        {
            let pts: Vec<_> = [-2,0,7].iter().copied().map(Rational32::from_integer).collect();
            assert!(f.mp_eval(pts.iter()).into_iter().eq(
                pts.iter().map(|x| f.eval(x))));
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

    // XXX available in nightly as part of Iterator
    fn eq_by<T,U,F>(mut a: T, mut b: U, mut eq: F) -> bool
    where T: Iterator,
          U: Iterator,
          F: FnMut(T::Item, U::Item) -> bool,
    {
        loop {
            match a.next() {
                Some(x) => match b.next() {
                    Some(y) => if ! eq(x,y) {
                        return false;
                    },
                    None => return false,
                },
                None => return b.next().is_none(),
            };
        }
    }

    #[test]
    fn classical_sparse_interp_exact() {
        let f: ClassicalPoly<_> = vec![3., 0., -2., 0., 0., 0., -1.].into_iter().collect();
        let theta = 1.2f64;
        let t = 3;
        let xs: Vec<_> = (0..2*t).map(|i| theta.powi(i as i32)).collect();
        let ys = f.mp_eval(xs.iter());
        let eq_test = RelativeParams::<f64>::new(Some(0.00000001), Some(0.00000001));
        let expected_sparse = f.rep.iter().enumerate().filter(|(_,c)| **c != 0.);
        // sparse interpolation starts here
        let si_info = ClassicalPoly::sparse_interp_prep(&theta, t, 0..10);
        let sparse_f = ClassicalPoly::sparse_interp_post(&ys, &si_info, &eq_test).expect("sparse interp failed");
        assert!(eq_by(sparse_f.iter(), expected_sparse,
            |(d1,c1), (d2,c2)| *d1 == d2 && eq_test.close_to(c1, c2)
            ));
    }

    #[test]
    fn classical_sparse_interp_overshoot() {
        let f: ClassicalPoly<_> = vec![3., 0., -2., 0., 0., 0., -1.].into_iter().collect();
        let theta = 1.2f64;
        let t = 5;
        let xs: Vec<_> = (0..2*t).map(|i| theta.powi(i as i32)).collect();
        let ys = f.mp_eval(xs.iter());
        let eq_test = RelativeParams::<f64>::new(Some(0.0000001), Some(0.0000001));
        let expected_sparse = f.rep.iter().enumerate().filter(|(_,c)| **c != 0.);
        // sparse interpolation starts here
        let si_info = ClassicalPoly::sparse_interp_prep(&theta, t, 0..10);
        let sparse_f = ClassicalPoly::sparse_interp_post(&ys, &si_info, &eq_test).expect("sparse interp failed");
        assert!(eq_by(sparse_f.iter(), expected_sparse,
            |(d1,c1), (d2,c2)| *d1 == d2 && eq_test.close_to(c1, c2)
            ));
    }
}
