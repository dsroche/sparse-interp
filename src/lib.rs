#![allow(dead_code, unused_imports)] // TODO remove

use core::{
    ops::{
        Add,
        Mul,
        AddAssign,
        SubAssign,
        MulAssign,
        DivAssign,
    },
    borrow::{
        Borrow,
        BorrowMut,
    },
    iter::{
        self,
        FromIterator,
        IntoIterator,
    },
    marker::{
        PhantomData,
    },
    convert,
    mem,
    slice,
};
use num_traits::{
    Zero,
    One,
};

pub trait PolyTraits {
    type Coeff;
    type EvalInfo;

    fn slice_mul<'a,'b,'c>(out: &'a mut [Self::Coeff], lhs: &'b [Self::Coeff], rhs: &'c [Self::Coeff]);

    fn mp_eval_prep<'a>(pts: impl Iterator<Item=&'a Self::Coeff>) -> Self::EvalInfo
    where Self::Coeff: 'a;

    fn mp_eval_slice<'a,'b,'c>(out: &'a mut Vec<Self::Coeff>,
                               coeffs: &'b [Self::Coeff], info: &'c Self::EvalInfo)
    where Self::Coeff: Zero +
            for<'d> MulAssign<&'d Self::Coeff> +
            for<'d> AddAssign<&'d Self::Coeff>,
    ;
}

#[derive(Debug,Default,PartialEq,Eq)]
pub struct ClassicalTraits<C>(PhantomData<C>);

impl<C> PolyTraits for ClassicalTraits<C>
where C: Clone + AddAssign,
      for<'a> &'a C: Mul<Output=C>,
{
    type Coeff = C;
    type EvalInfo = Vec<C>;

    #[inline(always)]
    fn slice_mul<'a,'b,'c>(out: &'a mut [Self::Coeff], lhs: &'b [Self::Coeff], rhs: &'c [Self::Coeff]) {
        classical_slice_mul(out, lhs, rhs);
    }

    fn mp_eval_prep<'a>(pts: impl Iterator<Item=&'a Self::Coeff>) -> Self::EvalInfo
    where Self::Coeff: 'a
    {
        pts.map(Clone::clone).collect()
    }

    fn mp_eval_slice<'a,'b,'c>(out: &'a mut Vec<Self::Coeff>,
                               coeffs: &'b [Self::Coeff], info: &'c Self::EvalInfo)
    where Self::Coeff: Zero +
            for<'d> MulAssign<&'d Self::Coeff> +
            for<'d> AddAssign<&'d Self::Coeff>,
    {
        out.clear();
        out.extend(info.iter().map(|x| horner_slice(coeffs, x)));
    }
}

#[derive(Debug,Default,PartialEq,Eq)]
#[repr(transparent)]
pub struct Poly<T,U> {
    rep: T,
    traits: PhantomData<U>,
}

pub type ClassicalPoly<C> = Poly<Vec<C>, ClassicalTraits<C>>;

impl<T,U> Poly<T,U> {
    pub fn new(rep: T) -> Self {
        Self {
            rep,
            traits: PhantomData,
        }
    }
}

impl<T,U> Poly<T,U>
where U: PolyTraits,
      T: Borrow<[U::Coeff]>,
      U::Coeff: Zero + Clone + for<'c> MulAssign<&'c U::Coeff> + for<'c> AddAssign<&'c U::Coeff>,
{
    #[inline(always)]
    pub fn eval(&self, x: &U::Coeff) -> U::Coeff {
        horner_slice(self.rep.borrow(), x)
    }

    #[inline]
    pub fn mp_eval(&self, info: U::EvalInfo) -> Vec<U::Coeff> {
        let mut out = Vec::new();
        U::mp_eval_slice(&mut out, self.rep.borrow(), &info);
        out
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

impl<'a,'b,T,U> Add<&'b Poly<T,U>> for &'a Poly<T,U>
where U: PolyTraits,
      U::Coeff: Clone,
      &'a U::Coeff: Add<&'b U::Coeff,Output=U::Coeff>,
      for<'c> &'c T: IntoIterator<Item=&'c U::Coeff>,
      T: FromIterator<U::Coeff>,
{
    type Output = Poly<T,U>;

    fn add(self, rhs: &'b Poly<T,U>) -> Poly<T,U> {
        Poly::new(LongZip(
                self.rep.into_iter(),
                rhs.rep.into_iter(),
                Add::add,
                Clone::clone,
                Clone::clone,
                ).collect()
            )
    }
}

impl<T,U> Add for Poly<T,U>
where U: PolyTraits,
      U::Coeff: Add<Output=U::Coeff>,
      T: IntoIterator<Item=U::Coeff> + FromIterator<U::Coeff>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(LongZip(
                self.rep.into_iter(),
                rhs.rep.into_iter(),
                Add::add,
                convert::identity,
                convert::identity,
            ).collect()
        )
    }
}

impl<'a,'b,T,U> Mul<&'b Poly<T,U>> for &'a Poly<T,U>
where U: PolyTraits,
      U::Coeff: Default,
      T: BorrowMut<[U::Coeff]> + FromIterator<U::Coeff>,
{
    type Output = Poly<T,U>;

    fn mul(self, rhs: &'b Poly<T,U>) -> Poly<T,U> {
        let aslice = self.rep.borrow();
        let bslice = rhs.rep.borrow();
        if aslice.len() == 0 || bslice.len() == 0 {
            return Poly::new(iter::empty().collect());
        }

        let clen = aslice.len() + bslice.len() - 1;
        let mut out = Poly::<T,_>::new(iter::repeat_with(Default::default).take(clen).collect());
        U::slice_mul(out.rep.borrow_mut(), aslice, bslice);
        out
    }
}


fn dot_product_in<'a,'b,'c,T>(result: &'a mut T, lhs: impl Iterator<Item=&'b T>, rhs: impl Iterator<Item=&'c T>)
where T: AddAssign + 'b + 'c,
      &'b T: Mul<&'c T, Output=T>,
{
    let mut zipit = lhs.zip(rhs);
    let (a,b) = zipit.next().expect("dot product must be with non-empty iterators");
    *result = a * b;
    for (a,b) in zipit {
        *result += a * b;
    }
}

fn classical_slice_mul<'a,'b,'c,T>(output: &'a mut [T], lhs: &'b [T], rhs: &'c [T])
where T: AddAssign + 'b + 'c,
      &'b T: Mul<&'c T, Output=T>,
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

fn horner_slice<'a,'b,T>(coeffs: &'a [T], x: &'b T) -> T
where T: Clone + Zero + MulAssign<&'b T> + for<'c> AddAssign<&'c T>,
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

// --- BAD LINEAR SOLVER, TODO REPLACE WITH BETTER "SUPERFAST" SOLVERS ---

fn bad_linear_solve<'a,M,T>(matrix: M, rhs: &[T]) -> Option<Vec<T>>
where M: IntoIterator,
      M::Item: IntoIterator<Item=&'a T>,
      T: 'a + Clone + Zero + One + SubAssign + for<'b> DivAssign<&'b T>,
      for<'b> &'b T: Mul<Output=T>,
{
    let mut workmat: Vec<Vec<_>> =
        matrix .into_iter()
            .map(|row| row.into_iter().map(Clone::clone).collect())
            .collect();
    let mut sol: Vec<_> = rhs.iter().map(Clone::clone).collect();

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
        sol[i] /= &workmat[i][i];
        {
            let (left, right) = workmat[i].split_at_mut(i+1);
            for x in right {
                *x /= &left[i];
            }
            left[i].set_one();
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
where T: Clone + Zero + One + SubAssign + for<'b> DivAssign<&'b T>,
      for<'b> &'b T: Mul<Output=T>,
{
    assert_eq!(seq.len() % 2, 0);
    let n = seq.len() / 2;
    bad_linear_solve(
        (0..n).map(|i| &seq[i..i+n]),
        &seq[n..2*n])
}

fn bad_trans_vand_solve<T>(roots: &[T], rhs: &[T]) -> Option<Vec<T>>
where T: Clone + Zero + One + SubAssign + for<'c> DivAssign<&'c T>,
      for<'c> &'c T: Mul<Output=T>,
{
    let n = roots.len();
    assert_eq!(rhs.len(), n);
    let mat: Vec<_> = iter::successors(
        Some(iter::repeat_with(One::one).take(n).collect::<Vec<T>>()),
        |row| Some(row.iter().zip(roots.iter()).map(|(x,y)| x*y).collect())
    ).take(n).collect();
    bad_linear_solve(&mat, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_linear_algebra() {
        {
            let mat :Vec<Vec<f64>> = vec![
                vec![2., 2., 2.],
                vec![-3., -3., -1.],
                vec![5., 3., -3.],
            ];
            let rhs = vec![12., -12., 10.];
            assert_eq!(bad_linear_solve(&mat, &rhs), Some(vec![5., -2., 3.]));
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
            assert_eq!(bad_berlekamp_massey(&seq), Some(vec![3., 2., -1.]));
        }

        {
            let roots = vec![2., -1., 3.];
            let rhs = vec![8.5, 29., 70.];
            assert_eq!(bad_trans_vand_solve(&roots, &rhs), Some(vec![4.5,-2.,6.]));
        }
    }

    #[test]
    fn add() {
        let a = vec![10, 20, 30, 40];
        let b = vec![3, 4, 5];
        let c = vec![13, 24, 35, 40];

        let ap = ClassicalPoly::new(a);
        let bp = ClassicalPoly::new(b);
        let cp = ClassicalPoly::new(c);

        assert_eq!(&ap + &bp, cp);
        assert_eq!(&bp + &ap, cp);
        assert_eq!(ap + bp, cp);
    }

    #[test]
    fn mul() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![300, 4, 50000];
        let c = vec![300, 604, 50908, 101212, 151516, 200020, 250000];

        let ap = ClassicalPoly::new(a);
        let bp = ClassicalPoly::new(b);
        let cp = ClassicalPoly::new(c);

        assert_eq!(&ap * &bp, cp);
        assert_eq!(&bp * &ap, cp);
    }

    #[test]
    fn eval() {
        let f = ClassicalPoly::new(vec![-5,3,-1,2]);
        assert_eq!(f.eval(&-3),
            -5 + 3*-3 + -1*-3*-3 + 2*-3*-3*-3);
        {
            let pts = vec![-2,0,7];
            let info = ClassicalTraits::mp_eval_prep(pts.iter());
            assert!(f.mp_eval(info).into_iter().eq(
                pts.iter().map(|x| f.eval(x))));
        }
    }
}
