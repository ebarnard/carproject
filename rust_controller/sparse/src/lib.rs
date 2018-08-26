extern crate itertools;
extern crate nalgebra;

mod erased_into_iter;

use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Dynamic as Dy, MatrixMN, MatrixSlice};
use std::convert::AsRef;
use std::iter::once;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Mul, Neg};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

use erased_into_iter::{Erased, ErasedIntoIter};

#[allow(non_camel_case_types)]
type float = f64;

pub struct BlockRef<M, N> {
    id: usize,
    nrows: usize,
    ncols: usize,
    phantom: PhantomData<(M, N)>,
}

#[derive(Clone, Debug)]
pub struct Builder {
    tracked_blocks: Vec<(usize, usize, usize, MatrixMN<bool, Dy, Dy>)>,
    coords: Vec<(usize, usize, float)>,
    nrows: usize,
    ncols: usize,
}

impl Builder {
    pub fn with_capacity(nrows: usize, ncols: usize, nnz: usize) -> Builder {
        Builder {
            tracked_blocks: Vec::new(),
            coords: Vec::with_capacity(nnz),
            nrows,
            ncols,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn build_csc(&mut self) -> CscMatrix {
        // Sort and sum any duplicates in the same cell
        let mut coords = mem::replace(&mut self.coords, Vec::new());
        coords.sort_unstable_by_key(|&(r, c, _)| (c, r));
        self.coords = coords
            .into_iter()
            .coalesce(|l, r| {
                if l.0 == r.0 && l.1 == r.1 {
                    Ok((l.0, l.1, l.2 + r.2))
                } else {
                    Err((l, r))
                }
            }).collect();

        let mut indptr = vec![0; self.ncols + 1];
        let mut indices = vec![0; self.coords.len()];
        let mut data = vec![0.0; self.coords.len()];

        // Fill in CSC column start and end pointer and row index
        let mut last_c = 0;
        for (i, &(r, c, val)) in self.coords.iter().enumerate() {
            while last_c < c {
                last_c += 1;
                indptr[last_c] = i;
            }
            indices[i] = r;
            data[i] = val;
        }
        // Set the remaining column index pointers to one past the end of the data array
        while last_c < self.ncols {
            last_c += 1;
            indptr[last_c] = self.coords.len();
        }

        // Calculate the tracked block indices
        let mut tracked_blocks = self
            .tracked_blocks
            .iter()
            .map(|&(id, row, col, ref sparsity)| {
                let (nrows, ncols) = sparsity.shape();
                let mut block_indices = MatrixMN::from_element_generic(dy(nrows), dy(ncols), None);

                for c in 0..ncols {
                    let rows_start = indptr[col + c];
                    let rows_end = indptr[col + c + 1];
                    let row_indices = &indices[rows_start..rows_end];

                    let mut first_row_index = None;
                    let mut sparse_r = 0;

                    // Ignore entries that are always empty
                    for r in (0..nrows).filter(|&r| sparsity[(r, c)]) {
                        // row_indices are guarenteed to be in ascending order
                        let first_row_index = *first_row_index.get_or_insert_with(|| {
                            rows_start + row_indices.binary_search(&(row + r)).unwrap()
                        });
                        block_indices[(r, c)] = Some(first_row_index + sparse_r);
                        sparse_r += 1;
                    }
                }

                (id, block_indices)
            }).collect::<Vec<_>>();

        tracked_blocks.sort_unstable_by_key(|b| b.0);

        CscMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            indptr,
            indices,
            data,
            tracked_blocks,
        }
    }
}

impl AsRef<Builder> for Builder {
    fn as_ref(&self) -> &Builder {
        self
    }
}

impl Add for Builder {
    type Output = Builder;

    fn add(self, other: Builder) -> Builder {
        add(&[self, other])
    }
}

impl<'a> Add<&'a Builder> for Builder {
    type Output = Builder;

    fn add(self, other: &'a Builder) -> Builder {
        add(&[&self, other])
    }
}

impl<'a> Add<Builder> for &'a Builder {
    type Output = Builder;

    fn add(self, other: Builder) -> Builder {
        add(&[self, &other])
    }
}

impl<'a, 'b> Add<&'b Builder> for &'a Builder {
    type Output = Builder;

    fn add(self, other: &'b Builder) -> Builder {
        add(&[self, other])
    }
}

impl Mul<float> for Builder {
    type Output = Builder;

    fn mul(mut self, other: float) -> Builder {
        for &mut (_, _, ref mut val) in &mut self.coords {
            *val = other * (*val);
        }
        self
    }
}

impl<'a> Mul<float> for &'a Builder {
    type Output = Builder;

    fn mul(self, other: float) -> Builder {
        let cloned: Builder = self.clone();
        cloned * other
    }
}

impl Neg for Builder {
    type Output = Builder;

    fn neg(mut self) -> Builder {
        for &mut (_, _, ref mut val) in &mut self.coords {
            *val = -*val;
        }
        self
    }
}

impl<'a> Neg for &'a Builder {
    type Output = Builder;

    fn neg(self) -> Builder {
        let cloned: Builder = self.clone();
        -cloned
    }
}

pub fn zeros(nrows: usize, ncols: usize) -> Builder {
    Builder::with_capacity(nrows, ncols, 0)
}

pub fn eye(n: usize) -> Builder {
    let mut builder = Builder::with_capacity(n, n, n);
    for i in 0..n {
        builder.coords.push((i, i, 1.0));
    }
    builder
}

pub fn diags(n: usize, vals: &[float], diag: &[isize]) -> Builder {
    let cap = diag.iter().map(|&d| n - d.abs() as usize).sum();
    let mut builder = Builder::with_capacity(n, n, cap);

    for (&k, &v) in diag.iter().zip(vals) {
        assert!((k.abs() as usize) < n);
        if k >= 0 {
            let k = k as usize;
            for i in 0..(n - k) {
                builder.coords.push((i, i + k, v));
            }
        } else if k < 0 {
            let k = -k as usize;
            for i in 0..(n - k) {
                builder.coords.push((i + k, i, v));
            }
        }
    }
    builder
}

pub fn block<M: Dim, N: Dim>(block: &MatrixMN<float, M, N>) -> Builder
where
    DefaultAllocator: Allocator<float, M, N>,
{
    let (nrows, ncols) = block.shape();

    let coords = (0..nrows)
        .flat_map(move |r| (0..ncols).map(move |c| (r, c, block[(r, c)])))
        .filter(|&(_, _, val)| val != 0.0)
        .collect();

    Builder {
        tracked_blocks: Vec::new(),
        coords,
        nrows,
        ncols,
    }
}

pub fn block_mut<M: Dim, N: Dim>(sparsity: &MatrixMN<bool, M, N>) -> (Builder, BlockRef<M, N>)
where
    DefaultAllocator: Allocator<bool, M, N>,
{
    static NEXT_ID: AtomicUsize = ATOMIC_USIZE_INIT;

    let (nrows, ncols) = sparsity.shape();

    let coords = (0..nrows)
        .flat_map(move |r| {
            (0..ncols)
                .filter(move |&c| sparsity[(r, c)])
                .map(move |c| (r, c, 0.0))
        }).collect();

    let sparsity = to_dynamic(sparsity);

    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

    (
        Builder {
            tracked_blocks: vec![(id, 0, 0, sparsity)],
            coords,
            nrows,
            ncols,
        },
        BlockRef {
            id,
            nrows,
            ncols,
            phantom: PhantomData,
        },
    )
}

pub fn add<B: AsRef<Builder>>(blocks: &[B]) -> Builder {
    fn op(acc: &mut Builder, nrows: &mut usize, ncols: &mut usize, block: &Builder) {
        assert_eq!(
            *nrows, block.nrows,
            "matrices being added must have the same number of rows"
        );
        assert_eq!(
            *ncols, block.ncols,
            "matrices being added must have the same number of columns"
        );
        block_merge(acc, block, 0, 0);
    }

    let (nrows, ncols) = blocks
        .get(0)
        .map(AsRef::as_ref)
        .map(|b| (b.nrows, b.ncols))
        .unwrap_or((0, 0));
    merge_op(&mut erase_iter(blocks), nrows, ncols, op)
}

pub fn hstack<B: AsRef<Builder>>(blocks: &[B]) -> Builder {
    fn op(acc: &mut Builder, nrows: &mut usize, ncols: &mut usize, block: &Builder) {
        assert_eq!(
            *nrows, block.nrows,
            "vstack requires matrices to have the same number of rows"
        );
        block_merge(acc, block, 0, *ncols);
        *ncols += block.ncols;
    }

    let nrows = blocks.get(0).map(|b| b.as_ref().nrows).unwrap_or(0);
    merge_op(&mut erase_iter(blocks), nrows, 0, op)
}

pub fn vstack<B: AsRef<Builder>>(blocks: &[B]) -> Builder {
    fn op(acc: &mut Builder, nrows: &mut usize, ncols: &mut usize, block: &Builder) {
        assert_eq!(
            *ncols, block.ncols,
            "vstack requires matrices to have the same number of columns"
        );
        block_merge(acc, block, *nrows, 0);
        *nrows += block.nrows;
    }

    let ncols = blocks.get(0).map(|b| b.as_ref().ncols).unwrap_or(0);
    merge_op(&mut erase_iter(blocks), 0, ncols, op)
}

pub fn block_diag<B: AsRef<Builder>>(blocks: &[B]) -> Builder {
    fn op(acc: &mut Builder, nrows: &mut usize, ncols: &mut usize, block: &Builder) {
        block_merge(acc, block, *nrows, *ncols);
        *nrows += block.nrows;
        *ncols += block.ncols;
    }

    merge_op(&mut erase_iter(blocks), 0, 0, op)
}

pub fn bmat<B: AsRef<Builder>>(blocks: &[&[Option<B>]]) -> Builder {
    let nrows = blocks.len();
    if nrows == 0 {
        return zeros(0, 0);
    }
    let ncols = blocks[0].len();
    assert!(
        blocks.iter().all(|row| row.len() == ncols),
        "bmat must be given a rectangular layout"
    );

    // Check that all blocks in a row/column must have the same number of rows/columns
    let mut block_nrows = vec![usize::max_value(); nrows];
    let mut block_ncols = vec![usize::max_value(); ncols];
    fn check_or_assign(left: &mut usize, right: usize, name: &str) {
        assert!(right != usize::max_value());
        if *left == usize::max_value() {
            *left = right;
        } else {
            assert_eq!(
                *left, right,
                "bmat requires all blocks in a {0} to have the same number of {0}s",
                name,
            );
        }
    }
    for (row, nrows) in blocks.iter().zip(&mut block_nrows) {
        for (block, ncols) in row.iter().zip(&mut block_ncols) {
            if let &Some(ref block) = block {
                let block = block.as_ref();
                check_or_assign(nrows, block.nrows, "row");
                check_or_assign(ncols, block.ncols, "column");
            }
        }
    }

    // Calculate row and column offsets and check that the resulting matrix has a known size
    fn cumsum(vals: &mut [usize], name: &str) {
        let mut acc = 0;
        for n in vals {
            if *n == usize::max_value() {
                panic!("bmat requires all {}s to have a known size", name);
            }
            acc += *n;
            *n = acc;
        }
    }
    cumsum(&mut block_nrows, "row");
    cumsum(&mut block_ncols, "column");

    // Merge the blocks
    let mut acc = preallocate_for_merge(
        &mut blocks
            .iter()
            .flat_map(|r| r.iter().filter_map(Option::as_ref))
            .map(AsRef::as_ref),
    );
    acc.nrows = *block_nrows.last().unwrap();
    acc.ncols = *block_ncols.last().unwrap();;

    blocks
        .iter()
        .zip(once(&0).chain(&block_nrows))
        .fold(acc, |acc, (row, &row_offset)| {
            row.iter().zip(once(&0).chain(&block_ncols)).fold(
                acc,
                |mut acc, (block, &col_offset)| {
                    if let &Some(ref block) = block {
                        block_merge(&mut acc, block.as_ref(), row_offset, col_offset);
                    }
                    acc
                },
            )
        })
}

fn preallocate_for_merge(blocks: &mut Iterator<Item = &Builder>) -> Builder {
    let (nnz, nt) = blocks.map(AsRef::as_ref).fold((0, 0), |(nnz, nt), b| {
        (nnz + b.coords.len(), nt + b.tracked_blocks.len())
    });
    let mut builder = Builder::with_capacity(0, 0, nnz);
    builder.tracked_blocks = Vec::with_capacity(nt);
    builder
}

fn erase_iter<'a, B: AsRef<Builder>>(
    blocks: &'a [B],
) -> Erased<::std::iter::Map<::std::slice::Iter<'a, B>, for<'r> fn(&'r B) -> &'r Builder>> {
    Erased::new(blocks.iter().map(AsRef::as_ref))
}

fn merge_op(
    blocks: &mut ErasedIntoIter<Item = &Builder>,
    nrows: usize,
    ncols: usize,
    op: fn(&mut Builder, &mut usize, &mut usize, &Builder),
) -> Builder {
    let acc = preallocate_for_merge(blocks.into_iter());

    let (mut acc, final_nrows, final_ncols) = blocks.into_iter().fold(
        (acc, nrows, ncols),
        |(mut acc, mut nrows, mut ncols), block| {
            op(&mut acc, &mut nrows, &mut ncols, block);
            (acc, nrows, ncols)
        },
    );

    acc.nrows = final_nrows;
    acc.ncols = final_ncols;
    acc
}

fn block_merge(left: &mut Builder, right: &Builder, row_shift: usize, col_shift: usize) {
    let left_coords_len = left.coords.len();
    let left_tracked_blocks_len = left.tracked_blocks.len();

    left.coords.extend_from_slice(&right.coords);
    left.tracked_blocks.extend_from_slice(&right.tracked_blocks);

    // Update right coordinates
    for &mut (ref mut r, ref mut c, _) in &mut left.coords[left_coords_len..] {
        *r += row_shift;
        *c += col_shift;
    }

    // Update tracked block positions
    for &mut (_, ref mut r, ref mut c, _) in &mut left.tracked_blocks[left_tracked_blocks_len..] {
        *r += row_shift;
        *c += col_shift;
    }
}

pub struct CscMatrix {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<float>,
    tracked_blocks: Vec<(usize, MatrixMN<Option<usize>, Dy, Dy>)>,
}

impl CscMatrix {
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn data(&self) -> &[float] {
        &self.data
    }

    pub fn set_block<M: Dim, N: Dim>(
        &mut self,
        block: &BlockRef<M, N>,
        value: &MatrixMN<float, M, N>,
    ) where
        DefaultAllocator: Allocator<float, M, N>,
    {
        let (nrows, ncols) = value.shape();
        assert_eq!(block.nrows, nrows);
        assert_eq!(block.ncols, ncols);

        self.set_block_inner(
            block.id,
            value.slice_with_steps((0, 0), (nrows, ncols), (0, 0)),
        );
    }

    fn set_block_inner(&mut self, block_id: usize, value: MatrixSlice<float, Dy, Dy, Dy, Dy>) {
        let start_idx = self
            .tracked_blocks
            .binary_search_by(|&(id, _)| id.cmp(&block_id));
        let mut start_idx = start_idx.expect("Block not in this matrix");

        for i in (0..start_idx).rev() {
            if self.tracked_blocks[i].0 != block_id {
                break;
            }
            start_idx = i;
        }

        for i in start_idx..self.tracked_blocks.len() {
            let &(id, ref indices) = &self.tracked_blocks[i];
            if id != block_id {
                break;
            }
            for (index, &val) in indices.iter().zip(value.iter()) {
                if let &Some(index) = index {
                    self.data[index] = val;
                } else {
                    assert_eq!(
                        0.0, val,
                        "unexpected non-zero element in sparse tracked block"
                    );
                }
            }
        }
    }

    pub fn to_dense(&self) -> MatrixMN<float, Dy, Dy> {
        let mut mat = MatrixMN::zeros_generic(dy(self.nrows), dy(self.ncols));

        for c in 0..self.ncols {
            for i in self.indptr[c]..self.indptr[c + 1] {
                let r = self.indices[i];
                mat[(r, c)] = self.data[i];
            }
        }

        mat
    }
}

fn to_dynamic<V: nalgebra::Scalar, M: Dim, N: Dim>(mat: &MatrixMN<V, M, N>) -> MatrixMN<V, Dy, Dy>
where
    DefaultAllocator: Allocator<V, M, N>,
{
    let (nrows, ncols) = mat.shape();
    let data = mat.as_slice().to_owned();
    let data = nalgebra::MatrixVec::new(dy(nrows), dy(ncols), data);
    nalgebra::Matrix::from_data(data)
}

fn dy(n: usize) -> Dy {
    Dy::new(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix1, Matrix2, Matrix2x3, Matrix3, Matrix3x2, Matrix6};

    #[test]
    fn empty_ops_should_not_panic() {
        add::<Builder>(&[]);
        hstack::<Builder>(&[]);
        vstack::<Builder>(&[]);
        block_diag::<Builder>(&[]);
    }

    #[test]
    fn add_simple() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = Matrix3::new(
            14.0, 0.0, 9.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 7.0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = Matrix3::new(
            2.0, 0.0, 5.0,
            0.0, 4.0, 6.0,
            0.0, 0.0, 3.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = to_dynamic(&Matrix3::new(
            16.0, 0.0, 14.0,
            0.0, 4.0, 6.0,
            1.0, 0.0, 10.0,
        ));

        let comparison = (block(&a) + block(&b)).build_csc().to_dense();

        assert_eq!(expected, comparison);
    }

    #[test]
    #[should_panic]
    fn add_panic() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = Matrix3::new(
            14.0, 0.0, 9.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 7.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = Matrix2x3::new(
            2.0, 0.0, 5.0,
            0.0, 4.0, 6.0,
        );

        let _ = block(&a) + block(&b);
    }

    #[test]
    fn mul_f64() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let a = Matrix3::new(
            14.0, 0.0, 9.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 7.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = Matrix3::new(
            42.0, 0.0, 27.0,
            0.0, 0.0, 0.0,
            3.0, 0.0, 21.0,
        );

        assert_eq!(expected, a * 3.0);
    }

    #[test]
    fn block_diag_empty_row() {
        let a = Matrix1::new(1.0);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = Matrix2x3::new(
            2.0, 3.0, 0.0,
            5.0, 6.0, 0.0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let c = Matrix3x2::new(
            0.0, 0.0,
            10.0, 0.0,
            12.0, 0.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = to_dynamic(&Matrix6::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 3.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 6.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 12.0, 0.0,
        ));

        let comparison = block_diag(&[block(&a), block(&b), block(&c)])
            .build_csc()
            .to_dense();

        assert_eq!(expected, comparison);
    }

    #[test]
    fn bmat_test() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = Matrix2x3::new(
            2.0, 3.0, 4.0,
            5.0, 0.0, 7.0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let c = Matrix2::new(
            14.0, 0.0,
            16.0, 17.0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let d = Matrix3x2::new(
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected = to_dynamic(&Matrix6::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 3.0, 4.0, 14.0, 0.0,
            0.0, 5.0, 0.0, 7.0, 16.0, 17.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let comparison = bmat(&[
            &[Some(zeros(1, 1)), None, None],
            &[None, Some(block(&b)), Some(block(&c))],
            &[None, None, Some(block(&d))],
        ]).build_csc()
            .to_dense();

        assert_eq!(expected, comparison);
    }

    #[test]
    fn bmat_mut_blocks() {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let b = Matrix2::new(
            2.0, 3.0,
            5.0, 0.0,
        );

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let c_sp = Matrix2::new(
            true, true,
            true, true,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let c_val = Matrix2::new(
            14.0, 0.0,
            16.0, 17.0,
        );
        let (c, c_block) = block_mut(&c_sp);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let d_sp = Matrix3x2::new(
            false, false,
            true, false,
            true, false,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let d_val_1 = Matrix3x2::new(
            0.0, 0.0,
            0.0, 0.0,
            1.0, 0.0,
        );
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let d_val_2 = Matrix3x2::new(
            0.0, 0.0,
            99.0, 0.0,
            18.0, 0.0,
        );
        let (d, d_block) = block_mut(&d_sp);

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected_1 = to_dynamic(&Matrix6::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 3.0, 14.0, 0.0,
            0.0, 0.0, 5.0, 0.0, 16.0, 17.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let expected_2 = to_dynamic(&Matrix6::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 3.0, 14.0, 0.0,
            0.0, 0.0, 5.0, 0.0, 16.0, 17.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            99.0, 0.0, 0.0, 0.0, 99.0, 0.0,
            18.0, 0.0, 0.0, 0.0, 18.0, 0.0,
        ));

        let mut comparison = bmat(&[
            &[Some(&zeros(1, 2)), None, None],
            &[None, Some(&block(&b)), Some(&c)],
            &[Some(&d), None, Some(&d)],
        ]).build_csc();

        comparison.set_block(&c_block, &c_val);
        comparison.set_block(&d_block, &d_val_1);

        assert_eq!(expected_1, comparison.to_dense());

        comparison.set_block(&d_block, &d_val_2);

        assert_eq!(expected_2, comparison.to_dense());
    }
}
