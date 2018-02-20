use nalgebra::{self, DimAdd, DimSum, MatrixMN, U0};

use prelude::*;
use {ControlModel, State};

pub struct CombineState<M>(pub M);

impl<M: ControlModel> CombineState<M>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0> + Dims3<M::NS, M::NI, M::NP>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    pub fn from_ref(model: &M) -> &CombineState<M> {
        unsafe { &*(model as *const M as *const CombineState<M>) }
    }

    pub fn split_x(x: &Vector<<Self as ControlModel>::NS>) -> (Vector<M::NS>, Vector<M::NP>) {
        let model_x = x.fixed_rows::<M::NS>(0).into_owned();
        let model_p = x.fixed_rows::<M::NP>(M::NS::dim()).into_owned();
        (model_x, model_p)
    }
}

impl<M: ControlModel> ControlModel for CombineState<M>
where
    DefaultAllocator: Dims3<DimSum<M::NS, M::NP>, M::NI, U0> + Dims3<M::NS, M::NI, M::NP>,
    M::NS: DimAdd<M::NP>,
    DimSum<M::NS, M::NP>: DimName,
{
    type NS = DimSum<M::NS, M::NP>;
    type NI = M::NI;
    type NP = U0;

    fn new() -> Self
    where
        Self: Sized,
    {
        panic!("ControlModel::new cannot be called on a CombineState<M>")
    }

    fn name() -> &'static str {
        "combine_state"
    }

    fn state_equation(
        &self,
        x: &Vector<Self::NS>,
        u: &Vector<Self::NI>,
        _p: &Vector<Self::NP>,
    ) -> Vector<Self::NS> {
        let (model_x, model_p) = Self::split_x(x);
        let x_dot = self.0.state_equation(&model_x, u, &model_p);
        let mut x_dot_combined: Vector<Self::NS> = nalgebra::zero();
        x_dot_combined.fixed_rows_mut::<M::NS>(0).copy_from(&x_dot);
        x_dot_combined
    }

    fn linearise(
        &self,
        x0: &Vector<Self::NS>,
        u0: &Vector<Self::NI>,
        _p0: &Vector<Self::NP>,
    ) -> (Matrix<Self::NS, Self::NS>, Matrix<Self::NS, Self::NI>) {
        let (model_x, model_p) = Self::split_x(x0);
        let (A, B) = self.0.linearise(&model_x, u0, &model_p);
        let P = self.0.linearise_parameters(&model_x, u0, &model_p);

        let mut A_combined: Matrix<Self::NS, Self::NS> = nalgebra::zero();
        A_combined
            .fixed_slice_mut::<M::NS, M::NS>(0, 0)
            .copy_from(&A);
        A_combined
            .fixed_slice_mut::<M::NS, M::NP>(0, M::NS::dim())
            .copy_from(&P);

        let mut B_combined: Matrix<Self::NS, Self::NI> = nalgebra::zero();
        B_combined
            .fixed_slice_mut::<M::NS, M::NI>(0, 0)
            .copy_from(&B);

        (A_combined, B_combined)
    }

    fn linearise_sparsity(
        &self,
    ) -> (
        MatrixMN<bool, Self::NS, Self::NS>,
        MatrixMN<bool, Self::NS, Self::NI>,
    ) {
        let (A_mask, B_mask) = self.0.linearise_sparsity();
        let P_mask = self.0.linearise_parameters_sparsity();

        let mut A_combined = MatrixMN::<bool, Self::NS, Self::NS>::from_element(false);
        A_combined
            .fixed_slice_mut::<M::NS, M::NS>(0, 0)
            .copy_from(&A_mask);
        A_combined
            .fixed_slice_mut::<M::NS, M::NP>(0, M::NS::dim())
            .copy_from(&P_mask);

        let mut B_combined = MatrixMN::<bool, Self::NS, Self::NI>::from_element(false);
        B_combined
            .fixed_slice_mut::<M::NS, M::NI>(0, 0)
            .copy_from(&B_mask);

        (A_combined, B_combined)
    }

    fn linearise_parameters(
        &self,
        _x0: &Vector<Self::NS>,
        _u0: &Vector<Self::NI>,
        _p0: &Vector<Self::NP>,
    ) -> Matrix<Self::NS, Self::NP> {
        nalgebra::zero()
    }

    fn x_from_state(&self, _state: &State) -> Vector<Self::NS> {
        panic!("x_form_state cannot be called on a CombineState<M>")
    }

    fn x_to_state(&self, x: &Vector<Self::NS>) -> State {
        self.0.x_to_state(&x.fixed_rows::<M::NS>(0).into_owned())
    }

    fn input_bounds(&self) -> (Vector<Self::NI>, Vector<Self::NI>) {
        self.0.input_bounds()
    }

    fn input_delta_bounds(&self) -> (Vector<Self::NI>, Vector<Self::NI>) {
        self.0.input_delta_bounds()
    }
}
