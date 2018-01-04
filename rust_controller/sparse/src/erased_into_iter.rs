pub trait ErasedIntoIter {
    type Item;

    fn into_iter(&mut self) -> &mut Iterator<Item = Self::Item>;
}

pub struct Erased<I> {
    orig: I,
    iter: I,
}

impl<I: Clone + Iterator> Erased<I> {
    pub fn new(iter: I) -> Erased<I> {
        Erased {
            orig: iter.clone(),
            iter: iter,
        }
    }
}

impl<I: Clone + Iterator> ErasedIntoIter for Erased<I> {
    type Item = I::Item;

    fn into_iter(&mut self) -> &mut Iterator<Item = Self::Item> {
        self.iter = self.orig.clone();
        &mut self.iter
    }
}
