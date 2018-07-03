use std::ops::{Deref, DerefMut};

pub struct AlignedBuffer {
    unaligned: Vec<u8>,
    offset: usize,
}

impl AlignedBuffer {
    pub fn zeroed(len: usize, alignment: usize) -> AlignedBuffer {
        let mut unaligned = vec![0; len + alignment - 1];
        let start_ptr = unaligned.as_ptr() as usize;
        let next_aligned_ptr = (1 + (start_ptr - 1) / alignment) * alignment;
        let offset = next_aligned_ptr - start_ptr;
        unaligned.truncate(offset + len);
        AlignedBuffer { unaligned, offset }
    }

    pub fn from_slice(slice: &[u8], alignment: usize) -> AlignedBuffer {
        let mut buf = AlignedBuffer::zeroed(slice.len(), alignment);
        buf.copy_from_slice(slice);
        buf
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.unaligned[self.offset..]
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.unaligned[self.offset..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_correct_length() {
        let buf = AlignedBuffer::zeroed(101, 32);
        assert_eq!(buf.len(), 101);
    }

    #[test]
    fn is_aligned() {
        let buf = AlignedBuffer::zeroed(101, 32);
        assert_eq!(buf.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn copies_slice() {
        let buf = AlignedBuffer::from_slice(&[1, 2, 3, 4, 5, 6], 16);
        assert_eq!(buf[0], 1);
        assert_eq!(buf[5], 6);
    }
}
