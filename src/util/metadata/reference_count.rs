use crate::util::metadata::MetadataValue;
use crate::util::ObjectReference;
use crate::vm::VMBinding;
use crate::vm::VMLocalReferenceCountSpec;
use std::sync::atomic::Ordering;

impl VMLocalReferenceCountSpec {
    /// Load the current reference count.
    pub fn load<VM: VMBinding>(&self, object: ObjectReference, ordering: Ordering) -> usize {
        match self.num_bits() {
            1..=8 => self.load_inner::<VM, u8>(object, ordering) as usize,
            9..=16 => self.load_inner::<VM, u16>(object, ordering) as usize,
            17..=32 => self.load_inner::<VM, u32>(object, ordering) as usize,
            33..=64 => self.load_inner::<VM, u64>(object, ordering) as usize,
            _ => unreachable!("reference count metadata must be between 1 bit and one word"),
        }
    }

    /// Store a reference count value.
    pub fn store<VM: VMBinding>(&self, object: ObjectReference, value: usize, ordering: Ordering) {
        debug_assert!(value <= self.max_value());
        match self.num_bits() {
            1..=8 => self.store_inner::<VM, u8>(object, value as u8, ordering),
            9..=16 => self.store_inner::<VM, u16>(object, value as u16, ordering),
            17..=32 => self.store_inner::<VM, u32>(object, value as u32, ordering),
            33..=64 => self.store_inner::<VM, u64>(object, value as u64, ordering),
            _ => unreachable!("reference count metadata must be between 1 bit and one word"),
        }
    }

    /// Increment the reference count.
    ///
    /// Returns `true` if the counter was incremented, or `false` if it was already at the
    /// maximum value representable by this metadata spec.
    pub fn inc<VM: VMBinding>(&self, object: ObjectReference, ordering: Ordering) -> bool {
        match self.num_bits() {
            1..=8 => self.inc_inner::<VM, u8>(object, ordering, self.max_value() as u8),
            9..=16 => self.inc_inner::<VM, u16>(object, ordering, self.max_value() as u16),
            17..=32 => self.inc_inner::<VM, u32>(object, ordering, self.max_value() as u32),
            33..=64 => self.inc_inner::<VM, u64>(object, ordering, self.max_value() as u64),
            _ => unreachable!("reference count metadata must be between 1 bit and one word"),
        }
    }

    /// Decrement the reference count.
    ///
    /// Returns `true` if the counter was decremented, or `false` if it was already zero.
    pub fn dec<VM: VMBinding>(&self, object: ObjectReference, ordering: Ordering) -> bool {
        match self.num_bits() {
            1..=8 => self.dec_inner::<VM, u8>(object, ordering),
            9..=16 => self.dec_inner::<VM, u16>(object, ordering),
            17..=32 => self.dec_inner::<VM, u32>(object, ordering),
            33..=64 => self.dec_inner::<VM, u64>(object, ordering),
            _ => unreachable!("reference count metadata must be between 1 bit and one word"),
        }
    }

    /// The maximum value representable by this reference count metadata spec.
    pub fn max_value(&self) -> usize {
        let num_bits = self.num_bits();
        if num_bits == usize::BITS as usize {
            usize::MAX
        } else {
            (1usize << num_bits) - 1
        }
    }

    fn inc_inner<VM: VMBinding, T: MetadataValue>(
        &self,
        object: ObjectReference,
        ordering: Ordering,
        max: T,
    ) -> bool {
        let one = T::from_u8(1).unwrap();
        self.fetch_update_metadata::<VM, T, _>(
            object,
            ordering,
            fetch_order_for_update(ordering),
            |old| {
                if old == max {
                    None
                } else {
                    Some(old + one)
                }
            },
        )
        .is_ok()
    }

    fn load_inner<VM: VMBinding, T: MetadataValue>(
        &self,
        object: ObjectReference,
        ordering: Ordering,
    ) -> T {
        self.load_atomic::<VM, T>(object, None, ordering)
    }

    fn store_inner<VM: VMBinding, T: MetadataValue>(
        &self,
        object: ObjectReference,
        value: T,
        ordering: Ordering,
    ) {
        self.store_atomic::<VM, T>(object, value, None, ordering)
    }

    fn dec_inner<VM: VMBinding, T: MetadataValue>(
        &self,
        object: ObjectReference,
        ordering: Ordering,
    ) -> bool {
        let one = T::from_u8(1).unwrap();
        self.fetch_update_metadata::<VM, T, _>(
            object,
            ordering,
            fetch_order_for_update(ordering),
            |old| {
                if old.is_zero() {
                    None
                } else {
                    Some(old - one)
                }
            },
        )
        .is_ok()
    }
}

fn fetch_order_for_update(ordering: Ordering) -> Ordering {
    match ordering {
        Ordering::Release => Ordering::Relaxed,
        Ordering::AcqRel => Ordering::Acquire,
        _ => ordering,
    }
}

#[cfg(all(test, feature = "mock_test"))]
mod tests {
    use super::*;
    use crate::util::test_util::mock_vm::{default_setup, no_cleanup, with_mockvm, MockVM};
    use crate::util::{Address, ObjectReference};
    use std::sync::atomic::Ordering;

    fn object_from_header(header: &mut usize) -> ObjectReference {
        ObjectReference::from_raw_address(Address::from_ref(header)).unwrap()
    }

    #[test]
    fn inc_returns_false_without_wrapping_when_ref_count_overflows() {
        with_mockvm(
            default_setup,
            || {
                let spec = VMLocalReferenceCountSpec::in_header(0, 1);
                let mut header = 0b10usize;
                let object = object_from_header(&mut header);

                assert!(spec.inc::<MockVM>(object, Ordering::SeqCst));
                assert_eq!(header, 0b11);
                assert!(!spec.inc::<MockVM>(object, Ordering::SeqCst));
                assert_eq!(header, 0b11);
            },
            no_cleanup,
        );
    }

    #[test]
    fn dec_returns_false_without_wrapping_when_ref_count_underflows() {
        with_mockvm(
            default_setup,
            || {
                let spec = VMLocalReferenceCountSpec::in_header(0, 1);
                let mut header = 0b01usize;
                let object = object_from_header(&mut header);

                assert!(spec.dec::<MockVM>(object, Ordering::SeqCst));
                assert_eq!(header, 0);
                assert!(!spec.dec::<MockVM>(object, Ordering::SeqCst));
                assert_eq!(header, 0);
            },
            no_cleanup,
        );
    }

    #[test]
    fn load_and_store_use_configured_ref_count_width() {
        with_mockvm(
            default_setup,
            || {
                let spec = VMLocalReferenceCountSpec::in_header(0, 2);
                let mut header = 0usize;
                let object = object_from_header(&mut header);

                spec.store::<MockVM>(object, 3, Ordering::SeqCst);

                assert_eq!(spec.load::<MockVM>(object, Ordering::SeqCst), 3);
                assert_eq!(header, 3);
            },
            no_cleanup,
        );
    }
}
