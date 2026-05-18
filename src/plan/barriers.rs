//! Read/Write barrier implementations.

use std::sync::Mutex;

use crate::vm::slot::{MemorySlice, Slot};
use crate::vm::ObjectModel;
use crate::{
    util::{metadata::MetadataSpec, *},
    vm::VMBinding,
};
use atomic::Ordering;
use downcast_rs::Downcast;

/// BarrierSelector describes which barrier to use.
///
/// This is used as an *indicator* for each plan to enable the correct barrier.
/// For example, immix can use this selector to enable different barriers for analysis.
///
/// VM bindings may also use this to enable the correct fast-path, if the fast-path is implemented in the binding.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BarrierSelector {
    /// No barrier is used.
    NoBarrier,
    /// Object remembering post-write barrier is used.
    ObjectBarrier,
    /// Object remembering pre-write barrier with weak reference loading barrier.
    // TODO: We might be able to generalize this to object remembering pre-write barrier.
    SATBBarrier,
    /// Object remembering post-write barrier that logs each object followed by its children.
    LogBufferBarrier,
}

impl BarrierSelector {
    /// A const function to check if two barrier selectors are the same.
    pub const fn equals(&self, other: BarrierSelector) -> bool {
        // cast enum to u8 then compare. Otherwise, we cannot do it in a const fn.
        *self as u8 == other as u8
    }
}

/// A barrier is a combination of fast-path behaviour + slow-path semantics.
/// This trait exposes generic barrier interfaces. The implementations will define their
/// own fast-path code and slow-path semantics.
///
/// Normally, a binding will call these generic barrier interfaces (`object_reference_write` and `memory_region_copy`) for subsuming barrier calls.
///
/// If a subsuming barrier cannot be easily deployed due to platform limitations, the binding may chosse to call both `object_reference_write_pre` and `object_reference_write_post`
/// barrier before and after the store operation.
///
/// As a performance optimization, the binding may also choose to port the fast-path to the VM side,
/// and call the slow-path (`object_reference_write_slow`) only if necessary.
pub trait Barrier<VM: VMBinding>: 'static + Send + Downcast {
    /// Flush thread-local states like buffers or remembered sets.
    fn flush(&mut self) {}

    /// Weak reference loading barrier.  A mutator should call this when loading from a weak
    /// reference field, for example, when executing  `java.lang.ref.Reference.get()` in JVM, or
    /// loading from a global weak table in CRuby.
    ///
    /// Note: Merely loading from a field holding weak reference into a local variable will create a
    /// strong reference from the stack to the referent, changing its reachablilty from weakly
    /// reachable to strongly reachable.  Concurrent garbage collectors may need to handle such
    /// events specially.  See [SATBBarrier::load_weak_reference] for a concrete example.
    ///
    /// Arguments:
    /// *   `referent`: The referent object which the weak reference is pointing to.
    fn load_weak_reference(&mut self, _referent: ObjectReference) {}

    /// Subsuming barrier for object reference write
    fn object_reference_write(
        &mut self,
        src: ObjectReference,
        slot: VM::VMSlot,
        target: ObjectReference,
    ) {
        self.object_reference_write_pre(src, slot, Some(target));
        slot.store(target);
        self.object_reference_write_post(src, slot, Some(target));
    }

    /// Full pre-barrier for object reference write
    fn object_reference_write_pre(
        &mut self,
        _src: ObjectReference,
        _slot: VM::VMSlot,
        _target: Option<ObjectReference>,
    ) {
    }

    /// Full post-barrier for object reference write
    fn object_reference_write_post(
        &mut self,
        _src: ObjectReference,
        _slot: VM::VMSlot,
        _target: Option<ObjectReference>,
    ) {
    }

    /// Object reference write slow-path call.
    /// This can be called either before or after the store, depend on the concrete barrier implementation.
    fn object_reference_write_slow(
        &mut self,
        _src: ObjectReference,
        _slot: VM::VMSlot,
        _target: Option<ObjectReference>,
    ) {
    }

    /// Subsuming barrier for array copy
    fn memory_region_copy(&mut self, src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        self.memory_region_copy_pre(src.clone(), dst.clone());
        VM::VMMemorySlice::copy(&src, &dst);
        self.memory_region_copy_post(src, dst);
    }

    /// Full pre-barrier for array copy
    fn memory_region_copy_pre(&mut self, _src: VM::VMMemorySlice, _dst: VM::VMMemorySlice) {}

    /// Full post-barrier for array copy
    fn memory_region_copy_post(&mut self, _src: VM::VMMemorySlice, _dst: VM::VMMemorySlice) {}

    /// A pre-barrier indicating that some fields of the object will probably be modified soon.
    /// Specifically, the caller should ensure that:
    ///     * The barrier must called before any field modification.
    ///     * Some fields (unknown at the time of calling this barrier) might be modified soon, without a write barrier.
    ///     * There are no safepoints between the barrier call and the field writes.
    ///
    /// **Example use case for mmtk-openjdk:**
    ///
    /// The OpenJDK C2 slowpath allocation code
    /// can do deoptimization after the allocation and before returning to C2 compiled code.
    /// The deoptimization itself contains a safepoint. For generational plans, if a GC
    /// happens at this safepoint, the allocated object will be promoted, and all the
    /// subsequent field initialization should be recorded.
    ///
    // TODO: Review any potential use cases for other VM bindings.
    fn object_probable_write(&mut self, _obj: ObjectReference) {}
}

impl_downcast!(Barrier<VM> where VM: VMBinding);

/// Empty barrier implementation.
/// For GCs that do not need any barriers
///
/// Note that since NoBarrier noes nothing but the object field write itself, it has no slow-path semantics (i.e. an no-op slow-path).
pub struct NoBarrier;

impl<VM: VMBinding> Barrier<VM> for NoBarrier {}

/// A log buffer entry with object-start entries tagged in the low bit.
///
/// The buffer stores raw addresses so an object entry can be encoded as `object | 1`. Object
/// references are word-aligned, so the low bit is available for this tag. Child entries are stored
/// untagged.
pub struct LogBuffer {
    entries: Vec<Address>,
}

impl LogBuffer {
    const OBJECT_START_TAG: usize = 1;
    const CAPACITY: usize = crate::scheduler::EDGES_WORK_BUFFER_SIZE;

    /// Create an empty log buffer.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return true if the buffer has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of encoded entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if the buffer has reached the default work-packet capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= Self::CAPACITY
    }

    /// Push an object-start entry, encoded as `object | 1`.
    pub fn push_object(&mut self, object: ObjectReference) {
        self.reserve_if_empty();
        self.entries.push(Self::encode_object_start(object));
    }

    /// Push an untagged child reference entry.
    pub fn push_child(&mut self, object: ObjectReference) {
        self.reserve_if_empty();
        self.entries.push(object.to_raw_address());
    }

    /// Return the encoded entries without draining the buffer.
    pub fn as_slice(&self) -> &[Address] {
        &self.entries
    }

    /// Drain the buffer and return the encoded entries.
    pub fn take(&mut self) -> Vec<Address> {
        std::mem::take(&mut self.entries)
    }

    /// Iterate over logged object groups in this buffer.
    pub fn iter_object_groups(&self) -> LogBufferObjectIter<'_> {
        Self::iter_object_groups_in(&self.entries)
    }

    /// Iterate over logged object groups in an encoded entry slice.
    pub fn iter_object_groups_in(entries: &[Address]) -> LogBufferObjectIter<'_> {
        LogBufferObjectIter { entries, cursor: 0 }
    }

    /// Return true if this encoded entry starts a new logged object.
    pub fn is_object_start(entry: Address) -> bool {
        entry.as_usize() & Self::OBJECT_START_TAG != 0
    }

    /// Decode either an object-start entry or a child entry as an object reference.
    pub fn decode_object(entry: Address) -> ObjectReference {
        let raw = entry.as_usize() & !Self::OBJECT_START_TAG;
        ObjectReference::from_raw_address(unsafe { Address::from_usize(raw) })
            .expect("log buffer entries must decode to non-null object references")
    }

    fn encode_object_start(object: ObjectReference) -> Address {
        unsafe { Address::from_usize(object.to_raw_address().as_usize() | Self::OBJECT_START_TAG) }
    }

    fn reserve_if_empty(&mut self) {
        if self.entries.is_empty() {
            self.entries.reserve(Self::CAPACITY);
        }
    }
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// A decoded object group in a [`LogBuffer`].
pub struct LoggedObject<'a> {
    /// The tagged object-start entry decoded as an object reference.
    pub object: ObjectReference,
    children: &'a [Address],
}

impl LoggedObject<'_> {
    /// Iterate over the logged children for this object.
    pub fn children(&self) -> impl Iterator<Item = ObjectReference> + '_ {
        self.children.iter().copied().map(LogBuffer::decode_object)
    }

    /// Return the encoded, untagged child entries for this object.
    pub fn encoded_children(&self) -> &[Address] {
        self.children
    }
}

/// Iterator over object-start entries and their following child entries.
pub struct LogBufferObjectIter<'a> {
    entries: &'a [Address],
    cursor: usize,
}

impl<'a> Iterator for LogBufferObjectIter<'a> {
    type Item = LoggedObject<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.entries.len()
            && !LogBuffer::is_object_start(self.entries[self.cursor])
        {
            debug_assert!(
                false,
                "log buffer object groups should start with tagged object entries"
            );
            self.cursor = self.entries.len();
            return None;
        }

        if self.cursor == self.entries.len() {
            return None;
        }

        let object = LogBuffer::decode_object(self.entries[self.cursor]);
        let children_start = self.cursor + 1;
        let mut next_object = children_start;
        while next_object < self.entries.len()
            && !LogBuffer::is_object_start(self.entries[next_object])
        {
            next_object += 1;
        }
        self.cursor = next_object;

        Some(LoggedObject {
            object,
            children: &self.entries[children_start..next_object],
        })
    }
}

/// A mutex-backed destination for combining per-mutator log buffers.
///
/// Plans that need different scheduling or packetization can implement
/// [`LogBufferBarrierSemantics`] directly instead of using this helper.
pub struct GlobalLogBuffer {
    entries: Mutex<Vec<Address>>,
}

impl GlobalLogBuffer {
    /// Create an empty global log buffer.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Append a drained mutator-local log buffer.
    pub fn combine(&self, mut entries: Vec<Address>) {
        if entries.is_empty() {
            return;
        }
        self.entries.lock().unwrap().append(&mut entries);
    }

    /// Drain all globally combined entries.
    pub fn take(&self) -> Vec<Address> {
        std::mem::take(&mut *self.entries.lock().unwrap())
    }

    /// Return true if no entries are currently combined globally.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }

    /// Return the number of globally combined entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

impl Default for GlobalLogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(raw: usize) -> ObjectReference {
        ObjectReference::from_raw_address(unsafe { Address::from_usize(raw) }).unwrap()
    }

    #[test]
    fn log_buffer_tags_object_starts() {
        let obj = object(0x1000);
        let child = object(0x2000);
        let mut buffer = LogBuffer::new();

        buffer.push_object(obj);
        buffer.push_child(child);

        let entries = buffer.as_slice();
        assert_eq!(entries.len(), 2);
        assert!(LogBuffer::is_object_start(entries[0]));
        assert!(!LogBuffer::is_object_start(entries[1]));
        assert_eq!(LogBuffer::decode_object(entries[0]), obj);
        assert_eq!(LogBuffer::decode_object(entries[1]), child);
        assert_eq!(entries[0].as_usize(), obj.to_raw_address().as_usize() | 1);
        assert_eq!(entries[1].as_usize(), child.to_raw_address().as_usize());
    }

    #[test]
    fn taking_log_buffer_drains_entries() {
        let mut buffer = LogBuffer::new();
        buffer.push_object(object(0x1000));
        buffer.push_child(object(0x2000));

        let entries = buffer.take();

        assert_eq!(entries.len(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn log_buffer_iterates_object_groups() {
        let obj1 = object(0x1000);
        let child1 = object(0x2000);
        let child2 = object(0x3000);
        let obj2 = object(0x4000);
        let child3 = object(0x5000);
        let mut buffer = LogBuffer::new();

        buffer.push_object(obj1);
        buffer.push_child(child1);
        buffer.push_child(child2);
        buffer.push_object(obj2);
        buffer.push_child(child3);

        let groups: Vec<_> = buffer.iter_object_groups().collect();

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].object, obj1);
        assert_eq!(
            groups[0].children().collect::<Vec<_>>(),
            vec![child1, child2]
        );
        assert_eq!(groups[1].object, obj2);
        assert_eq!(groups[1].children().collect::<Vec<_>>(), vec![child3]);
    }
}

/// A barrier semantics defines the barrier slow-path behaviour. For example, how an object barrier processes it's modbufs.
/// Specifically, it defines the slow-path call interfaces and a call to flush buffers.
///
/// A barrier is a combination of fast-path behaviour + slow-path semantics.
/// The fast-path code will decide whether to call the slow-path calls.
pub trait BarrierSemantics: 'static + Send {
    type VM: VMBinding;

    const UNLOG_BIT_SPEC: MetadataSpec =
        *<Self::VM as VMBinding>::VMObjectModel::GLOBAL_LOG_BIT_SPEC.as_spec();

    /// Flush thread-local buffers or remembered sets.
    /// Normally this is called by the slow-path implementation whenever the thread-local buffers are full.
    /// This will also be called externally by the VM, when the thread is being destroyed.
    fn flush(&mut self);

    /// Slow-path call for object field write operations.
    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        slot: <Self::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    );

    /// Slow-path call for mempry slice copy operations. For example, array-copy operations.
    fn memory_region_copy_slow(
        &mut self,
        src: <Self::VM as VMBinding>::VMMemorySlice,
        dst: <Self::VM as VMBinding>::VMMemorySlice,
    );

    /// Object will probably be modified
    fn object_probable_write_slow(&mut self, _obj: ObjectReference) {}

    /// Loading from a weak reference field
    fn load_weak_reference(&mut self, _o: ObjectReference) {}
}

/// Slow-path semantics for [`LogBufferBarrier`].
///
/// The barrier owns the mutator-local [`LogBuffer`]. When that local buffer fills, or when the
/// mutator is flushed, the encoded entries are passed to the semantics implementation. A typical
/// implementation will combine them into a plan-owned [`GlobalLogBuffer`] or turn them into GC
/// work packets.
pub trait LogBufferBarrierSemantics: 'static + Send {
    type VM: VMBinding;

    const UNLOG_BIT_SPEC: MetadataSpec =
        *<Self::VM as VMBinding>::VMObjectModel::GLOBAL_LOG_BIT_SPEC.as_spec();

    /// Flush a drained mutator-local log buffer.
    fn flush(&mut self, entries: Vec<Address>);

    /// Slow-path call for memory slice copy operations.
    fn memory_region_copy_slow(
        &mut self,
        _src: <Self::VM as VMBinding>::VMMemorySlice,
        _dst: <Self::VM as VMBinding>::VMMemorySlice,
    ) {
    }
}

/// A write barrier that logs an object followed by the object's children before modification.
pub struct LogBufferBarrier<S: LogBufferBarrierSemantics> {
    semantics: S,
    tls: VMMutatorThread,
    log_buffer: LogBuffer,
}

impl<S: LogBufferBarrierSemantics> LogBufferBarrier<S> {
    /// Create a new log-buffer barrier with a mutator-local buffer.
    pub fn new(semantics: S, tls: VMMutatorThread) -> Self {
        Self {
            semantics,
            tls,
            log_buffer: LogBuffer::new(),
        }
    }

    /// Return the mutator-local buffer. This is mostly useful for tests and diagnostics.
    pub fn log_buffer(&self) -> &LogBuffer {
        &self.log_buffer
    }

    fn object_is_unlogged(&self, object: ObjectReference) -> bool {
        unsafe { S::UNLOG_BIT_SPEC.load::<S::VM, u8>(object, None) != 0 }
    }

    /// Attempt to atomically log an object.
    /// Returns true if the object is not logged previously.
    fn log_object(&self, object: ObjectReference) -> bool {
        #[cfg(all(feature = "vo_bit", feature = "extreme_assertions"))]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "object bit is unset"
        );
        loop {
            let old_value =
                S::UNLOG_BIT_SPEC.load_atomic::<S::VM, u8>(object, None, Ordering::SeqCst);
            if old_value == 0 {
                return false;
            }
            if S::UNLOG_BIT_SPEC
                .compare_exchange_metadata::<S::VM, u8>(
                    object,
                    1,
                    0,
                    None,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    fn log_object_and_children(&mut self, object: ObjectReference) {
        self.log_buffer.push_object(object);
        crate::plan::tracing::SlotIterator::<S::VM>::iterate_fields(object, self.tls.0, |slot| {
            if let Some(child) = slot.load() {
                self.log_buffer.push_child(child);
            }
        });
        self.flush_if_full();
    }

    fn flush_if_full(&mut self) {
        if self.log_buffer.is_full() {
            self.flush();
        }
    }
}

impl<S: LogBufferBarrierSemantics> Barrier<S::VM> for LogBufferBarrier<S> {
    fn flush(&mut self) {
        let entries = self.log_buffer.take();
        if !entries.is_empty() {
            self.semantics.flush(entries);
        }
    }

    fn object_reference_write_pre(
        &mut self,
        src: ObjectReference,
        slot: <S::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    ) {
        if self.object_is_unlogged(src) {
            self.object_reference_write_slow(src, slot, target);
        }
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        _slot: <S::VM as VMBinding>::VMSlot,
        _target: Option<ObjectReference>,
    ) {
        if self.log_object(src) {
            self.log_object_and_children(src);
        }
    }

    fn memory_region_copy_pre(
        &mut self,
        src: <S::VM as VMBinding>::VMMemorySlice,
        dst: <S::VM as VMBinding>::VMMemorySlice,
    ) {
        self.semantics.memory_region_copy_slow(src, dst);
    }

    fn object_probable_write(&mut self, obj: ObjectReference) {
        if self.object_is_unlogged(obj) && self.log_object(obj) {
            self.log_object_and_children(obj);
        }
    }
}

/// Generic object barrier with a type argument defining it's slow-path behaviour.
pub struct ObjectBarrier<S: BarrierSemantics> {
    semantics: S,
}

impl<S: BarrierSemantics> ObjectBarrier<S> {
    /// Create a new ObjectBarrier with the given semantics.
    pub fn new(semantics: S) -> Self {
        Self { semantics }
    }

    /// Attempt to atomically log an object.
    /// Returns true if the object is not logged previously.
    fn object_is_unlogged(&self, object: ObjectReference) -> bool {
        unsafe { S::UNLOG_BIT_SPEC.load::<S::VM, u8>(object, None) != 0 }
    }

    /// Attempt to atomically log an object.
    /// Returns true if the object is not logged previously.
    fn log_object(&self, object: ObjectReference) -> bool {
        #[cfg(all(feature = "vo_bit", feature = "extreme_assertions"))]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "object bit is unset"
        );
        loop {
            let old_value =
                S::UNLOG_BIT_SPEC.load_atomic::<S::VM, u8>(object, None, Ordering::SeqCst);
            if old_value == 0 {
                return false;
            }
            if S::UNLOG_BIT_SPEC
                .compare_exchange_metadata::<S::VM, u8>(
                    object,
                    1,
                    0,
                    None,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return true;
            }
        }
    }
}

impl<S: BarrierSemantics> Barrier<S::VM> for ObjectBarrier<S> {
    fn flush(&mut self) {
        self.semantics.flush();
    }

    fn object_reference_write_post(
        &mut self,
        src: ObjectReference,
        slot: <S::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    ) {
        if self.object_is_unlogged(src) {
            self.object_reference_write_slow(src, slot, target);
        }
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        slot: <S::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    ) {
        if self.log_object(src) {
            self.semantics
                .object_reference_write_slow(src, slot, target);
        }
    }

    fn memory_region_copy_post(
        &mut self,
        src: <S::VM as VMBinding>::VMMemorySlice,
        dst: <S::VM as VMBinding>::VMMemorySlice,
    ) {
        self.semantics.memory_region_copy_slow(src, dst);
    }

    fn object_probable_write(&mut self, obj: ObjectReference) {
        if self.object_is_unlogged(obj) {
            self.semantics.object_probable_write_slow(obj);
        }
    }
}

/// A SATB (Snapshot-At-The-Beginning) barrier implementation.
/// This barrier is basically a pre-write object barrier with a weak reference loading barrier.
pub struct SATBBarrier<S: BarrierSemantics> {
    weak_ref_barrier_enabled: bool,
    semantics: S,
}

impl<S: BarrierSemantics> SATBBarrier<S> {
    /// Create a new SATBBarrier with the given semantics.
    pub fn new(semantics: S) -> Self {
        Self {
            weak_ref_barrier_enabled: false,
            semantics,
        }
    }

    pub(crate) fn set_weak_ref_barrier_enabled(&mut self, value: bool) {
        self.weak_ref_barrier_enabled = value;
    }

    fn object_is_unlogged(&self, object: ObjectReference) -> bool {
        S::UNLOG_BIT_SPEC.load_atomic::<S::VM, u8>(object, None, Ordering::SeqCst) != 0
    }
}

impl<S: BarrierSemantics> Barrier<S::VM> for SATBBarrier<S> {
    fn flush(&mut self) {
        self.semantics.flush();
    }

    fn load_weak_reference(&mut self, o: ObjectReference) {
        if self.weak_ref_barrier_enabled {
            self.semantics.load_weak_reference(o)
        }
    }

    fn object_probable_write(&mut self, obj: ObjectReference) {
        self.semantics.object_probable_write_slow(obj);
    }

    fn object_reference_write_pre(
        &mut self,
        src: ObjectReference,
        slot: <S::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    ) {
        if self.object_is_unlogged(src) {
            self.semantics
                .object_reference_write_slow(src, slot, target);
        }
    }

    fn object_reference_write_post(
        &mut self,
        _src: ObjectReference,
        _slot: <S::VM as VMBinding>::VMSlot,
        _target: Option<ObjectReference>,
    ) {
        unimplemented!()
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        slot: <S::VM as VMBinding>::VMSlot,
        target: Option<ObjectReference>,
    ) {
        self.semantics
            .object_reference_write_slow(src, slot, target);
    }

    fn memory_region_copy_pre(
        &mut self,
        src: <S::VM as VMBinding>::VMMemorySlice,
        dst: <S::VM as VMBinding>::VMMemorySlice,
    ) {
        self.semantics.memory_region_copy_slow(src, dst);
    }

    fn memory_region_copy_post(
        &mut self,
        _src: <S::VM as VMBinding>::VMMemorySlice,
        _dst: <S::VM as VMBinding>::VMMemorySlice,
    ) {
        unimplemented!()
    }
}
