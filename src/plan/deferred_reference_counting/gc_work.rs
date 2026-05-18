use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;

use super::DeferredReferenceCounting;
use crate::plan::global::HasSpaces;
use crate::plan::tracing::SlotIterator;
use crate::plan::{PlanTraceObject, VectorObjectQueue};
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::scheduler::gc_work::{ProcessEdgesBase, ProcessEdgesWork, ScanObjects, SlotOf};
use crate::scheduler::{GCWork, GCWorkContext, GCWorker, WorkBucketStage};
use crate::util::object_enum::ClosureObjectEnumerator;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::{ObjectModel, ObjectTracer, Scanning, VMBinding};
use crate::MMTK;

pub struct DRCGCWorkContext<VM: VMBinding>(PhantomData<VM>);

impl<VM: VMBinding> GCWorkContext for DRCGCWorkContext<VM> {
    type VM = VM;
    type PlanType = DeferredReferenceCounting<VM>;
    type DefaultProcessEdges = DRCProcessEdges<VM>;
    type PinningProcessEdges = DRCProcessEdges<VM>;
}

pub struct DRCProcessEdges<VM: VMBinding> {
    plan: &'static DeferredReferenceCounting<VM>,
    base: ProcessEdgesBase<VM>,
}

impl<VM: VMBinding> ProcessEdgesWork for DRCProcessEdges<VM> {
    type VM = VM;
    type ScanObjectsWorkType = ScanObjects<Self>;
    const OVERWRITE_REFERENCE: bool = false;

    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        let base = ProcessEdgesBase::new(slots, roots, mmtk, bucket);
        let plan = base
            .plan()
            .downcast_ref::<DeferredReferenceCounting<VM>>()
            .unwrap();
        Self { plan, base }
    }

    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        VM::VMObjectModel::LOCAL_REFERENCE_COUNT_SPEC.inc::<VM>(object, Ordering::SeqCst);
        object
    }

    fn process_slot(&mut self, slot: SlotOf<Self>) {
        let Some(object) = slot.load() else {
            return;
        };
        self.trace_object(object);
        if self.roots {
            self.plan.add_new_root(object);
        }
    }

    fn create_scan_work(&self, _nodes: Vec<ObjectReference>) -> Option<Self::ScanObjectsWorkType> {
        None
    }
}

impl<VM: VMBinding> Deref for DRCProcessEdges<VM> {
    type Target = ProcessEdgesBase<VM>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<VM: VMBinding> DerefMut for DRCProcessEdges<VM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

pub struct ProcessDeferredReferenceCounts<VM: VMBinding> {
    plan: &'static DeferredReferenceCounting<VM>,
}

impl<VM: VMBinding> ProcessDeferredReferenceCounts<VM> {
    pub fn new(plan: &'static DeferredReferenceCounting<VM>) -> Self {
        Self { plan }
    }
}

impl<VM: VMBinding> GCWork<VM> for ProcessDeferredReferenceCounts<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        for root in self.plan.take_old_roots() {
            decrement::<VM>(root);
        }

        let entries = self.plan.log_buffer.take();
        for logged_object in crate::plan::LogBuffer::iter_object_groups_in(&entries) {
            for old_child in logged_object.children() {
                decrement::<VM>(old_child);
            }

            scan_object_children::<VM>(logged_object.object, |child| {
                increment::<VM>(child);
            });
        }
    }
}

pub struct FinishDeferredReferenceCounting<VM: VMBinding> {
    plan: &'static DeferredReferenceCounting<VM>,
}

impl<VM: VMBinding> FinishDeferredReferenceCounting<VM> {
    pub fn new(plan: &'static DeferredReferenceCounting<VM>) -> Self {
        Self { plan }
    }
}

impl<VM: VMBinding> GCWork<VM> for FinishDeferredReferenceCounting<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let mut zero_queue = VecDeque::new();
        self.plan.for_each_space(&mut |space| {
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|object| {
                if ref_count::<VM>(object) == 0 {
                    zero_queue.push_back(object);
                }
            });
            space.enumerate_objects(&mut enumerator);
        });

        cascade_zero_counts::<VM>(zero_queue);

        self.plan.for_each_space(&mut |space| {
            let mut enumerator = ClosureObjectEnumerator::<_, VM>::new(|object| {
                if ref_count::<VM>(object) != 0 {
                    mark_live::<VM>(self.plan, worker, object);
                    reset_log_bit::<VM>(object);
                }
            });
            space.enumerate_objects(&mut enumerator);
        });

        self.plan.install_new_roots_as_old_roots();
    }
}

fn increment<VM: VMBinding>(object: ObjectReference) -> bool {
    VM::VMObjectModel::LOCAL_REFERENCE_COUNT_SPEC.inc::<VM>(object, Ordering::SeqCst)
}

fn decrement<VM: VMBinding>(object: ObjectReference) -> bool {
    VM::VMObjectModel::LOCAL_REFERENCE_COUNT_SPEC.dec::<VM>(object, Ordering::SeqCst)
}

fn ref_count<VM: VMBinding>(object: ObjectReference) -> usize {
    VM::VMObjectModel::LOCAL_REFERENCE_COUNT_SPEC.load::<VM>(object, Ordering::SeqCst)
}

fn cascade_zero_counts<VM: VMBinding>(
    initial_zero_count_objects: impl IntoIterator<Item = ObjectReference>,
) -> HashSet<ObjectReference> {
    cascade_zero_counts_with_children::<VM, _, _>(initial_zero_count_objects, |object, visit| {
        scan_object_children::<VM>(object, visit);
    })
}

fn cascade_zero_counts_with_children<VM: VMBinding, I, F>(
    initial_zero_count_objects: I,
    mut scan_children: F,
) -> HashSet<ObjectReference>
where
    I: IntoIterator<Item = ObjectReference>,
    F: FnMut(ObjectReference, &mut dyn FnMut(ObjectReference)),
{
    let mut zero_queue: VecDeque<_> = initial_zero_count_objects.into_iter().collect();
    let mut processed_dead = HashSet::new();
    while let Some(object) = zero_queue.pop_front() {
        if ref_count::<VM>(object) != 0 || !processed_dead.insert(object) {
            continue;
        }
        let mut visit_child = |child| {
            if decrement::<VM>(child) && ref_count::<VM>(child) == 0 {
                zero_queue.push_back(child);
            }
        };
        scan_children(object, &mut visit_child);
    }
    processed_dead
}

fn mark_live<VM: VMBinding>(
    plan: &'static DeferredReferenceCounting<VM>,
    worker: &mut GCWorker<VM>,
    object: ObjectReference,
) {
    let mut queue = VectorObjectQueue::new();
    plan.trace_object::<VectorObjectQueue, DEFAULT_TRACE>(&mut queue, object, worker);
}

fn reset_log_bit<VM: VMBinding>(object: ObjectReference) {
    VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.store_atomic::<VM, u8>(
        object,
        1,
        None,
        Ordering::SeqCst,
    );
}

fn scan_object_children<VM: VMBinding>(
    object: ObjectReference,
    mut visit: impl FnMut(ObjectReference),
) {
    let tls = crate::util::VMWorkerThread(crate::util::VMThread::UNINITIALIZED);
    if VM::VMScanning::support_slot_enqueuing(tls, object) {
        SlotIterator::<VM>::iterate_fields(object, tls.0, |slot| {
            if let Some(child) = slot.load() {
                visit(child);
            }
        });
    } else {
        let mut tracer = ChildVisitor { visit: &mut visit };
        VM::VMScanning::scan_object_and_trace_edges(tls, object, &mut tracer);
    }
}

struct ChildVisitor<'a, F: FnMut(ObjectReference)> {
    visit: &'a mut F,
}

impl<F: FnMut(ObjectReference)> ObjectTracer for ChildVisitor<'_, F> {
    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        (self.visit)(object);
        object
    }
}

#[cfg(all(test, feature = "mock_test"))]
mod tests {
    use super::*;
    use crate::util::test_util::mock_vm::{default_setup, no_cleanup, with_mockvm, MockVM};
    use crate::util::Address;

    struct TestNode {
        header: usize,
    }

    impl TestNode {
        fn new() -> Self {
            Self { header: 0 }
        }

        fn object(&mut self) -> ObjectReference {
            ObjectReference::from_raw_address(Address::from_ref(&self.header)).unwrap()
        }
    }

    #[test]
    fn zero_count_cascade_processes_a_binary_tree() {
        with_mockvm(
            default_setup,
            || {
                let mut nodes = [
                    TestNode::new(),
                    TestNode::new(),
                    TestNode::new(),
                    TestNode::new(),
                    TestNode::new(),
                    TestNode::new(),
                    TestNode::new(),
                ];
                let root = nodes[0].object();
                let left = nodes[1].object();
                let right = nodes[2].object();
                let left_left = nodes[3].object();
                let left_right = nodes[4].object();
                let right_left = nodes[5].object();
                let right_right = nodes[6].object();

                MockVM::LOCAL_REFERENCE_COUNT_SPEC.store::<MockVM>(root, 1, Ordering::SeqCst);
                for object in [left, right, left_left, left_right, right_left, right_right] {
                    MockVM::LOCAL_REFERENCE_COUNT_SPEC.store::<MockVM>(object, 1, Ordering::SeqCst);
                }

                assert!(decrement::<MockVM>(root));
                let processed =
                    cascade_zero_counts_with_children::<MockVM, _, _>([root], |object, visit| {
                        match object {
                            object if object == root => {
                                visit(left);
                                visit(right);
                            }
                            object if object == left => {
                                visit(left_left);
                                visit(left_right);
                            }
                            object if object == right => {
                                visit(right_left);
                                visit(right_right);
                            }
                            _ => {}
                        }
                    });

                for object in [
                    root,
                    left,
                    right,
                    left_left,
                    left_right,
                    right_left,
                    right_right,
                ] {
                    assert!(processed.contains(&object));
                    assert_eq!(ref_count::<MockVM>(object), 0);
                }
            },
            no_cleanup,
        );
    }
}
