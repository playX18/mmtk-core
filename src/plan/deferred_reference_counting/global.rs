use std::sync::Mutex;

use crate::plan::barriers::BarrierSelector;
use crate::plan::deferred_reference_counting::gc_work::{
    DRCGCWorkContext, FinishDeferredReferenceCounting, ProcessDeferredReferenceCounts,
};
use crate::plan::deferred_reference_counting::mutator::ALLOCATOR_MAPPING;
use crate::plan::global::{BasePlan, CommonPlan, CreateGeneralPlanArgs, CreateSpecificPlanArgs};
use crate::plan::{AllocationSemantics, GlobalLogBuffer, Plan, PlanConstraints};
use crate::policy::space::Space;
use crate::scheduler::{GCWorkScheduler, WorkBucketStage};
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::{ObjectReference, VMWorkerThread};
use crate::vm::VMBinding;
use enum_map::EnumMap;
use mmtk_macros::{HasSpaces, PlanTraceObject};

#[cfg(feature = "malloc_mark_sweep")]
pub type MarkSweepSpace<VM> = crate::policy::marksweepspace::malloc_ms::MallocSpace<VM>;
#[cfg(feature = "malloc_mark_sweep")]
use crate::policy::marksweepspace::malloc_ms::MAX_OBJECT_SIZE;

#[cfg(not(feature = "malloc_mark_sweep"))]
pub type MarkSweepSpace<VM> = crate::policy::marksweepspace::native_ms::MarkSweepSpace<VM>;
#[cfg(not(feature = "malloc_mark_sweep"))]
use crate::policy::marksweepspace::native_ms::MAX_OBJECT_SIZE;

#[derive(HasSpaces, PlanTraceObject)]
pub struct DeferredReferenceCounting<VM: VMBinding> {
    #[parent]
    pub(crate) common: CommonPlan<VM>,
    #[space]
    ms: MarkSweepSpace<VM>,
    pub(crate) log_buffer: GlobalLogBuffer,
    old_roots: Mutex<Vec<ObjectReference>>,
    new_roots: Mutex<Vec<ObjectReference>>,
}

/// The plan constraints for the deferred reference counting plan.
pub const DRC_CONSTRAINTS: PlanConstraints = PlanConstraints {
    moves_objects: false,
    max_non_los_default_alloc_bytes: MAX_OBJECT_SIZE,
    needs_log_bit: true,
    barrier: BarrierSelector::LogBufferBarrier,
    may_trace_duplicate_edges: true,
    needs_prepare_mutator: (!cfg!(feature = "malloc_mark_sweep")
        && !cfg!(feature = "eager_sweeping"))
        || PlanConstraints::default().needs_prepare_mutator,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for DeferredReferenceCounting<VM> {
    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        scheduler.schedule_common_work::<DRCGCWorkContext<VM>>(self);
        scheduler.work_buckets[WorkBucketStage::Closure]
            .add(ProcessDeferredReferenceCounts::new(self));
        scheduler.work_buckets[WorkBucketStage::Closure]
            .set_sentinel(Box::new(FinishDeferredReferenceCounting::new(self)));
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        self.new_roots.lock().unwrap().clear();
        self.common.prepare(tls, true);
        self.ms.prepare(true);
    }

    fn release(&mut self, tls: VMWorkerThread) {
        self.ms.release();
        self.common.release(tls, true);
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.ms.end_of_gc();
        self.common.end_of_gc(tls);
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        self.base().collection_required(self, space_full)
    }

    fn current_gc_may_move_object(&self) -> bool {
        false
    }

    fn get_used_pages(&self) -> usize {
        self.common.get_used_pages() + self.ms.reserved_pages()
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.common.base
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.common
    }

    fn constraints(&self) -> &'static PlanConstraints {
        &DRC_CONSTRAINTS
    }
}

impl<VM: VMBinding> DeferredReferenceCounting<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let mut global_side_metadata_specs = SideMetadataContext::new_global_specs(&[]);
        MarkSweepSpace::<VM>::extend_global_side_metadata_specs(&mut global_side_metadata_specs);

        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &DRC_CONSTRAINTS,
            global_side_metadata_specs,
        };

        DeferredReferenceCounting {
            ms: MarkSweepSpace::new(plan_args._get_space_args(
                "drc_ms",
                true,
                false,
                true,
                true,
                VMRequest::discontiguous(),
            )),
            common: CommonPlan::new(plan_args),
            log_buffer: GlobalLogBuffer::new(),
            old_roots: Mutex::new(Vec::new()),
            new_roots: Mutex::new(Vec::new()),
        }
    }

    pub fn ms_space(&self) -> &MarkSweepSpace<VM> {
        &self.ms
    }

    pub(crate) fn add_new_root(&self, root: ObjectReference) {
        self.new_roots.lock().unwrap().push(root);
    }

    pub(crate) fn take_old_roots(&self) -> Vec<ObjectReference> {
        std::mem::take(&mut *self.old_roots.lock().unwrap())
    }

    pub(crate) fn install_new_roots_as_old_roots(&self) {
        let mut new_roots = self.new_roots.lock().unwrap();
        let mut old_roots = self.old_roots.lock().unwrap();
        *old_roots = std::mem::take(&mut *new_roots);
    }
}
