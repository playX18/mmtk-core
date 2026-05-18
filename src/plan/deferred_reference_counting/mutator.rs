use crate::plan::barriers::{LogBufferBarrier, LogBufferBarrierSemantics};
use crate::plan::deferred_reference_counting::DeferredReferenceCounting;
use crate::plan::mutator_context::create_allocator_mapping;
use crate::plan::mutator_context::{
    common_prepare_func, common_release_func, Mutator, MutatorBuilder, MutatorConfig,
    ReservedAllocators, SpaceMapping,
};
use crate::plan::{AllocationSemantics, Plan};
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::{Address, VMMutatorThread, VMWorkerThread};
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::EnumMap;

#[cfg(feature = "malloc_mark_sweep")]
mod allocator {
    use super::*;

    pub(crate) const RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
        n_malloc: 1,
        ..ReservedAllocators::DEFAULT
    };

    lazy_static! {
        pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
            let mut map = create_allocator_mapping(RESERVED_ALLOCATORS, true);
            map[AllocationSemantics::Default] = AllocatorSelector::Malloc(0);
            map
        };
    }

    pub fn drc_mutator_prepare<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
        common_prepare_func(mutator, tls);
    }

    pub fn drc_mutator_release<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
        common_release_func(mutator, tls);
    }
}

#[cfg(not(feature = "malloc_mark_sweep"))]
mod allocator {
    use super::*;
    use crate::util::alloc::FreeListAllocator;

    pub(crate) const RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
        n_free_list: 1,
        ..ReservedAllocators::DEFAULT
    };

    lazy_static! {
        pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
            let mut map = create_allocator_mapping(RESERVED_ALLOCATORS, true);
            map[AllocationSemantics::Default] = AllocatorSelector::FreeList(0);
            map
        };
    }

    fn get_freelist_allocator_mut<VM: VMBinding>(
        mutator: &mut Mutator<VM>,
    ) -> &mut FreeListAllocator<VM> {
        unsafe {
            mutator
                .allocators
                .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
        }
        .downcast_mut::<FreeListAllocator<VM>>()
        .unwrap()
    }

    pub fn drc_mutator_prepare<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
        get_freelist_allocator_mut::<VM>(mutator).prepare();
        common_prepare_func(mutator, tls);
    }

    pub fn drc_mutator_release<VM: VMBinding>(mutator: &mut Mutator<VM>, tls: VMWorkerThread) {
        get_freelist_allocator_mut::<VM>(mutator).release();
        common_release_func(mutator, tls);
    }
}

pub(crate) use allocator::*;

pub(crate) fn create_space_mapping<VM: VMBinding>(
    plan: &'static dyn Plan<VM = VM>,
) -> Box<SpaceMapping<VM>> {
    let drc = plan
        .downcast_ref::<DeferredReferenceCounting<VM>>()
        .unwrap();
    Box::new({
        let mut vec =
            crate::plan::mutator_context::create_space_mapping(RESERVED_ALLOCATORS, true, plan);
        vec.push((
            ALLOCATOR_MAPPING[AllocationSemantics::Default],
            drc.ms_space(),
        ));
        vec
    })
}

pub struct DRCBarrierSemantics<VM: VMBinding> {
    plan: &'static DeferredReferenceCounting<VM>,
}

impl<VM: VMBinding> DRCBarrierSemantics<VM> {
    pub fn new(plan: &'static DeferredReferenceCounting<VM>) -> Self {
        Self { plan }
    }
}

impl<VM: VMBinding> LogBufferBarrierSemantics for DRCBarrierSemantics<VM> {
    type VM = VM;

    fn flush(&mut self, entries: Vec<Address>) {
        self.plan.log_buffer.combine(entries);
    }
}

pub fn create_drc_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let plan = mmtk
        .get_plan()
        .downcast_ref::<DeferredReferenceCounting<VM>>()
        .unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: create_space_mapping(mmtk.get_plan()),
        prepare_func: &drc_mutator_prepare,
        release_func: &drc_mutator_release,
    };

    let builder = MutatorBuilder::new(mutator_tls, mmtk, config);
    builder
        .barrier(Box::new(LogBufferBarrier::new(
            DRCBarrierSemantics::new(plan),
            mutator_tls,
        )))
        .build()
}
