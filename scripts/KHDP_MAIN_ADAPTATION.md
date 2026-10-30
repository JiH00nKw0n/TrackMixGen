# KHDP Soft execution based on main 0625348

The policy implementations and image task budgets are unchanged from main 0625348c88ffa3e7e5862a753774e88c0acaf635, including the DP AdamW/per-sample crop update and BC-RNN None/tuple serialization fixes.

The KHDP-only driver explicitly retains the campaign's existing paths. The composed selection remains official. Image training reads exactly the same HDF5 frames on demand to fit the 46 GiB memory limit. Statistics keep main's dtype, sampled indices, order and reductions. Only old optimizer resume states are pruned after a newer checkpoint commits; all ten inference checkpoints remain until validation and test complete.

The existing measured resource admission rules, two policy jobs, four evaluation processes at most, and verified retirement are preserved. The native Soft and training installations are reused. No Dex work is scheduled here.

Generation keeps the original image-only task observations (`task.object_state=[]`, `task.lowdim_terms=null`) and the old unset source-count assertion (`task.num_sources=null`). This lets the already completed Towel chunks retain strict identity comparison (missing and null optional observation terms both mean disabled). These overrides add no crop or change to physics, sources, seed, success rule, scene, chunk size, or training/evaluation budgets. The source file count must be independently verified as three. The latest main generation code is used for the remaining chunks. Completed chunks' original provenance stays intact.

The main scratch-spec rebase is extended to nested launch arguments; otherwise a completed Isaac Lab chunk's output path compares unequal to the temporary path. Completed pools also bypass the new-generation disk allocation check. New or incomplete pools still require the configured 50 GiB free space.

The inherited protocol metadata fields keep_failed=true, select_src_per_subtask=true and selection_strategy=random are retained solely for completed-chunk identity compatibility. SoftMimicGen reads only protocol.guarantee; its actual source-selection, failure recording, interpolation and noise remain those of the official Soft task configuration. The read-only preflight must compare both completed chunks and verify the three-demo source file before launch.
