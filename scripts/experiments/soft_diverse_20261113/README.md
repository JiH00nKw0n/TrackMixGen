# Rope and Towel experiment scripts

These are the scripts used in the November 2026 diverse-source experiment. They retain the original workstation and server paths and require the original experiment directory layout. Running launch scripts starts jobs or transfers files.

## Annotation

assign.py assigns each demonstration to an independent Astra session. build.py adds datagen_info and checks its calculations against released human demonstrations. check_inputs.py checks preservation of recorded arrays. assemble_publish.py assembles and transfers the annotated sources. reader_check.py and verify_on_isaac.sh verify official generator loading. smoke.sh launches generation attempts.

Required inputs include per-demonstration input.json and boundary.json, original HDF5 files, and reference/rope.hdf5 and reference/towel.hdf5. Data files are stored separately from Git.

## Concurrent generation

throughput.py measures successful attempts using file timestamps. compare_parallel.py compares concurrent workers. apply_parallel.py selects the observed fastest setting. finish_parallel.py combines completed chunks and builds a dataset of 1,000 successes.

The comparison used approximately 54 minutes for one worker and 12 minutes for higher concurrency. Four workers per GPU for Rope and three for Towel had the highest observed success throughput. The short measurement windows limit precision.

These scripts also require the remote tuning/start launcher and experiment state. They are archived execution scripts and require adapting paths and restoring inputs before reuse.

## Training and evaluation

train_hangang.sbatch runs baseline training on Hangang. follow_generation.sh transfers completed datasets, submits training and returns checkpoints to Isaac for evaluation.

The shared generation and evaluation scene is configs/experiment/generator/softmimicgen/scene/diverse20.yaml. Whole-object translation widths are 20 cm on each horizontal axis with yaw of +/-60 degrees. The rope half has additional 20 cm translation widths and yaw of +/-90 degrees.

Publishing these files does not start or register jobs.
