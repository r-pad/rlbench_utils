# rlbench_utils

## Instructions for installing RLBench
1. Download Coppeliasim
2. Modify your environment files
3. Install cffi - this is a required build dependency for pyrep, and the package is not packaged properly
```
pip install cffi==1.14.2 wheel

# Use --no-build-isolation so that we can install pyrep from source without having to
# contaminate our filesystem with pyrep.
pip install --no-build-isolation "pyrep @ git+https://github.com/stepjam/PyRep.git"

```

## Usage with headless mode
```
VGL_DISPLAY=:0.0
DISPLAY=:0.1
```

## TODO: make sure the CI works: https://github.com/stepjam/PyRep/blob/master/.github/workflows/build.yml

## Here's how to collect the demos from RLBench. This will put the recording directly where you think they are.
```
python tools/dataset_generator.py --tasks=stack_wine,insert_onto_square_peg,phone_on_base,put_toilet_roll_on_stand --save_path="/home/beisner/datasets/rlbench" --image_size=256,256 --processes=1 --variations=1 --episodes_per_task=10 --debug
```

This is using my custom variation of the dataset_generator script. Find it in this fork: https://github.com/beneisner/RLBench/tree/beisner/add_custom_to_dataset_generation


# Adding a new task

1. Generate demos from the dataset generator: https://github.com/stepjam/RLBench/blob/master/tools/dataset_generator.py

```
python tools/dataset_generator.py --tasks=insert_onto_square_peg --save_path="/home/beisner/datasets/rlbench" --processes=10
```

2. Figure out the segmentation labels for relevant objects is in each phase.
