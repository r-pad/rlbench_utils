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
python tools/dataset_generator.py --tasks=stack_wine --save_path="/home/beisner/datasets/rlbench"
```
