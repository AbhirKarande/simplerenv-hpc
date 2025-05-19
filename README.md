# How to use SimplerEnv

Prerequisites: You have a directory in HPC that has the example.py and a container definition file (.def) in it.

1. Start a new GPU node: `salloc --time=01:00:00 --partition=gpu --ntasks=1 --cpus-per-task=4 --gpus-per-task=p100:1 --mem=64GB --account=jdeshmuk_1278`
   You may need to change CPU/GPU/memory parameters based on free resources, check using `noderes -f -g`

2. Load apptainer: `module load apptainer`

2a. For octo inference, use simplerenv-octo.def
2b. For rt-1-x inference, use simplerenv-rt1x


3. Build container: `apptainer build simplerenv.sif simplerenv-[your-model].def`

4. Start container `apptainer shell --nv --writable-tmpfs simplerenv.sif`

5. Test vulkan: `vulkaninfo | head -n 5`
   If there's an error at this step, try restarting the container or trying a different GPU node

6. Run script: `python example.py`
example.py currently has code for running octo inference. It needs to be adjusted to also return action-dimension entropy. 




