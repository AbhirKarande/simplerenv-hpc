# How to use SimplerEnv

1. Start a new GPU node: `salloc --time=01:00:00 --partition=gpu --ntasks=1 --cpus-per-task=4 --gpus-per-task=p100:1 --mem=64GB --account=jdeshmuk_1278`
   You may need to change CPU/GPU/memory parameters based on free resources, check using `noderes -f -g`

2. Load apptainer: `module load apptainer`

3. Build container: `apptainer build simplerenv.sif simplerenv.def`

4. Start container `apptainer shell --nv --writable-tmpfs simplerenv.sif`

5. Test vulkan: `vulkaninfo | head -n 5`
   If there's an error at this step, try restarting the container or trying a different GPU node

6. Run script: `python example.py`
