#!/bin/bash
#SBATCH --job-name=conversation_generation
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --time=48:00:00          # Max runtime (48 hours)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --ntasks=1               # 1 task per job
#SBATCH --cpus-per-task=8        # 8 CPU cores for the task
#SBATCH --gres=gpu:1             # Request 1 GPU (A100)
#SBATCH --mem=491520MB           # Request 480GB of memory
#SBATCH --partition=gpu          # Use the GPU partition

echo "Starting job on $(hostname) at $(date)"

# --- 1. Load required modules ---
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# --- 2. Set Environment Variables (for GPU) ---
export PATH="~/anaconda3/envs/worldtaskeval/bin:$PATH" # Use the virtual env worldtaskeval
# GPU Optimization
export CUDA_VISIBLE_DEVICES=0  # Use the allocated GPU
export OLLAMA_FORCE_CUDA=1
export OLLAMA_NUM_GPU=1        # Match the number of requested GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OLLAMA_NUM_GPU=4
export OLLAMA_ACCELERATE=1
export OLLAMA_LLM_LIBRARY=cublas  # Use cuBLAS for acceleration
# export OLLAMA_LLM_LIBRARY=flash-attn
export OLLAMA_PRECISION=fp16  # Enable 16-bit floats for Tensor Cores # A100 GPUs perform best with mixed precision:
export OLLAMA_BATCH_SIZE=32  # Adjust based on GPU memory
export OLLAMA_NUM_THREAD=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OLLAMA_USE_CUDA_GRAPHS=1
export OLLAMA_CONTEXT_SIZE=8192  # Increase from default (~2048)
export OLLAMA_KEEP_LOADED=1
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
alias python=~/anaconda3/envs/worldtaskeval/bin/python

# --- 3. Activate Virtual Environment ---
# source ~/.bashrc
# conda activate worldtaskeval
# which python  # Verify Python path

# --- 4. Singularity and GPU Access ---
# 4.1 Verify
singularity --version
# Check If Singularity Detects GPUs
singularity exec --nv ollama_latest.sif nvidia-smi
echo "Checking available executables inside Singularity:"
singularity exec --nv ollama_latest.sif ~/anaconda3/envs/worldtaskeval/bin/python -c "import torch; print('*'*20, torch.cuda.is_available(), torch.cuda.device_count())"
singularity exec --nv ollama_latest.sif echo $LD_LIBRARY_PATH
singularity exec --nv ollama_latest.sif which python
singularity exec --nv ollama_latest.sif which ollama
# Check If Ollama Supports Multi-GPU
# singularity exec --nv ollama_latest.sif ollama show

# 4.2 Start and test Ollama with GPU Support ---
echo "Starting Ollama server..."
singularity exec --nv ollama_latest.sif ollama serve &
# singularity exec --nv --bind ~/anaconda3/envs/worldtaskeval ollama_latest.sif ollama serve &
# singularity exec --nv --bind ~/anaconda3/envs/worldtaskeval ollama_latest.sif ollama serve &
OLLAMA_PID=$!
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 5
done
echo "Ollama server is ready!"
# 4.4 Pull LLM Model (Ensure Model is Downloaded) ---
singularity exec --nv ollama_latest.sif ollama pull llama3
echo "Ollama model is downloaded!"
singularity exec --nv ollama_latest.sif ollama run llama3 --verbose
echo "Test Ollama model inference time!"
time singularity exec --nv ollama_latest.sif ollama run llama3 "Explain quantum mechanics in 100 words."

# --- 5. Run the Python Script with Correct Python Path ---
# echo "Running the Python workflow..."
# singularity exec --nv ollama_latest.sif bash -c "source activate worldtaskeval && /gpfs/home3/jpei1/anaconda3/envs/worldtaskeval/bin/python Agents/multiple_agent_workflow.py"
~/anaconda3/envs/worldtaskeval/bin/python Agents/multiple_agent_workflow.py
# singularity exec --nv ollama_latest.sif ~/anaconda3/envs/worldtaskeval/bin/python my_script.py

# --- 6. Cleanup: Kill Ollama Server ---
echo "Job completed at $(date)"
kill $OLLAMA_PID
