#!/bin/bash
#SBATCH --job-name=conversation_generation
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --time=2-00:00:00        # Max runtime (2 days)
#SBATCH --nodes=1                # Use #number physical machines
#SBATCH --ntasks=1               # 🔥 Run #number parallel python scripts when you have different settings
#SBATCH --gres=gpu:1             # Request #number GPU, when you need more control over GPU type or specific features  (A100)
#SBATCH --cpus-per-task=8        # 🔥 Assign #number CPUs per task; Match with args.processes=8; If inference is GPU-bound, having too many CPU processes won't help.

#SBATCH --mem=16GB               # Request of memory
#SBATCH --partition=gpu_a100     # Use the GPU partition

echo "Starting job on $(hostname) at $(date)"
echo "Total CPUs allocated: $SLURM_JOB_CPUS_PER_NODE"
echo "Number of CPUs allocated by Slurm=$SLURM_CPUS_PER_TASK"

# --- 1. Load required modules ---
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0

echo 'export NLTK_DATA="/gpfs/home3/jpei1/nltk_data"' >> ~/.bashrc
source ~/.bashrc
source activate worldtaskeval

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
OLLAMA_PID=$!
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 5
done
echo "Ollama server is ready!"
# 4.4 Pull LLM Model (Ensure Model is Downloaded) ---
singularity exec --nv ollama_latest.sif ollama pull deepseek-r1
echo "Ollama model is downloaded!"
singularity exec --nv ollama_latest.sif ollama run deepseek-r1 --verbose
echo "Test Ollama model inference time!"
time singularity exec --nv ollama_latest.sif ollama run deepseek-r1 "Explain quantum mechanics in 100 words."

# --- 5. Run the Python Script with Correct Python Path ---
export PYTHONPATH=$PWD  # Ensure Python finds your package
# echo "Running the Python workflow..."
# singularity exec --nv ollama_latest.sif bash -c "source activate worldtaskeval && /gpfs/home3/jpei1/anaconda3/envs/worldtaskeval/bin/python Agents/multiple_agent_workflow.py"
# DEBUG: fast run of 6 doc and skip existing generation
# ~/anaconda3/envs/worldtaskeval/bin/python Agents/multiple_agent_workflow.py --max_doc 2 --batch_size 4
# ~/anaconda3/envs/worldtaskeval/bin/python Agents/multiple_agent_workflow.py --processes $SLURM_CPUS_PER_TASK --batch_size 32 --skip_existing_gen
~/anaconda3/envs/worldtaskeval/bin/python main.py --processes $SLURM_CPUS_PER_TASK --batch_size 128 --config_teacher conf/ollama-deepseek-r1.yaml --config_learner conf/ollama-deepseek-r1.yaml --config_evaluator conf/ollama-deepseek-r1.yaml

# --- 6. Cleanup: Kill Ollama Server ---
echo "Job completed at $(date)"
kill $OLLAMA_PID
