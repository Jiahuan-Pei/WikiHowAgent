# Most HPC systems do not allow Docker, but they support Singularity (also called Apptainer).
# Step 1: Check If Singularity is Installed
singularity --version
# Step 2: Create and Run a Singularity Container for Ollama
singularity pull docker://ollama/ollama
# singularity run ollama_latest.sif
# If the container doesn't have CUDA support, rebuild it with GPU libraries:
singularity build --fakeroot --nv ollama_latest.sif docker://ollama/ollama
# Start Ollama in Singularity with GPU support
singularity run --nv ollama_latest.sif
# Step 3: test if the server is running:
curl http://localhost:11434/api/tags