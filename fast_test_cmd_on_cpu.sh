# Step 1: create a config file under folder conf/
# Step 2: run your python script with small data
YAML_File=conf/openai-gpt4.yaml
python main.py --max_dialog=2 --job_id='011' --config_teacher ${YAML_File} --config_learner ${YAML_File} --config_evaluator ${YAML_File}
# If continue to run, e.g.,
# First time 
# python main.py --max_dialog=3 --job_id='10002'
# Second time continue
# python main.py --max_dialog=3 --job_id='10002' --previous_dialogue_path result/T-llama3_L-llama3_E-llama3_10001.json
# Step 3: check your ouptut file under folder result/

