# Step 1: create a config file under folder conf/
# Step 2: run your python script with small data
YAML_File=conf/openai-gpt4.yaml
python main.py --max_doc=2 --job_id='011' --config_teacher ${YAML_File} --config_learner ${YAML_File} --config_evaluator ${YAML_File}
# Step 3: check your ouptut file under folder result/

