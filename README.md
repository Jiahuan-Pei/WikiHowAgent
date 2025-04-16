# WorldTaskEval
## Motivation
Large language models (LLMs), such as ChatGPT, are increasingly used to generate responses based on human instructions. 

> Can LLMs be effective in helping humans complete concrete tasks in real-world teaching-learning scenarios?

While several studies have evaluated LLMs' performance on specific tasks, the scope of these evaluations remains limited. In contrast, a vast number of practical tasks exist in the real world. For example, WikiHow contains hundreds of guidelines that teach people how to perform tasks, which cover  a wide variety of human-written tasks ranging from "how to bake a cake" to "how to build a jet." 

The objective of this project is to assess LLMs' ability to assist with real-world task completion.

This code implements a conversational AI system where a Teacher Agent guides a Learner Agent through a tutorial, simulating an interactive learning experience. Here's a high-level breakdown of the logic:

## How to run
### Step 1: create a config file under folder [conf/](conf) 
### Step 2: run your python script with small data
```
python main.py --max_doc=2 --job_id='011' --config_teacher conf/ollama-mistral.yaml --config_learner conf/ollama-mistral.yaml --config_evaluator conf/ollama-mistral.yaml
```
### Step 3: check your ouptut file under folder [result/](result)


## Project Structure

```bash
project_root/
│-- main.py  # Entry point
│-- config/
│   │-- config_1.yaml
│   │-- config_2.yaml
│   │-- config_3.yaml
│-- utils/
│   │-- __init__.py
│   │-- util.py
│   │-- ...
│-- agents/                        # Define the multiple LLM agents.
│   │-- __init__.py
│   │-- teacher_agent.py
│   │-- learner_agent.py
│   │-- evaluator_agent.py
│-- manager/
│   │-- __init__.py
│   │-- mutiple_agent_workflow.py  # Define the conversation flow.
│-- evaluation/
│   │-- __init__.py
│   │-- evaluation_metrics.py
│-- data/
│   │-- wikihow/                   # Store tutorial data
│-- result/                        # Store experimental results
│-- requirements.txt
```


## Implementation details 

This diagram illutrates the workflow where Teacher-Learner Agents interact in Conversation Flow, following a comprehensive Evaluation. 

#### 1. Initialization & Configuration
- Loads dependencies (LLMs, embeddings, utilities, multiprocessing, logging, evaluation metrics).
- Reads YAML-based configurations for:
    - Teacher (config_teacher)
    - Learner (config_learner)
    - Evaluator (config_evaluator)

#### 2. Teacher and Learner Agents
**Teacher Agent**
- Guides the learner through a tutorial step-by-step.
- Responds to learner questions without advancing.
- Moves to the next step if the learner indicates understanding and wants to move on.

**Learner Agent**
- Reads the teacher's instructions and responds naturally.
- Acknowledges clear instructions or asks specific questions when confused.
- Responds with gratitude when the tutorial is completed.

#### 3. Conversation Graph (State Management)
- Uses LangGraph to simulate conversation flow.
- Nodes:
    - Teacher Node: Responds to the learner’s message.
    - Learner Node: Replies to the teacher’s instruction.
- Flow Logic:
    - If the learner asks a question → teacher responds.
    - If the learner understands and wants to next step → next tutorial step.
    - Stops when the conversation reaches a completion condition.

#### 4. Conversation Generation
- Reads a tutorial from a JSON file.
- Creates a conversation based on the tutorial steps.
- Saves the generated dialogues and evaluations.

#### 5. Conversation Evaluation
- Conversation-level metrics
    - `Question Ratio`: How many learner responses contain questions?
    - `Task Completion Ratio`: How many percentage of the conversations reach "FINISHED"?
    - `Diversity Score`: Measures response uniqueness.
- Utterance-level metrics
    - NLP-based quality metrics (i.e., `BLEU`, `ROUGE`, `METEOR`, `BERTScore`).
    - LLM-based metrics based on [rubrics](data/evaluation_rubrics.json) *with* and *without* reference, evaluting `Clarity`, `Truthfulness`, `Engagement`, `Coherence`, `Depth`, `Relevance`, `Progress`. The target of evaluation role include *teacher*, *learner*, and *conversation*. 

## Dataset

### Key concepts
- Categories
- Sub-categories
- Topic
- Question

### Statistics
|   | Total Number  | Avg Token  |
|:-|-:|-:|
| Document | 3,680  | 1516.66 |
| Method | 14,738  | 304.34 |
| Step | 49,344  | 88.60 |
| Q & A | 6,093  | 52.32 |
| Tips | 6,495  | 29.59 |
| Warnings | 2,899  | 30.35 |
| Reference | 27,169 | 7.38 |

- Average methods per document: 4.00
- Average steps per method: 5.18
- Average QAs per document: 1.66

### Task distribution over massive topics
![alt text](figure/tasks_per_topic.png)

### Knowledge Graph Screenshot & Statistics
- Total nodes (categories): 727
- Root nodes: 1
- Leaf nodes: 415

[View Interactive Knowledge Graph via a Browser](figure/interactive_knowledge_graph.html)

![Knowledge Graph Screenshot](figure/knowledge_graph_screenshot.png)

