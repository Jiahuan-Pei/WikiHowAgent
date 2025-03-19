import langchain
import re
import yaml
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils import setup_llm_and_embeddings, read_all_file_suffix_X 


class SingleAgent:
    def __init__(self, tutorial_document: list, config_file: str):
        self.tutorial_document = tutorial_document
        self.current_step = 1
        self.total_steps=len(tutorial_document) - 1 # The 0th restores the summary

        self.prompt_template = PromptTemplate(input_variables=["text"], template="{text}")
        self.llm, _, _ = setup_llm_and_embeddings(config_file)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.memory = ConversationBufferMemory()

    def teacher_explain_step(self, step_instruction):
        """Generates the explanation for the current step."""
        if self.current_step < self.total_steps:
            # prompt = f"You are a teacher explaining a tutorial. Explain the following step in detail: {step_instruction}"
            prompt = (
                f"You are a teacher briefly generate an instruction based on a tutorial step. "
                f"Provide a clear and concise explanation in 1-2 sentences. "
                f"Tutorial step: {step_instruction}"
                f"Do not use quotation marks in your response."
                f"Directly generate what the teacher say:"
            )
            explanation_output = self.chain.invoke({"text": prompt})
            explanation = explanation_output["text"] if isinstance(explanation_output, dict) and "text" in explanation_output else str(explanation_output)
            self.memory.save_context(inputs={"teacher_prompt": prompt}, outputs={"teacher_response": explanation})
            return explanation
        else:
            return "All steps have been explained."

    def teacher_response(self, learner_input: str):
        """Processes the learner's response or question and generates a reply."""
        prompt = (f"You are a teacher responding to a learner's question based on the tutorial context.\n"
                    f"Tutorial context: {self.tutorial_document[self.current_step]}\n"
                    f"Learner's request: {learner_input}\n"
                    f"Provide a briefly and clear answer."
                    f"Do not use quotation marks in your response."
                    f"Directly generate what the teacher say:"
        )
        response_output = self.chain.invoke({"text": prompt})
        response = response_output["text"] if isinstance(response_output, dict) and "text" in response_output else str(response_output)
        self.memory.save_context(inputs={"teacher_prompt": prompt}, outputs={"teacher_response": response})
        return response
    
    def learner_request(self, teacher_instruction: str):
        """Learns the material and provide feedback if understand or ask a question."""
        prompt = (
            "You are a learner providing feedback on whether you fully understand the learning content.\n"
            f"Teacher instruction: {teacher_instruction}\n"
            f"Tutorial step: {self.tutorial_document[self.current_step]}\n"
            f"Provide direct feedback on whether you 'understand' the learning content or not. "
            f"If you understand, respond with 'I understand. Please proceed to the next step.' "
            f"If you do not understand, clearly and briefly state what part is unclear and ask ONLY one specific question."
            f"Do not use quotation marks in your response."
            f"Directly generate what the learner say:"
        )
        response_output = self.chain.invoke({"text": prompt})
        response = response_output["text"] if isinstance(response_output, dict) and "text" in response_output else str(response_output)
        self.memory.save_context(inputs={"learner_prompt": teacher_instruction}, outputs={"learner_response": response})
        return response          

    def proceed_to_next_step(self):
        """Moves to the next step in the tutorial."""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            return f"Proceeding to Step {self.current_step + 1}."
        else:
            return "You have completed all the steps in the tutorial!"

    def run_simulation(self):
        """Runs a simulation of the agent working through the tutorial."""
        print('-'*100, '\n', 'Starting Conversation Simulation\n')
        conversation = []
        while self.current_step < self.total_steps:
            ### Step instruction
            step_instruction = self.tutorial_document[self.current_step]
            step_instruction = re.sub(r'\n+', '\n', step_instruction).strip()
            print('*'*50, '\n', f'Step {self.current_step + 1}: {step_instruction}\n', '*'*50)
            
            ### Teacher instruction
            teacher_instruction = self.teacher_explain_step(step_instruction)
            teacher_instruction = re.sub(r'\n+', '\n', teacher_instruction).strip()
            print(f'===>Teacher:  {teacher_instruction}')
            conversation.append('Teacher: ' + teacher_instruction)
            
            ### Learner feedback
            # learner_input = input('Learner: ')  # Simulate learner input
            learner_input = self.learner_request(teacher_instruction)
            learner_input = re.sub(r'\n+', '\n', learner_input).strip()
            print(f'<=Learner: {learner_input}')
            conversation.append('Learner: ' + learner_input)
            if '?' in learner_input:
                teacher_response = self.teacher_response(learner_input)
                teacher_response = re.sub(r'\n+', '\n', teacher_response).strip()
                print(f'===>Teacher: {teacher_response}')
                conversation.append('Teacher: ' + teacher_response)

            ### Simply assuming the teacher answered the learner's question
            self.current_step += 1
        print(f'===>Teacher: Congrats! We have done with learning.')
        conversation.append('Teacher: Congrats! We have done with learning.')
        return conversation


def main():
    count_conversation = 0
    docs = read_all_file_suffix_X(mdir=mdir, suffix='json')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    inference_model = config['llm']['model']
    for doc in docs:
        task = doc['title'].replace('How to ', '')
        topic = doc['categories'][-1].replace(' ','-') if doc['categories'][-1]!='Categories' else task
        conversation_path = f'{mdir}/{topic}/{task}/{inference_model}-single_{task}'
        summary = doc['introduction']
        methods = doc['methods']
        # Multiple methods if title contains 'Section x of Y:'
        if re.match(r'Method \d+ of \d+:.*', methods[0]['title']):
            for i, method in enumerate(methods):
                count_conversation += 1
                tutorial_document = [summary+method['title']] + method['steps']
                # Initialize the agent
                agent = SingleAgent(tutorial_document=tutorial_document, config_file=config_file)
                conversation = agent.run_simulation()
                with open(f'{conversation_path}_{i}.txt', 'w') as fw:
                    fw.write('\n'.join(conversation).replace('"', ''))            
        # One method only
        else:
            count_conversation += 1
            tutorial_document = [summary] + [s['title']+s['steps'][0] for s in methods]
            agent = SingleAgent(tutorial_document=tutorial_document, config_file=config_file)
            conversation = agent.run_simulation()
            with open(f'{conversation_path}_0.txt', 'w') as fw:
                fw.write('\n'.join(conversation).replace('"', ''))        


if __name__ == "__main__":
    global mdir, config_file
    mdir='./data/wikihow'
    config_file = 'conf/ollma-llama3.yaml'
    main()