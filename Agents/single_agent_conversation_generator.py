import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils import setup_llm_and_embeddings
import re


class SingleAgent:
    def __init__(self, tutorial_document: dict, config_file: str):
        self.tutorial_document = tutorial_document
        self.total_steps=len(tutorial_document["steps"])
        self.current_step = 0

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
                    f"Tutorial context: {self.tutorial_document['steps'][self.current_step]}\n"
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
            f"Tutorial step: {self.tutorial_document['steps'][self.current_step]}\n"
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
        print("-"*100, "\n", "Starting Conversation Simulation\n")
        while self.current_step < self.total_steps:
            ### Step instruction
            step_instruction = self.tutorial_document["steps"][self.current_step]
            step_instruction = re.sub(r'\n+', '\n', step_instruction).strip()
            print('*'*50, '\n', f"Step {self.current_step + 1}: {step_instruction}\n", '*'*50)
            
            ### Teacher instruction
            teacher_instruction = self.teacher_explain_step(step_instruction)
            teacher_instruction = re.sub(r'\n+', '\n', teacher_instruction).strip()
            print(f"===>Teacher:  {teacher_instruction}")
            
            ### Learner feedback
            # learner_input = input("Learner: ")  # Simulate learner input
            learner_input = self.learner_request(teacher_instruction)
            learner_input = re.sub(r'\n+', '\n', learner_input).strip()
            print(f"<=Learner: {learner_input}")
            if "?" in learner_input:
                teacher_response = self.teacher_response(learner_input)
                teacher_response = re.sub(r'\n+', '\n', teacher_response).strip()
                print(f"===>Teacher: {teacher_response}")

            ### Simply assuming the teacher answered the learner's question
            self.current_step += 1
        print(f"===>Teacher: Congrats! We have done with learning.")


def main():
    tutorial_document = {
        "id": "aabbcc",
        "steps": [
            "Fill a container with salt. A jug or pitcher, a deep bowl, a plastic food container, etc. will all suffice.",
            "Squeeze a little tempera paint into the salt.",
            "Mix with a spoon or other item. Stir until the paint is evenly distributed through the salt.",
            "Let stand overnight to dry. Make as many more colors as you need for your project. That way, they'll all be ready at the same time.",
            "Test before using. Check that the salt has dried before using in your craft, rangoli, teaching, etc. projects."
        ]
    }

    # Initialize the agent
    agent = SingleAgent(tutorial_document=tutorial_document, config_file='conf/ollma-llama3.yaml')
    agent.run_simulation()


if __name__ == "__main__":
    main()