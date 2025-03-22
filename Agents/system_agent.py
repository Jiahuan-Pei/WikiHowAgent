from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from Agents.util import setup_llm_and_embeddings 

class TeacherLLMAgent:
    """
    A class representing an LLM-based teacher agent that teaches a topic step-by-step.
    The agent retrieves a tutorial/document, provides instructions step-by-step,
    and moves on based on user feedback ("understand" or "pass").
    """
    
    def __init__(self, config_file='conf/ollma-llama3.yaml', tutorial_document=""):
        """
        Initialize the TeacherLLMAgent with a language model, memory, and tutorial document.
        
        :param model_name: The name of the language model to use (default: "gpt-4").
        :param temperature: The temperature setting for the LLM's creativity (default: 0.7).
        :param tutorial_document: A string representing the tutorial/document to be used for teaching.
        """
        self.tutorial_document = tutorial_document
        self.current_step = 0  # Start at the first step in the tutorial
        
        # Set up the language model, memory, and prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["current_step", "tutorial_document"],
            template=(
                "You are a teacher providing step-by-step instructions for learning. "
                "Given the tutorial document: {tutorial_document}, provide the instruction for step {current_step}. "
                "Be clear, concise, and engaging in your explanation."
            ),
        )
        
        self.memory = ConversationBufferMemory(memory_key="context", input_key="current_step")
        self.llm, _, self.params = setup_llm_and_embeddings(config_file)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=True
        )

    def retrieve_step(self) -> str:
        """
        Retrieves the current step of the tutorial based on the current step index.
        """
        # Here you would implement document retrieval logic based on the tutorial_document
        # For this example, we simulate the steps from the tutorial document.
        step = f"Step {self.current_step + 1}: Explain the concept or action at this step."
        return step

    def get_instruction(self) -> str:
        """
        Generate instruction for the current step using the LLM agent.
        """
        # Ensure that current_step and tutorial_document are strings before passing them to the LLMChain
        current_step_str = str(self.current_step + 1)  # Convert to string
        tutorial_document = self.tutorial_document  # Make sure it's a string
        
        instruction = self.chain.run(current_step=current_step_str, tutorial_document=tutorial_document)
        return instruction
    
    def process_feedback(self, feedback: str) -> bool:
        """
        Process the user's feedback. Move to the next step if feedback indicates understanding or 'pass'.
        
        :param feedback: User feedback ("understand" or "pass").
        :return: True if the agent should move to the next step, False otherwise.
        """
        if feedback.lower() in ["understand", "pass"]:
            self.current_step += 1  # Move to the next step
            return True
        return False

    def teach(self, feedback: str = "") -> str:
        """
        Initiates the teaching process. First retrieves and generates the current step's instruction.
        Processes feedback and moves on to the next step accordingly.
        
        :param feedback: The user's feedback (if any).
        :return: The agent's response (instruction or confirmation of feedback processing).
        """
        if feedback:  # Process any feedback first
            if self.process_feedback(feedback):
                return self.get_instruction()  # Proceed with the next step instruction
            else:
                return "Please provide a valid response, such as 'understand' or 'pass'."
        
        # No feedback yet, so generate the current instruction
        return self.get_instruction()

# Example usage
if __name__ == "__main__":
    # Initialize the teacher agent with a dummy tutorial document
    tutorial_document = "This tutorial covers the basic concepts of Python programming."
    
    teacher_agent = TeacherLLMAgent(tutorial_document=tutorial_document)
    
    # Simulate teaching step-by-step
    while teacher_agent.current_step < 5:  # Example: Tutorial with 5 steps
        instruction = teacher_agent.teach()
        print(f"Instruction (Step {teacher_agent.current_step + 1}): {instruction}")
        
        # Simulate user feedback
        feedback = "understand"  # User understands the current step
        print(f"User Feedback: {feedback}")
        
        # Pass feedback to the agent
        response = teacher_agent.teach(feedback)
        print(f"Agent Response: {response}\n")
