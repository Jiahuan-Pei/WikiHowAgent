from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings


# Teacher Agent
class TeacherLLMAgent:
    def __init__(self, config):
        super().__init__()
        self.llm, _, self.config = setup_llm_and_embeddings(config)

        self.prompt_template = PromptTemplate(
            input_variables=[ "summary", "current_step_index", "current_step_content", "user_utterance"],
            template=(
                "You are an expert 🤖 teacher guiding a learner step by step through a tutorial via multi-turn conversations.\n\n"
                "🎯 Your role:\n"
                "- Provide clear instructions for the current step.\n"
                "- Answer any learner questions.\n"
                "- If the learner asks a question **before you’ve given detailed instructions** or something **irrelevant**, gently guide them back to the tutorial. \n"
                "- Acknowledge completion by appending the token 'FINISHED' to the response if the final step has been learned and the student has no further questions.\n"
                "- Highlight key phrases in your response if they appear in the tutorial.\n\n"
                "📖 Tutorial Summary: {summary}\n"
                "🔹 Step {current_step_index}: {current_step_content}\n"
                "Learner: {user_utterance}\n"
                "Teacher:"
            ),
        )

        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm, verbose=True)
        # self.chain =  RunnableSequence(self.prompt_template, self.llm)

    def respond(self, data: dict) -> str:
        """Process the learner's response and generate the teacher's reply."""
        # Generate teacher response
        data = {
            'summary': data['summary'], 
            'current_step_index': data['current_step_index'],
            'current_step_content': data['current_step_content'],
            'user_utterance': data['user_utterance']
        }
        teacher_response = self.chain.invoke(data)['text']
        print('[[🤖 Teacher]]', teacher_response)
        return teacher_response


    def teach(self, data: dict) -> str:
        return
    
    def answer(self, data: dict) -> str:
        return
    
    def humanize(self, data: dict) -> str:
        """Process the teacher's utterance human like."""
        return