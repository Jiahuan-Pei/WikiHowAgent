from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.util import setup_llm_and_embeddings
import re

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
                "- Answer learner questions related to the instructions **as the first priority**. \n"
                "- If the learner requests to move on without asking questions, provide the current step’s instruction directly.\n"
                "- Once the final step has been completed and the learner has no further questions, acknowledge completion by appending the token 'FINISHED' to your response.\n"
                "- Highlight key phrases in your response if they appear in the tutorial.\n\n"
                 "📝 Respond in character as a teacher — no system-level messages or meta-comments. Keep responses short and focused.\n"
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
        teacher_response = teacher_response.encode('utf-8').decode('unicode_escape')
        # Removing non-ASCII characters using regex
        teacher_response = re.sub(r'[^\x00-\x7F]+', '', teacher_response)
        print('[[🤖 Teacher]]', teacher_response)
        return teacher_response


    def teach(self, data: dict) -> str:
        return
    
    def answer(self, data: dict) -> str:
        return
    
    def humanize(self, data: dict) -> str:
        """Process the teacher's utterance human like."""
        return