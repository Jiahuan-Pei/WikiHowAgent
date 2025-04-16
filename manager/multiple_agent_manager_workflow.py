from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Dict, Literal
from langchain_core.runnables import RunnableLambda
from config import args, config_teacher, config_learner, config_evaluator, PINK_COLOR, RESET_COLOR, logger

# Import Agents
from agents.teacher_agent import TeacherLLMAgent
from agents.learner_agent import LearnerLLMAgent
from agents.evaluator_agent import ConversationEvaluator


class LearningState(TypedDict):
    """State representation of the learning process.
    The state is a dictionary holding everything the agents need to continue the conversation:
    - messages: the entire history
    - current_step_index: progress through the tutorial
    - needs_clarification: whether learner needs clarification
    - finished: whether conversation should end
    - next_node: decision routing
    💡 Update state based on the effect of the node you're in, not based on what the next node will do.
    > The teacher node should update current_step_index, finished, needs_clarification
    > The learner node should update needs_clarification (based on whether the response has a ?)
    > The manager or dialogue_policy only decides routing — it shouldn't mutate state
    """
    messages: Annotated[List, add_messages]  # Conversation history
    summary: str
    tutorial: List[str]
    current_step_index: int
    needs_clarification: bool
    finished: bool

def teacher_node(state: LearningState) -> LearningState:
    messages = state["messages"]
    summary = state["summary"]
    tutorial = state["tutorial"]
    current_step_index = state["current_step_index"]
    finished = False
    teacher_agent = TeacherLLMAgent(config=config_teacher)

    last_message = messages[-1] if messages else None
    needs_clarification = False
    response_parts = []

    # Check if learner asked a question
    if last_message and isinstance(last_message, HumanMessage):
        needs_clarification = '?' in last_message.content and 'next step' not in last_message.content

    # If this is the very beginning, start the tutorial
    if current_step_index == 0 and last_message is None:
        instruction = tutorial[0]
        response_parts.append(f"Let's begin: {instruction}. BEGIN")
        current_step_index += 1  # move to next step
    # If tutorial is complete
    elif current_step_index >= len(tutorial):
        if needs_clarification and last_message:
            clarification = teacher_agent.respond({
                "summary": summary,
                "current_step_index": current_step_index - 1,
                "current_step_content": tutorial[-1],
                "user_utterance": last_message.content
            })
            response_parts.append(clarification)
        response_parts.append("We've reached the end of the tutorial. Bye! FINISHED")
        finished = True
        needs_clarification = False
    else:
        # If learner has a question, respond to it, but do NOT advance the step
        if needs_clarification and last_message:
            clarification = teacher_agent.respond({
                "summary": summary,
                "current_step_index": current_step_index,
                "current_step_content": tutorial[current_step_index],
                "user_utterance": last_message.content
            })
            response_parts.append(clarification)
        else:
            # Provide the next instruction, then advance
            instruction = teacher_agent.respond({
                "summary": summary,
                "current_step_index": current_step_index,
                "current_step_content": tutorial[current_step_index],
                "user_utterance": last_message.content if last_message else ""
            })
            response_parts.append(instruction)
            current_step_index += 1  # advance only if no question
            needs_clarification = False

    messages.append(AIMessage(content="\n".join(response_parts)))

    return {
        **state,
        "messages": messages,
        "current_step_index": current_step_index,
        "needs_clarification": needs_clarification,
        "finished": finished,
    }




def learner_node(state: LearningState) -> LearningState:
    """Handles learner responses and understanding assessment."""
    messages = state["messages"]
    learner_agent = LearnerLLMAgent(config=config_learner)
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage):
        response = learner_agent.respond({"instruction": last_message.content})
        messages.append(HumanMessage(content=response))
        return {**state, "messages": messages}
    
    return state  # No valid teacher message to respond to



def dialogue_policy(state: LearningState) -> Literal["teacher", "learner", "__end__"]:
    """Determines the next action: 'teacher', 'learner', or '__end__'."""
    # 🧠 Sanity logs to debug loops
    print("📍 Routing decision checkpoint")
    print("🧠 Current step index:", state["current_step_index"], "/", len(state["tutorial"]))
    print("📚 Total messages so far:", len(state["messages"]))
    print("❓ Needs clarification:", state["needs_clarification"])
    print("✅ Finished flag:", state["finished"])

    # Core routing logic
    if state["finished"]:
        return "__end__"
    
    if len(state["messages"]) >= args.max_interaction:
        print("⚠️ Max interaction hit!")
        return "__end__"
    
    if state["needs_clarification"]:
        print("📍 Need clarification!")
        return "teacher"
    
    last_message = state["messages"][-1] if state["messages"] else None
    return "teacher" if isinstance(last_message, HumanMessage) else "learner"


def generate_single_conversation(summary: str, tutorial: List[str]) -> List[str]:
    """Simulates a structured learning conversation."""
    graph_builder = StateGraph(LearningState)
    graph_builder.add_node("teacher", teacher_node)
    graph_builder.add_node("learner", learner_node)
    graph_builder.add_conditional_edges("learner", dialogue_policy, {"__end__": END, "teacher": "teacher"})
    graph_builder.add_edge("teacher", "learner")
    graph_builder.add_edge(START, "teacher")

    chat_graph = graph_builder.compile().with_config(recursion_limit=100)
    
    if args.plot_agent_workflow:
        try:
            image_data = chat_graph.get_graph().draw_mermaid_png()
            with open("figure/chat_graph_simulation_workflow.png", "wb") as f:
                f.write(image_data)
        except Exception as e:
            logger.warning(f"Graph visualization failed: {e}")
    
    state = LearningState(messages=[], summary=summary, tutorial=tutorial, current_step_index=0, needs_clarification=False, finished=False)

    final_state = None
    for chunk in chat_graph.stream(state):
        if END in chunk:
            final_state = chunk[END]
        else:
            node = list(chunk.keys())[0]
            final_state = chunk[node]

    all_messages = final_state["messages"] if final_state else []
    return [f"{'Teacher' if isinstance(msg, AIMessage) else 'Learner'}: {msg.content.strip()}" for msg in all_messages]

def process_method(data: Dict) -> Dict:
    """Processes and evaluates a tutorial conversation."""
    method, summary = data.get("tutorial", {}), data.get("summary", "")
    method_id, source_path = data.get("method_id"), data.get("source_tutorial_path")

    if isinstance(method, dict) and "steps" in method:
        tutorial = [method.get("title")] + method["steps"]
    elif isinstance(method, list):
        tutorial = method
    else:
        logger.warning(f"Invalid tutorial format for Method ID: {method_id} | Source: {source_path}")
        return None

    conversation = generate_single_conversation(summary, tutorial)
    evaluation_results = ConversationEvaluator(config_evaluator).evaluate(conversation=conversation, tutorial=tutorial)
    # logger.info('----- Full Conversation -----\n')
    # logger.info('\n'.join(conversation))
    # logger.info('----- Evaluation Result -----\n')
    # logger.info(evaluation_results)
    return {**data, "conversation": conversation, "evaluation": evaluation_results}


def process_batch_method(batch_data: List[Dict]) -> tuple:
    """Processes multiple tutorials in parallel."""
    runnable = RunnableLambda(process_method)
    results = runnable.batch(batch_data)
    return ([res["conversation"] for res in results], [res["evaluation"] for res in results])
