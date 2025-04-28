# app_user_role.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from manager.multiple_agent_manager_workflow import teacher_node, learner_node, dialogue_policy, LearningState

# Example tutorial
EXAMPLE_TUTORIAL = {
    "summary": "How to make a peanut butter and jelly sandwich.",
    "tutorial": [
        "Get two slices of bread.",
        "Spread peanut butter on one slice.",
        "Spread jelly on the other slice.",
        "Put the slices together to make a sandwich.",
        "Cut the sandwich in half, if desired."
    ]
}

st.set_page_config(page_title="Tutor Chat", layout="centered")
st.title("🧑‍🏫 Interactive AI Tutor App")
st.markdown("Choose your role and interact in a real-time conversation!")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

# Role Selection
role = st.radio("Select your role:", ["Learner", "Teacher"])

# Initialize state
if "state" not in st.session_state:
    st.session_state.state = LearningState(
        messages=[],
        summary=EXAMPLE_TUTORIAL["summary"],
        tutorial=EXAMPLE_TUTORIAL["tutorial"],
        current_step_index=0,
        needs_clarification=False,
        finished=False,
    )

state = st.session_state.state

# Show conversation
st.markdown("### Tutorial")
st.markdown(EXAMPLE_TUTORIAL)

# Show conversation
st.markdown("### 🗨️ Conversation History")
# for msg in state["messages"]:
#     with st.chat_message("Teacher" if isinstance(msg, AIMessage) else "Learner"):
#         st.markdown(msg.content)

# Role-specific chat input
if not state["finished"]:
    # last_msg = state["messages"][-1] if state["messages"] else None
    # is_turn = (role == "Learner" and (last_msg is None or isinstance(last_msg, AIMessage))) or \
    #           (role == "Teacher" and isinstance(last_msg, HumanMessage))

    # if is_turn:
    user_input = st.chat_input(f"Your response as {role}:")
    if user_input:
        if role == "Learner":
            state["messages"].append(HumanMessage(content=user_input))
            state["needs_clarification"] = "?" in user_input.lower() and "next" not in user_input.lower()
        elif role == "Teacher":
            state["messages"].append(AIMessage(content=user_input))
        # for msg in state["messages"]:
        #     with st.chat_message("Teacher" if isinstance(msg, AIMessage) else "Learner"):
        #         st.markdown(msg.content)        
        # with st.chat_message("Teacher" if role == "Teacher" else "Learner"):
        #     last_msg = state["messages"][-1] if state["messages"] else None
        #     st.markdown(last_msg.content)
    # Automatic agent response
    next_node = dialogue_policy(state)
    if next_node == "__end__":
        state["finished"] = True
    elif next_node == "teacher" and role == "Learner":
        state = teacher_node(state)
        # last_msg = state["messages"][-1] if state["messages"] else None
        # with st.chat_message("Teacher"):
        #     st.markdown(last_msg.content)
    elif next_node == "learner" and role == "Teacher":
        state = learner_node(state)
        # last_msg = state["messages"][-1] if state["messages"] else None
        # with st.chat_message("Learner"):
        #     st.markdown(last_msg.content)
    for msg in state["messages"]:
        with st.chat_message("Teacher" if isinstance(msg, AIMessage) else "Learner"):
            st.markdown(msg.content) 
# Tutorial complete
if state["finished"]:
    with st.chat_message("System"):
        st.success("✅ Tutorial finished! Well done.")

# Update session state
st.session_state.state = state
