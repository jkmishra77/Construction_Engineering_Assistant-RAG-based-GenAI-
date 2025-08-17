from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from backend.llm import llm

class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    want_exit: bool

def chat_start(state: ChatState):
    if not state["messages"]:
        welcome_msg = AIMessage(content="Hello! I'm your AI assistant. How can I help you today?")
        return {"messages": [welcome_msg], "want_exit": False}
    
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        return {"messages": state["messages"], "want_exit": True}
    
    return {"messages": state["messages"] + [HumanMessage(content=user_input)], "want_exit": False}

def chatbot_node(state: ChatState):
    if isinstance(state["messages"][-1], HumanMessage):
        ai_response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [ai_response], "want_exit": state.get("want_exit", False)}
    return state

def exit_router(state: ChatState):
    return "end_conversation" if state.get("want_exit", False) else "continue_chat"

def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat_start", chat_start)
    graph.add_node("chatbot_node", chatbot_node)
    graph.add_edge(START, "chat_start")
    graph.add_edge("chat_start", "chatbot_node")
    graph.add_conditional_edges("chatbot_node", exit_router, {
        "continue_chat": "chat_start",
        "end_conversation": END
    })
    return graph.compile()
