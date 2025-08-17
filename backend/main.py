from graph import build_graph

def run_chat():
    app = build_graph()
    state = {"messages": [], "want_exit": False}
    while True:
        state = app.invoke(state)
        last = state["messages"][-1]
        print(f"{last.type}: {last.content}")
        if state.get("want_exit", False):
            break

if __name__ == "__main__":
    run_chat()
