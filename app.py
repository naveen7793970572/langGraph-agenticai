import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. SETUP
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


# ==========================================
# 2. MINIMAL GRAPH
# ==========================================
class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    output: str


def simple_agent(state: CrewState):
    task = state['messages'][-1].content
    response = llm.invoke(f"You are an AI Developer. Solve this: {task}")

    # AI format etla unna, daanni clear string ga marchadaniki safe extract chestunnam
    content = response.content
    if isinstance(content, list):
        clean_text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
    else:
        clean_text = str(content)

    return {"output": clean_text}


workflow = StateGraph(CrewState)
workflow.add_node("agent", simple_agent)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
rt_app = workflow.compile()

# ==========================================
# 3. FASTAPI & CLEAN UI
# ==========================================
app = FastAPI(title="Local AI Agent")


# Ee class dwara manam clean ga input theesukuntam (No JSON typing needed!)
class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    initial_state = {"messages": [HumanMessage(content=req.message)]}
    result = rt_app.invoke(initial_state)
    return {"response": result.get("output", "Error processing request")}


# Ee HTML manaki clean UI ni isthundi (Markdown formatting tho)
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Developer Assistant</title>
        <!-- Markdown parser library -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 40px; color: #333;}
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            h2 { color: #2c3e50; text-align: center; margin-bottom: 20px;}
            textarea { width: 95%; height: 80px; padding: 15px; margin-bottom: 15px; border-radius: 8px; border: 1px solid #dcdde1; font-size: 16px; font-family: inherit; resize: vertical;}
            button { width: 100%; padding: 12px; background: #2980b9; color: white; border: none; cursor: pointer; border-radius: 8px; font-size: 16px; font-weight: 600; transition: background 0.3s; }
            button:hover { background: #3498db; }
            #outputArea { margin-top: 30px; padding: 25px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e1e8ed; min-height: 100px; font-size: 15px; line-height: 1.6; display: none;}
            /* Code block styling */
            pre { background: #2d3436; color: #f5f6fa; padding: 15px; border-radius: 8px; overflow-x: auto; }
            code { font-family: 'Courier New', Courier, monospace; }
            .loading { text-align: center; color: #2980b9; font-weight: bold; margin-top: 10px; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>💻 AI Developer Assistant</h2>
            <textarea id="taskInput" placeholder="Enter your task here... (e.g., Write a SQL query for a natural join, or generate test cases for a login framework)"></textarea>
            <button onclick="runTask()">Generate Solution</button>
            <div id="loading" class="loading">Generating response... ⏳</div>
            <div id="outputArea"></div>
        </div>

        <script>
            async function runTask() {
                const task = document.getElementById('taskInput').value;
                if (!task) return;

                const outputArea = document.getElementById('outputArea');
                const loading = document.getElementById('loading');

                outputArea.style.display = 'none';
                loading.style.display = 'block';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: task })
                    });

                    const data = await response.json();

                    // Convert raw markdown to clean HTML
                    outputArea.innerHTML = marked.parse(data.response);
                    outputArea.style.display = 'block';
                } catch (error) {
                    outputArea.innerHTML = "<p style='color:red;'>Error connecting to AI.</p>";
                    outputArea.style.display = 'block';
                }
                loading.style.display = 'none';
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 CLEAN UI API is running locally!")
    print("👉 OPEN THIS LINK IN BROWSER: http://localhost:8000/")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)