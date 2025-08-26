import os, requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

# ---------- ENV ----------
N8N_BASE = os.getenv("N8N_BASE_URL")              # ex: https://n8n.seudominio.com/webhook
N8N_TOOL_TOKEN = os.getenv("N8N_TOOL_TOKEN")      # token opcional p/ autenticar seus webhooks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # sua key da OpenAI
MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-4o-mini")

def call_tool(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if N8N_TOOL_TOKEN:
        headers["Authorization"] = f"Bearer {N8N_TOOL_TOKEN}"
    r = requests.post(f"{N8N_BASE}{path}", json=payload, headers=headers, timeout=30)
    if not r.ok:
        raise HTTPException(r.status_code, f"Tool {path} error: {r.text}")
    return r.json()

# ---------- LLM ----------
LLM = ChatOpenAI(model=MODEL_MAIN, temperature=0.3, api_key=OPENAI_API_KEY)

# ---------- TOOLS ----------
TOOLS = [
    Tool(
        name="preco",
        func=lambda a: call_tool("/tool/preco", a),
        description="Gera a mensagem de preço com ancoragem. Input: {lead_id, produto, contexto}. Output: {mensagem}"
    ),
    Tool(
        name="enviar_msg",
        func=lambda a: call_tool("/tool/enviar-msg", a),
        description="Envia a mensagem final ao cliente. Input: {lead_id, instancia, texto}. Output: {message_id, ok}"
    ),
]

SYSTEM = """Você é um agente de vendas do Aliviozon.
- Sempre responda de forma natural e persuasiva.
- Quando o cliente pedir preço, use a tool 'preco' para obter os valores.
- Use a saída como base e adapte o tom ao cliente (sem alterar números).
- Finalize SEMPRE chamando 'enviar_msg' com o texto final para o cliente.
"""

app = FastAPI()

class AgentIn(BaseModel):
    lead_id: str
    mensagem: str
    instancia: Optional[str] = None
    contexto: Optional[Dict[str, Any]] = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/agent")
def agent(inp: AgentIn):
    agent = initialize_agent(TOOLS, LLM, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    prompt = f"""{SYSTEM}

LEAD_ID: {inp.lead_id}
INSTANCIA: {inp.instancia}
MENSAGEM_CLIENTE: {inp.mensagem}
CONTEXTO: {inp.contexto or {}}

Tarefa:
1) Se a intenção for preço, chame a tool 'preco' para obter a base de valores.
2) Reescreva de forma natural/persuasiva (sem alterar números).
3) Finalize chamando 'enviar_msg' com o texto final (campo 'texto').
"""
    result = agent.run(prompt)
    return {"ok": True, "trace": str(result)}