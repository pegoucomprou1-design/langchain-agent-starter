# app.py
import os
import json
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ----------------------------
# Config / Env
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_MAIN = os.getenv("MODEL_MAIN", "gpt-4o-mini")
N8N_BASE = os.getenv("N8N_BASE", "").rstrip("/")  # ex: https://seu-n8n.com/webhook
N8N_TOOL_TOKEN = os.getenv("N8N_TOOL_TOKEN", "")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")
if not N8N_BASE:
    raise RuntimeError("Falta N8N_BASE (ex: https://seu-n8n.com/webhook)")
if not N8N_TOOL_TOKEN:
    logging.warning("N8N_TOOL_TOKEN não definido — se o n8n exigir Authorization vai falhar.")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent-app")

# ----------------------------
# Helpers (chamada ao n8n)
# ----------------------------
def call_n8n(path: str, payload: Dict[str, Any]) -> str:
    """
    Chama um webhook do n8n (POST JSON) e retorna o texto de resposta.
    """
    url = f"{N8N_BASE}/{path.lstrip('/')}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {N8N_TOOL_TOKEN}",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        # Tente extrair texto útil
        try:
            data = resp.json()
        except Exception:
            data = {"text": resp.text}

        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail={
                    "tool": path,
                    "status": resp.status_code,
                    "response": data,
                },
            )
        # padroniza saída textual
        if isinstance(data, dict):
            # tenta chaves comuns
            for k in ("mensagem", "message", "texto", "text", "output"):
                if k in data and isinstance(data[k], (str, int, float)):
                    return str(data[k])
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Erro chamando n8n: %s", e)
        raise HTTPException(status_code=502, detail=f"Erro chamando n8n/{path}: {e}")

# ----------------------------
# Tools
# ----------------------------
def tool_preco(a: Dict[str, Any]) -> str:
    """
    Gera a mensagem de preço com ancoragem no n8n.
    Espera no payload: lead_id, instancia, produto, contexto, etc.
    """
    return call_n8n("preco", a)

def tool_enviar_msg(a: Dict[str, Any]) -> str:
    """
    Envia mensagem final ao cliente pelo n8n.
    Espera: lead_id, instancia, texto (mensagem formatada), etc.
    """
    return call_n8n("enviarmsg", a)

TOOLS = [
    Tool(
        name="preco",
        description=(
            "Use esta ferramenta para responder dúvidas de preço/promoção/"
            "valores. Ela retorna um texto pronto de preço."
        ),
        func=lambda a: tool_preco(a if isinstance(a, dict) else {}),
    ),
    Tool(
        name="enviar_msg",
        description=(
            "Use esta ferramenta para enviar uma mensagem final pronta ao cliente. "
            "Passe lead_id, instancia e o campo 'texto' com a mensagem."
        ),
        func=lambda a: tool_enviar_msg(a if isinstance(a, dict) else {}),
    ),
]

# ----------------------------
# LLM + Agent
# ----------------------------
SYSTEM = (
    "Você é um agente de vendas do Aliviozon. Seja direto, "
    "educado e persuasivo. Quando o usuário perguntar preço ou "
    "promoções, use a ferramenta 'preco'. Quando for para disparar "
    "uma mensagem final ao cliente, use 'enviar_msg'. "
    "Se precisar, peça detalhes faltantes de forma objetiva."
)

llm = ChatOpenAI(model=MODEL_MAIN, temperature=0.3, api_key=OPENAI_API_KEY)

agent = initialize_agent(
    TOOLS,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={"system_message": SYSTEM},
)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Langchain Agent App", version="1.0.0")

class AgentPayload(BaseModel):
    lead_id: str = Field(..., description="ID do lead")
    instancia: Optional[str] = Field(None, description="Instância/rota do provedor (ex: evolution-7)")
    mensagem: str = Field(..., description="Texto do cliente")
    contexto: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexto adicional (produto, etc.)")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/agent")
def agent_endpoint(body: AgentPayload):
    """
    Invoca o agente. Importante: o AgentExecutor espera a chave 'input'.
    Para simplificar, embalo lead/contexto em um texto único, e também
    passo um objeto para as ferramentas (elas recebem dict).
    """
    try:
        # 1) Entrada textual para o AGENTE
        user_input = (
            f"lead_id={body.lead_id}; "
            f"instancia={body.instancia or ''}; "
            f"contexto={json.dumps(body.contexto or {}, ensure_ascii=False)}; "
            f"pergunta={body.mensagem}"
        )

        # 2) Também preparo um objeto que as TOOLS possam usar
        tool_context = {
            "lead_id": body.lead_id,
            "instancia": body.instancia,
            "mensagem": body.mensagem,
            "contexto": body.contexto or {},
        }

        # Chame o agente passando a chave 'input' (obrigatória no AgentExecutor)
        # e injete o dict como se fosse "memory extra" via tags/metadata se precisar.
        # Aqui, uma forma prática: deixar o texto no 'input'
        # e, quando a tool for chamada, o agente passa 'tool_input'; nossas tools aceitam dict.
        result = agent.invoke({"input": user_input, **tool_context})

        # Algumas versões retornam dict {'output': '...'}
        if isinstance(result, dict) and "output" in result:
            text = str(result["output"])
        else:
            text = str(result)

        return {"ok": True, "text": text}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Erro no /agent: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
