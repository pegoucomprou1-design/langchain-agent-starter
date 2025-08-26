# LangChain Agent Starter

## Endpoints
- `GET /healthz` → health check
- `POST /agent` → recebe lead e decide resposta

## Exemplo payload
```json
{ 
  "lead_id":"123", 
  "instancia":"evolution-7", 
  "mensagem":"quanto custa?", 
  "contexto":{"produto":"Aliviozon"} 
}