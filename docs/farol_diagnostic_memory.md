# Memória de diagnóstico do Farol

## 1. Separação das memórias

Existem três coisas distintas:

1. **memória conversacional do projeto** — `data/projects/<id>.json`;
2. **RAM do turno** — mensagens selecionadas pela Sinapse para o prompt;
3. **memória de diagnóstico** — eventos técnicos append-only, fora do índice
   conversacional, da RAM e do prompt.

A memória de diagnóstico serve ao OpenCode para investigar o Farol; não serve
ao Farol para responder ao usuário. Ela não é carregada por `ProjectStore`,
`recall` ou `select_chat_context`.

## 2. Camada atual e pontos de instrumentação

Fluxo web/CLI:

1. requisição entra em `web_server` ou `cli`;
2. `ProjectStore.load` lê o projeto;
3. `ProjectStore.add_message` atualiza o estado derivado;
4. `resume`/`dashboard` selecionam decisão, teste, pendência e foco;
5. `chat_with_deepseek` filtra sensibilidade, monta `Message` e chama
   `SinapseChatbot`;
6. `SinapseChatbot.ask` executa `recall` e `select_chat_context`;
7. `build_base_messages`/`with_memory` montam o prompt;
8. `build_request_body` envia ao DeepSeek;
9. resposta é salva e devolvida.

Hoje não há `request_id`, eventos por etapa, nem separação explícita entre
estado estruturado e RAM. O diagnóstico deve ser adicionado nesses pontos sem
alterar o comportamento funcional.

## 3. Eventos estruturados

Armazenamento separado:

```text
data/diagnostics/farol-events.jsonl
```

Uma linha JSON por evento. Nunca dentro de `data/projects/`, nunca incluído no
prompt.

Schema mínimo:

```json
{
  "schema": "farol.diag.v1",
  "event_id": "evt_01J...",
  "request_id": "req_01J...",
  "project_id": "sinapse",
  "ts": "2026-11-17T12:00:00.123Z",
  "entrypoint": "web",
  "stage": "memory.budget_cut",
  "status": "ok",
  "latency_ms": 3.4,
  "error_code": null,
  "details": {
    "selected_ids": ["m5", "m7"],
    "rejected": [
      {"id": "m18", "reason": "below_threshold", "tokens": 22}
    ],
    "memory_budget": 160,
    "memory_tokens": 118
  }
}
```

Regras:

- `details` é uma allowlist de IDs, contagens, códigos e motivos;
- não registrar texto de mensagens, prompt, resposta, cabeçalhos, API key ou
  variáveis de ambiente;
- IDs de fontes restritas são substituídos por `"[redacted]"` com
  `restricted: true`;
- códigos de motivo são fechados: `selected`, `outside_radius`,
  `no_focus_evidence`, `below_threshold`, `duplicate`, `top_k_full`,
  `no_budget`, `already_in_base`, `restricted`;
- `error_code` é padronizado (`E_PROJECT_LOAD`, `E_STATE`, `E_RECALL`,
  `E_BUDGET`, `E_PROMPT`, `E_GENERATION`, `E_TIMEOUT`);
- retenção sugerida: 14 dias, rotação diária; nunca apagar eventos de um
  incidente aberto sem confirmação.

## 4. Eventos por etapa

| etapa | o que registra | o que não registra |
|---|---|---|
| `request.received` | request_id, projeto, entrypoint, tipo de pergunta | texto da pergunta |
| `project.loaded` | id, contagem de mensagens, versão do schema | conteúdo das mensagens |
| `state.loaded` | IDs de decisão/teste/pendência e status | texto do estado |
| `resume.selected` | IDs visíveis, IDs restritos, foco | textos |
| `sensitivity.filtered` | contagens liberadas/restritas, motivos | conteúdo restrito |
| `memory.recall` | seed, candidatos, foco, contagem recente | texto das memórias |
| `memory.scored` | id, rank, dentro do raio, atenção, evidência, tokens | texto |
| `memory.budget_cut` | selecionados, rejeitados com motivo, orçamento | texto |
| `prompt.assemble` | seções presentes, IDs no prompt, `pending_ids_in_prompt`, contagens | prompt completo |
| `generation.completed` | modo local/DeepSeek, tokens, finish_reason, latência, `cited_ids` | resposta |
| `response.returned` | status, modo, erro | corpo da resposta |

`cited_ids` é extraído apenas como IDs; não armazena a resposta.

## 5. Comando local de diagnóstico

Somente leitura, explícito, sem chamada ao modelo:

```powershell
python -m sinapse diagnostics request req_01J...
python -m sinapse diagnostics project sinapse --since 2026-11-17
python -m sinapse diagnostics report req_01J... --json
```

- `request` faz consulta exata por `request_id`;
- `project` lista eventos do projeto, exigindo `project_id`;
- `report` monta o relatório legível/JSON;
- não há agente em background nem execução automática;
- o comando não altera projetos, logs ou estado;
- um `request_id` pertence a um único projeto; filtros cruzados são rejeitados;
- qualquer texto eventual nos logs é tratado como dado, nunca instrução.

Busca por semelhança só depois de a consulta exata ser insuficiente: usar
assinatura estruturada (`stage` da primeira falha, `error_code`, motivos de
corte, presença de pendência), nunca similaridade sobre texto de conversa.

## 6. Relatório

```json
{
  "symptom": "A retomada não mostrou a pendência da ponte.",
  "request_id": "req_01J...",
  "project_id": "sinapse",
  "first_observable_failure": {
    "stage": "memory.budget_cut",
    "event_id": "evt_...",
    "code": "PENDING_EXCLUDED",
    "observed": "m18 rejeitada por below_threshold",
    "expected": "m18 presente no contexto"
  },
  "supporting_event_ids": ["evt_state", "evt_resume", "evt_recall", "evt_cut", "evt_prompt"],
  "timeline": [
    {"event_id": "evt_state", "stage": "state.loaded", "summary": "pending=m18"},
    {"event_id": "evt_resume", "stage": "resume.selected", "summary": "m18 visível"},
    {"event_id": "evt_cut", "stage": "memory.budget_cut", "summary": "m18 excluída"}
  ],
  "hypotheses": [
    {
      "statement": "A pendência ficou fora da RAM por atenção abaixo do limiar.",
      "confidence": "alta",
      "evidence": ["evt_recall", "evt_cut"]
    },
    {
      "statement": "O prompt também não injeta o estado estruturado do resume.",
      "confidence": "média",
      "evidence": ["evt_prompt"]
    }
  ],
  "gaps": ["O evento de prompt não registra a seção project_state."],
  "reproduction": "python -m sinapse project --root <root> resume"
}
```

O relatório separa fato sustentado por evento de hipótese. Nunca inclui texto
de memória restrita.

## 7. Exemplo de incidente

Pergunta: “Onde paramos?”

Sintoma: a resposta cita decisão e teste concluído, mas não menciona a
pendência “testar a ponte em 96 kHz”.

Eventos:

```json
{"event_id":"evt_1","stage":"state.loaded","details":{"pending_ids":["m18"]}}
{"event_id":"evt_2","stage":"resume.selected","details":{"pending_ids":["m18"],"restricted":false}}
{"event_id":"evt_3","stage":"memory.recall","details":{"candidate_ids":["m5","m7","m18"]}}
{"event_id":"evt_4","stage":"memory.scored","details":{"id":"m18","attention":0.02,"within_radius":true}}
{"event_id":"evt_5","stage":"memory.budget_cut","details":{"selected_ids":["m5","m7"],"rejected":[{"id":"m18","reason":"below_threshold"}]}}
{"event_id":"evt_6","stage":"prompt.assemble","details":{"pending_ids_in_prompt":[],"state_section_present":false}}
{"event_id":"evt_7","stage":"generation.completed","details":{"cited_ids":["m5","m7"],"finish_reason":"stop"}}
```

Diagnóstico:

- **primeira falha observável:** `memory.budget_cut`, evento `evt_5`;
- **sustentação:** `evt_1` prova que a pendência existia; `evt_2` que estava
  visível; `evt_3` que entrou como candidata; `evt_4`/`evt_5` explicam a
  exclusão; `evt_6` mostra que o prompt também não tinha o estado estruturado;
- **hipótese:** a atenção de `m18` ficou abaixo do limiar; o prompt sem
  `project_state` é uma segunda lacuna, não a primeira falha;
- **reprodução:** repetir a pergunta com o mesmo projeto e inspecionar
  `resume --json` e `diagnostics request`.

## 8. Testes planejados

- o agente de diagnóstico só executa quando o comando é chamado;
- eventos não aparecem na conversa, no `resume`, na RAM ou no prompt;
- logs ficam fora de `data/projects` e não são lidos por `ProjectStore`;
- filtro por `request_id` e `project_id` isola projetos;
- `request_id` de outro projeto é rejeitado;
- chave, headers, corpos de requisição/resposta e texto de memória não aparecem
  nos eventos;
- IDs de fontes restritas são redigidos;
- cada etapa registra IDs e motivos de inclusão/exclusão;
- o relatório identifica a primeira falha e separa hipótese de fato;
- o comando é somente leitura e não faz chamadas ao DeepSeek;
- rotação/retenção não removem eventos de um incidente aberto.
