# Inferência de intenção e quadro antecipatório do Farol

> **Nota (2026-11-17):** exemplos musicais preservados como linha de base. O
> Farol atual seleciona contexto técnico por tipo de pedido (bug, arquitetura,
> teste, casual) em `sinapse/software_context.py`; o quadro antecipatório
> continua fora do prompt.

## 1. Dois quadros por projeto

Cada projeto tem dois quadros lógicos:

| quadro | conteúdo | vai ao DeepSeek? | persistência |
|---|---|---|---|
| principal | memórias que sustentam o pedido atual | **sim** | temporário do turno |
| antecipatório | referências a memórias potencialmente úteis no próximo turno | **não** | por projeto, com expiração |

O quadro antecipatório não é estado confirmado, não altera decisões e não
executa ações. Ele apenas mantém hipóteses de continuação e IDs de origem.

A memória de diagnóstico do Farol continua fora dos dois quadros.

## 2. Previsão de intenção

```json
{
  "prediction_id": "pred_01",
  "project_id": "sinapse",
  "likely_intent": "pedir as configurações de exportação",
  "confidence": "media",
  "clues": [
    {"type": "user_message", "id": "m10", "signal": "CPU em 27%, abaixo de 30%"},
    {"type": "condition", "id": "m9", "signal": "abaixo de 30% → exportar"}
  ],
  "source_ids": ["m24", "m25"],
  "reevaluate_when": "próximo pedido do usuário, troca de projeto, contradição ou expiração",
  "status": "hypothesis",
  "expires_after_turns": 1
}
```

Campos:

- `likely_intent`: hipótese de intenção, não instrução;
- `confidence`: `baixa`, `media` ou `alta` baseada apenas nas pistas;
- `clues`: mensagens/condições observáveis que sustentam a previsão;
- `source_ids`: memórias em reserva que poderiam ser úteis;
- `reevaluate_when`: gatilho de invalidação;
- `status`: sempre `hypothesis`;
- `expires_after_turns`: evita previsões antigas contaminando o futuro.

O pedido explícito do usuário tem prioridade absoluta. A previsão nunca altera
decisão, pendência, checklist, foco ou qualquer ação.

## 3. Quadros

```json
{
  "project_id": "sinapse",
  "turn_id": "turn_42",
  "main_board": {
    "ids": ["m5", "m9", "m10", "m18"],
    "tokens": 96,
    "purposes": ["decision", "reason", "condition", "pending"]
  },
  "anticipatory_board": {
    "predictions": [
      {
        "prediction_id": "pred_01",
        "likely_intent": "pedir como exportar",
        "confidence": "media",
        "clues": [{"type": "condition", "id": "m9"}],
        "source_ids": ["m24", "m25"],
        "reevaluate_when": "próximo pedido sobre exportação",
        "status": "hypothesis",
        "expires_after_turns": 1
      }
    ],
    "reserved_ids": ["m24", "m25"],
    "search_budget": {
      "max_rounds": 1,
      "max_candidates": 8,
      "max_time_ms": 150
    },
    "used": {"rounds": 1, "candidates": 5, "time_ms": 62}
  }
}
```

O quadro antecipatório tem IDs e previsões, mas **não** é anexado ao prompt.

## 4. Promoção ao quadro principal

Uma memória antecipada só é promovida se todas as condições forem verdadeiras:

1. **utilidade para o pedido atual** — responde diretamente ao `current_task`
   ou a uma dependência necessária;
2. **validade no estado do projeto** — decisão/pendência atual, não uma versão
   substituída;
3. **sensibilidade** — fonte permitida; conteúdo restrito nunca é promovido;
4. **orçamento** — cabe no orçamento de tokens do quadro principal;
5. **referência resolvida** — IDs existem no projeto ativo e não pertencem a
   outro projeto.

Sem todos os critérios, a memória permanece em reserva e é marcada como
`not_promoted` com o motivo.

## 5. Orçamentos separados

Busca antecipatória:

- `max_rounds`: 1;
- `max_candidates`: 8;
- `max_time_ms`: 150;
- executa depois que o quadro principal está pronto e nunca bloqueia a
  resposta atual; se estourar o tempo, o turno segue só com o principal.

Quadro principal:

- `token_budget`: 160;
- `tokens_used`: medido pela seleção Sinapse;
- `promote` só consome tokens se a memória for útil ao pedido atual.

O trabalho antecipatório não altera o orçamento do quadro principal; apenas
consome o limite próprio de busca.

## 6. Exemplo — ponte com 27% de CPU

Pedido atual: “A CPU ficou em 27%.”

### Quadro principal

```json
{
  "ids": ["m5", "m9", "m18", "m10"],
  "purposes": ["decision", "reason", "condition", "pending"],
  "tokens": 96
}
```

- `m5`: decisão vigente Pigments e motivo CPU do Aurora;
- `m9`: condição “abaixo de 30% → exportar; senão reduzir oversampling”;
- `m18`: pendência “testar a ponte em 96 kHz”;
- `m10`: resultado 27%.

A regra de 30% interpreta o resultado atual: 27% abre o caminho de exportação.
As configurações de exportação **não entram** no principal ainda.

### Quadro antecipatório

```json
{
  "predictions": [
    {
      "prediction_id": "pred_export",
      "likely_intent": "pedir instruções de exportação",
      "confidence": "alta",
      "clues": [
        {"type": "condition", "id": "m9"},
        {"type": "result", "id": "m10"}
      ],
      "source_ids": ["m24", "m25"],
      "reevaluate_when": "próximo pedido sobre exportação",
      "status": "hypothesis"
    },
    {
      "prediction_id": "pred_oversampling",
      "likely_intent": "comparar oversampling caso o resultado fosse outro",
      "confidence": "baixa",
      "clues": [{"type": "condition", "id": "m9"}],
      "source_ids": ["m26"],
      "reevaluate_when": "novo resultado de CPU",
      "status": "hypothesis"
    }
  ],
  "reserved_ids": ["m24", "m25", "m26"]
}
```

`m24`/`m25` são configurações de exportação; `m26` é material de oversampling.
Ficam em reserva até um pedido que as torne úteis.

### Próximo turno — promoção

Usuário: “Como exporto em 48 kHz?”

- `current_task` = instruções de exportação;
- `pred_export` é útil, atual, permitida e cabe no orçamento;
- `m24`/`m25` são promovidas ao principal;
- `pred_oversampling` permanece em reserva ou expira;
- somente o principal é enviado ao DeepSeek.

### Mudança de assunto

Usuário: “Por que escolhi o Pigments?”

- O pedido atual vence o foco de exportação;
- `pred_export` deixa de ser útil e é marcada `expired`;
- `pred_oversampling` também expira se não houver novo resultado;
- o principal passa a conter `m5`/motivo, não as configurações de exportação;
- nenhuma decisão ou pendência muda.

### Troca de projeto

- cada projeto tem seus próprios quadros;
- ao trocar, o principal do projeto anterior não é enviado ao novo;
- previsões e fontes do projeto anterior ficam isoladas e não são usadas;
- respostas atrasadas do projeto anterior são descartadas;
- o novo projeto carrega seu próprio principal e antecipatório.

## 7. Metacognição e contrato

Diagnóstico curto:

```json
{
  "metacognition": [
    "resultado 27% interpretado pela condição m9",
    "configurações de exportação permanecem em reserva",
    "próximo pedido pode ser exportação, mas é hipótese"
  ]
}
```

Contrato resumido:

```json
{
  "main_board": {"ids": [], "tokens": 0},
  "anticipatory_board": {"predictions": [], "reserved_ids": []},
  "promoted": [],
  "not_promoted": [{"id": "m24", "reason": "fora do pedido atual"}],
  "stop_reason": "main_ready",
  "search_budget": {},
  "whiteboard_budget": {}
}
```

## 8. Teste A/B de antecipação

Duas condições, mesma memória principal, mesmo orçamento e mesmo modelo:

- **antecipação ligada**;
- **antecipação desligada**.

Medir por turno:

- memórias antecipadas efetivamente promovidas/usadas no turno seguinte;
- tempo de busca economizado no turno seguinte;
- trabalho adicional: rodadas, candidatos, ms e tokens de busca;
- distrações no turno atual: IDs antecipados que entraram no principal sem
  utilidade para o pedido atual;
- mudanças indevidas de decisão, pendência ou foco (esperado: zero);
- taxa de previsões expiradas sem uso.

Critério de sucesso: redução de tempo no turno seguinte sem aumento de
distrações nem alteração de estado. A antecipação é opcional e nunca substitui
o quadro principal.

## 9. Testes planejados

- previsão não altera decisão, pendência, checklist ou foco;
- pedido explícito vence a previsão;
- promoção exige utilidade, validade, sensibilidade e orçamento;
- configurações de exportação não entram no principal com 27% de CPU;
- promoção ocorre no pedido de exportação seguinte;
- mudança de assunto expira previsões;
- troca de projeto isola quadros e descarta respostas atrasadas;
- antecipação não bloqueia a resposta atual;
- quadro antecipatório nunca é enviado ao DeepSeek;
- A/B mede uso, tempo economizado, trabalho extra e distrações;
- fontes restritas não entram em nenhum quadro.
