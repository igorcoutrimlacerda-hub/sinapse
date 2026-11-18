# Sinapse

Memória inspecionável para chatbots. Cada mensagem original é **um único
ponto**; pontos próximos de uma semente formam uma lembrança. O turno atual
define o foco, a atenção temática prioriza o que importa e o orçamento decide
o que realmente chega ao chatbot.

Somente Python 3 e biblioteca padrão. Sem dependências externas.

## Produto planejado

O primeiro produto planejado depois da fase de desenvolvimento da Sinapse é um
**copiloto de projetos**. O primeiro fluxo será **retomar um projeto depois de
um intervalo**: o copiloto deve recuperar a **decisão vigente**, seu **motivo**,
o **último teste concluído** e o **próximo passo pendente**, com referências às
mensagens originais que sustentam cada item.

O cenário inicial do copiloto será o **desenvolvimento da própria Sinapse**:
decisões de arquitetura, testes executados e pendências do projeto servem de
memória real para o fluxo. Esta etapa permanece no **planejamento do produto**;
o trabalho atual continua sendo desenvolver e validar a camada de memória. O
benchmark de retomada abaixo usa conversas fictícias e cliente simulado, não o
copiloto de produção nem o DeepSeek real.

## MVP do copiloto (CLI)

Inventário curto do que já existe:

- núcleo Sinapse: evocação, quadro, temas, sensibilidade, utilidade e orçamento;
- cliente DeepSeek e `SinapseChatbot`;
- benchmarks de retomada, rubrica v2, revisão cega e comparação real;
- CLI única em `sinapse/cli.py`;
- novo MVP persistente em `sinapse/project_copilot.py`, com projetos em JSON.

Fluxo utilizável:

```powershell
python -m sinapse project create "Sinapse"
python -m sinapse project say "Comecei a Sinapse com o preset Lume no bass."
python -m sinapse project say "Troquei para o preset Safira porque a CPU do Lume ficava alta."
python -m sinapse project say "O teste do Safira no refrão ficou concluído."
python -m sinapse project say "Falta testar a ponte em 48 kHz."
python -m sinapse project resume
python -m sinapse project show m2
python -m sinapse project list
python -m sinapse project select sinapse
```

O comando `resume` mostra decisão vigente, motivo, último teste concluído e
próximo passo, com IDs das mensagens originais. `show` abre a mensagem citada.
Correções substituem a decisão anterior; conclusões encerram a pendência
correspondente. Cada projeto tem arquivo próprio e a seleção ativa é
persistida; o processo pode ser reiniciado.

`project restrict "licença"` declara uma restrição e o `resume` omite as fontes
restritas. O chat usa resposta local determinística por padrão; `--deepseek`
usa o `SinapseChatbot` com a chave configurada. `--root` troca o diretório de
projetos (padrão `data/projects`).

Ainda faltam para o piloto: interface web, autenticação/multiusuário,
concorrência, extração semântica mais ampla que as regras atuais, tratamento
de anexos e uma validação real do chat `--deepseek` com o fluxo completo.

O desenho detalhado do prompt do copiloto está em
`docs/project_copilot_prompt.md`. O desenho da memória de diagnóstico está em
`docs/farol_diagnostic_memory.md`; o da inferência de intenção e do quadro
antecipatório, em `docs/farol_intent_inference.md`.

## Estágio atual (2026-11-20)

Fluxo implementado:

1. cada mensagem é um ponto anotado;
2. evocação por semente, foco, distância e atenção temática;
3. curiosidade opcional: parar, seguir vínculos, explorar outro tema ou
   procurar correção, com uma rodada extra limitada;
4. seleção por propósito sob orçamento de tokens;
5. redução de contexto que reavalia o ganho marginal de cada item;
6. sensibilidade aplicada antes da atenção e do orçamento;
7. requisição final ao `deepseek-flash`.

### Resultado real da seleção por propósito

Segundo a rubrica automática congelada, nos 12 casos pareados:

| condição | rubrica aprovada | todos os propósitos | partes corretas | prompt | completion |
|---|---:|---:|---:|---:|---:|
| referência congelada | 7/12 | 7/12 | **9/16** | 1470 | 574 |
| seleção por propósito | **12/12** | **11/12** | **15/16** | 1738 | 647 |

Pareado: **5 melhoraram, 7 empataram, 0 pioraram**.
`new_priority` é aprovado pela rubrica porque a prioridade explícita foi
cumprida, embora uma parte secundária (preço) tenha ficado pendente e
reconhecida.

Localmente, a seleção por propósito cobriu **8/8 propósitos** e **7/7 turnos
completos**, contra **5/8** e **5/7** da referência, usando 428 tokens de
memória contra 364. Esse resultado local mede seleção, não resposta.

### Remoção de evidência

Na rodada dirigida:

- teto previsto: 20 chamadas; executadas: **12**;
- confirmadas: remoção de `p2`, `m6`, `b3` e da evidência prioritária `m6`;
- falhou: remover `p1` não eliminou a atribuição do motivo da bicicleta
  naquela execução, pois o turno já nomeava a entidade e a decisão;
- sem efeito na requisição: `m7`, itens de histórico e itens restritos;
- nenhuma memória redundante realmente presente foi removida nessa rodada;
- `p1` na bicicleta economizou **36 tokens de prompt e 11 de completion**.

### Omissão condicional de dependências

Seis casos novos, um deles a regressão da bicicleta:

- chamadas: **10**;
- omissões conservadoras preservaram a resposta: **2**;
- omissões conservadoras prejudicaram: **0**;
- sem efeito na requisição: **4**;
- bicicleta: `p1,p2` → `p2`, economia de **36/11 tokens**, motivo,
  entidade e decisão preservados;
- dois propósitos: `p1,p3,p2` → `p3,p2`, economia de **36/12 tokens**,
  motivo e preço preservados;
- duas entidades e duas decisões mantiveram a dependência por segurança;
- `dep_redundant` não tinha o item no contexto; economia 0.

### Redução de contexto em pares reais

Nos 12 pares reais com `deepseek-flash`, o seletor reduzido empatou em
**12/12** e reduziu o prompt de **2114 para 1673 tokens**. Não houve melhora
ou piora automática de resposta nessa amostra.

### Retomada de projeto (validação simulada)

Cenário de **dez conversas** (2 a 4 sessões cada): as seis anteriores — faixa
Norte, Serra, Litoral e Vale — mais Costa (pendência substituída), Dunas
(próximo passo condicionado a um resultado desconhecido), Norte e Sul
(dois projetos intercalados com tarefas parecidas). Os gabaritos e a rubrica
foram fixados antes da execução. As condições são histórico recente, política
atual da Sinapse e a retomada multi-propósito **congelada**, no mesmo orçamento
de 160 tokens e com o mesmo renderizador, usando o **cliente simulado**.

| condição | decisão | motivo | concluído | pendência | cobertura | refs | suporte | condição | erros | nota | prompt | completion | memória |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| histórico recente | 0/9 | 0/10 | 0/8 | 0/9 | 0,00 | 0 | 0 | 0/1 | 37 | 0,00 | 1170 | 50 | 0,0 |
| Sinapse | **9/9** | **10/10** | 6/8 | 8/9 | 0,93 | 6 | 7 | 1/1 | 3 | 0,93 | 2360 | 465 | 119,0 |
| Sinapse multi-propósito | **9/9** | **10/10** | **8/8** | **9/9** | **1,00** | **9** | **10** | **1/1** | **0** | **1,00** | 2389 | 469 | 121,9 |

- seis regressões preservadas: **18/20 → 20/20 elementos**;
- famílias novas: **15/16 → 16/16**;
- total: **33/36 → 36/36**;
- **3 casos melhoraram, 0 pioraram**; nenhuma tarefa antiga voltou como atual e
  nenhuma tarefa foi atribuída ao projeto errado;
- seleção mediana aproximada: **2,9 → 7,3 ms**; prompt **+29 tokens**,
  completion **+4**, memória **+2,9 tokens**;
- nenhum passo concluído foi sugerido para repetição e nenhuma afirmação sem
  suporte apareceu.

Rastreamento da pendência em `retomada_geral`:

- foco inferido: `onde`; semente: `pr5` (teste concluído);
- `pr6` e `pr7` (pendência) **viraram candidatas**, mas ficaram fora do raio
  (`0,943` e `0,907`) e com atenção `0,0`;
- primeira perda: pontuação/raio, porque a retomada aberta não foi interpretada
  como pedido de pendência; o propósito não foi inferido;
- em `proximo_teste`, o foco é `por_que` e a semente é `pr7`; a própria
  pendência entra como semente.

A retomada multi-propósito roda o seletor congelado para decisão, teste
concluído e pendência e recombina sementes/evidências sob o mesmo orçamento,
com as mesmas regras de sensibilidade. Nesta rodada a implementação **não foi
alterada**: apenas novos casos e nova rubrica foram adicionados.

Casos novos:

- **Costa**: `cs4` (arpejo) não voltou como pendência; `cs5`/`cs6` (ponte)
  sustentam o próximo passo;
- **Dunas**: `dn4` trouxe o teste pendente e a condição “CPU abaixo de 30% →
  exportar, senão reduzir o oversampling”; condição **1/1**, sem omissão;
- **Norte/Sul**: as tarefas semelhantes não se misturaram; nenhum ID do outro
  projeto foi selecionado nem citado.

As referências foram checadas contra o elemento a que estão ligadas: decisão,
motivo, concluído, pendência e condição. A versão multi-propósito teve suporte
**10/10** nas linhas avaliadas, contra **7/10** da versão anterior.

Os números acima são do **cliente simulado** e validam seleção, rastreamento e
rubrica. A execução real foi feita separadamente e está registrada abaixo.

### Validação local dos pares

Os 10 pares preparados foram validados localmente:

- **10/10 pares válidos** e **10/10 seleções idênticas às versões congeladas**;
- integridade das referências: **10/10** no foco único e **10/10** na
  multi-propósito; cada referência selecionada foi enviada e sustenta o
  elemento ao qual está ligada;
- completude: **7/10** no foco único e **10/10** na multi-propósito;
- **Dunas**: condição **1/1** e dois caminhos **1/1** nas duas versões —
  caminho de exportação e caminho de redução do oversampling;
- **Norte/Sul**: **4/4** respostas/referências no projeto solicitado, sem
  confusão entre projetos;
- nenhuma condição omitida, tarefa antiga como atual ou referência inválida na
  multi-propósito.

Comando reproduzível:

```powershell
python -m sinapse resumption-validate
```

O resultado fica em
`benchmark_out_resumption_round3/local_validation.json`. Os arquivos de
revisão existentes foram preservados.

### Execução real com `deepseek-flash`

A chave foi configurada e o pareamento foi executado com os mesmos dez pares:

- **20 chamadas efetivas** (10 pares × 2 condições), **0 erros**;
- tokens reais: **4824 de prompt** (2384 + 2440) e **2162 de completion**
  (1101 + 1061).

Rubrica automática congelada:

| condição | decisão | motivo | concluído | pendência | completos | refs sem falha | condições | projeto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| foco único | 0/9 | 10/10 | 6/8 | 8/9 | 1/10 | 10/10 | 1/1 | 2/2 |
| multi-propósito | 0/9 | 10/10 | **8/8** | **9/9** | 1/10 | 10/10 | 1/1 | 2/2 |

- **Dunas**: condição e dois caminhos preservados nas duas condições reais;
- **Norte/Sul**: **4/4** respostas/referências no projeto correto;
- nenhuma referência inválida: todas as referências existentes sustentam o
  elemento correspondente;
- a coluna **decisão 0/9** é um falso negativo da rubrica congelada: o gate
  exige frases como “decisão vigente”, “uso o” ou “troquei para”, enquanto as
  respostas reais usam “está valendo”, “é o” e “trocou para”. Um diagnóstico
  separado — que **não substitui a rubrica** — encontrou o fato da decisão em
  **9/9** respostas. Com esse diagnóstico, os casos completos seriam 7/10 no
  foco único e 10/10 na multi-propósito;
- a rubrica permanece congelada e não foi alterada para favorecer o resultado.

Arquivos reais:

- `benchmark_out_resumption_real/real_results.json`;
- `benchmark_out_resumption_real/real_review_blind.json` — 10 pares A/B sem
  rótulo de condição;
- `benchmark_out_resumption_real/real_summary.txt`.

As **54 respostas simuladas** (6 + 18 + 30) continuam preservadas. Os números
acima vêm da **rubrica automática**; a revisão cega real foi concluída e está
registrada separadamente abaixo.

### Revisão humana cega dos pares reais

O arquivo `benchmark_out_resumption_real/real_review_blind.json` contém os
**10 pares A/B** sem rótulo de condição. Cada resposta tem campos humanos em
branco para:

- **decisão vigente**: nota, `current_identified`, `reference_supports` e
  `old_or_ambiguous`;
- **motivo**, **último teste concluído** e **pendência**;
- **condição**, quando aplicável (Dunas).

As fontes citadas nas respostas são incluídas ao lado de cada par para o
revisor conferir se a referência sustenta a leitura. Paráfrases são aceitas; o
guia diferencia decisão vigente de alternativa antiga ou menção ambígua.

Preparar o formulário (sem novas chamadas):

```powershell
python -m sinapse resumption-review-prepare
```

Depois de preencher as notas, consolidar:

```powershell
python -m sinapse resumption-review-consolidate
```

Enquanto houver campo em branco, o comando recusa a consolidação e **não
revela A/B**. Depois de todas as notas, ele grava
`benchmark_out_resumption_real/human_consolidated.json`, revela as condições e
compara o humano com a rubrica automática e com o diagnóstico de decisão. A
rubrica automática e o diagnóstico permanecem em `real_results.json`, como
registros distintos.

#### Resultado da revisão cega

Os 10 pares foram revisados e consolidados. Esta é uma **revisão cega
assistida por IA**, com as respostas A/B sem rótulo de condição até a
consolidação.

| condição | decisão | motivo | concluído | pendência | condição | completos | automáticos |
|---|---:|---:|---:|---:|---:|---:|---:|
| foco único | 9/9 | 10/10 | 6/8 | 8/9 | 1/1 | 7/10 | 1/10 |
| multi-propósito | 9/9 | 10/10 | **8/8** | 8/9 | 1/1 | **9/10** | 1/10 |

- preferências: **multi-propósito 6**, foco único **3**, **1 empate**;
- completos pelo diagnóstico de decisão: **7/10** e **10/10**;
- **21 divergências** em relação à rubrica automática:
  - 18 de decisão — o humano aceitou paráfrases que a rubrica congelada
    rejeitou, 9 em cada condição;
  - 3 de pendência — `RP01` no foco único (0 humano × 1 automático), `RP02` na
    multi-propósito (0 × 1) e `RP06` no foco único (1 × 0);
- nenhuma divergência de condição; referências automáticas sem falha e nenhum
  projeto trocado;
- a multi-propósito perdeu um caso completo por causa de `RP02`: a resposta
  dizia “introduzir a faixa”, que o revisor não aceitou como o teste da
  introdução.

O arquivo consolidado revela o mapa A/B e mantém as notas humanas, a rubrica
automática e o diagnóstico em seções separadas.

#### Referência experimental do copiloto

A retomada **multi-propósito** fica registrada como referência experimental do
copiloto de projetos. Na revisão cega assistida por IA:

- **9/10 casos completos** contra **7/10** do foco único;
- **8/8 testes concluídos** contra **6/8**;
- preferência pareada: **6 a 3**, com **1 empate**.

A rubrica congelada, o diagnóstico e a consolidação existente foram
preservados.

**RP02 — diferença de formulação**

`pr7` estava na requisição das duas respostas. O contraste é:

| versão | formulação | ação | objeto | parâmetro | objetivo de CPU |
|---|---|---|---|---|---|
| multi-propósito | “introduzir a faixa em 96 kHz” | introduzir | faixa | 96 kHz | preservado |
| foco único | “introdução em 96 kHz” | testar (implícito em “próximo teste”) | introdução | 96 kHz | preservado |

“Testar a introdução” mantém a ação de teste e o objeto correto; “introduzir a
faixa” troca a ação para importar/converter e o objeto para a faixa inteira. O
parâmetro e o objetivo de medir a CPU permanecem. O revisor deu pendência **0**
para a multi-propósito e **1** para o foco único; o foco único também
acrescentou clipping/dropouts sem suporte em `pr7`. O detalhe está em
`benchmark_out_resumption_real/rp02_analysis.json`.

#### Rubrica v2

A v1 permanece congelada. A nova versão foi calibrada em **20 linhas** (10
pares × 2 condições) e em **6 casos sintéticos inéditos**. Ela exige:

- decisão com fato atual + paráfrase de atualidade, rejeitando decisão antiga
  como vigente;
- motivo, concluído e pendência com ação, objeto e parâmetros;
- condição com os dois caminhos.

| critério | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| decisão | 18 | 0 | **0** | **0** |
| motivo | 20 | 0 | **0** | **0** |
| concluído | 14 | 2 | **0** | **0** |
| pendência | 16 | 2 | **0** | **0** |
| condição | 2 | 0 | **0** | **0** |

Casos sintéticos: **6/6** corretos. Eles cobrem paráfrase correta, decisão
antiga como vigente, próximo passo reformulado, próximo passo fiel, condição
incompleta e condição com dois caminhos. A calibração zera FP/FN nos dados
usados para ajustar a rubrica; não é uma validação independente.

Comando:

```powershell
python -m sinapse resumption-rubric-v2
```

Arquivo: `benchmark_out_resumption_real/rubric_v2_calibration.json`.
Nenhuma chamada ao DeepSeek.

#### Comparação de instrução de geração

A comparação foi executada com a memória multi-propósito, o histórico e o
orçamento de **160 tokens** fixos; apenas a instrução de geração variou:

- **atual**: instrução existente;
- **experimental**: preservar ação, objeto do teste, parâmetros e condições das
  fontes, manter os dois caminhos e não apresentar decisão antiga como vigente.

Foram **20 chamadas efetivas** (10 casos × 2 instruções), **0 erros**, com
**5686 tokens de prompt** e **2133 de completion**.

| instrução | decisão | motivo | concluído | pendência | condição | completos |
|---|---:|---:|---:|---:|---:|---:|
| atual | 9/9 | 10/10 | 8/8 | 9/9 | 1/1 | **10/10** |
| experimental | 7/9 | 10/10 | 8/8 | 9/9 | 1/1 | 8/10 |

- **0 melhoraram, 8 empataram, 2 pioraram**;
- pioraram `proximo_condicionado` (Dunas) e `intercalado_norte`, ambos por
  decisão na v2;
- erros novos na experimental: 2, os mesmos dois casos; erros persistentes: 0;
- **RP02**: as duas instruções preservaram o teste da introdução em 96 kHz e o
  objetivo de verificar a CPU; nenhum acréscimo sem suporte apareceu nesta
  execução;
- **Dunas**: limite de 30% e os dois caminhos preservados nas duas instruções;
- custo da instrução: **+40 tokens estimados**, **+580 tokens reais de prompt**
  no total (cerca de +58 por chamada); o bloco de memória ficou em **130,1
  tokens estimados** nas duas condições;
- os dez cenários são conhecidos; não há casos inéditos nesta comparação.

A revisão cega dos pares A/B está em
`benchmark_out_instruction_experiment/instruction_review_blind.json`, sem
rótulo de instrução. Os resultados automáticos, com as instruções reveladas,
estão em `instruction_results.json`. A v1, o diagnóstico e a revisão cega
anterior permanecem preservados.

### Aprendizado local

- política genética: referência, busca aleatória e genética empataram em
  **12/12** casos conhecidos e **5/5** famílias inéditas; a evolução não
  superou a busca aleatória nessa amostra;
- seleção por propósito local: 8/8 propósitos e 7/7 turnos completos;
- curiosidade aprendida local: a variante com vínculo recuperou o caso da
  bicicleta, mas a política permaneceu empatada com a referência em cobertura;
- escala sintética local (1k/10k/100k): em 100k, a Sinapse cobriu 5/8
  consultas; com uma melhoria na busca de candidatos, passou a 7/8. Isso é
  avaliação local de recuperação, não de resposta real.

### Escala sintética de 100 mil mensagens

Execução reproduzida com os parâmetros originais: `scale=100000`, `seed=42`,
`candidate_k=200`, memória `160` e `output_reserve=4096`, sem chamadas ao
DeepSeek. As 8 consultas com evidência esperada foram cobertas em **5/8**;
as 3 consultas sem evidência ficam fora do agregado.

Rastreamento das perdas:

| consulta | evidência | indexada | fonte | rank antes do corte | depois do corte | quadro | primeira perda |
|---|---|---|---:|---:|---|---|---|
| `correcao` | `t5` | sim | não encontrado | 7215 | fora | não | candidatos |
| `referencia_curta` | `t3` | sim | não encontrado | 3705 | fora | não | candidatos |
| `ambiguidade` | `t5` | sim | não encontrado | 7215 | fora | não | candidatos |

Nos três casos a evidência está indexada, mas nenhuma fonte de candidatos a
alcançou: ela não aparece na lista lexical nem entre os vínculos dos itens
selecionados. A expansão por vínculo já existe em `build_shortlist`, porém só
adiciona pai/resposta de candidatos que já entraram no corte. Em
`referencia_curta`, o pai `t1` também ficou fora da lista, então a expansão não
tinha de onde partir. Em `correcao` e `ambiguidade`, a evidência não tem
`reply_to` para seguir.

**Melhoria única escolhida:** somar os tokens do histórico recente à consulta
de candidatos, mantendo congelados o seletor por propósito, o limite de
`candidate_k=200`, o renderizador e o orçamento. Nenhum peso foi reajustado.

Regressões (8 consultas com evidência):

| consulta | candidatos | quadro | final | pontos | busca (ms) | memória |
|---|---|---|---:|---:|---:|---:|
| `motivo` | 1→1 | 1→1 | 1→1 | 346→402 | 15,5→27,3 | 111→111 |
| `procedimento` | 1→1 | 1→1 | 1→1 | 201→327 | 10,5→18,1 | 80→80 |
| `duas_evidencias` | 1→1 | 1→1 | 1→1 | 346→402 | 16,0→28,2 | 111→111 |
| `correcao` | 0→0 | 0→0 | 0→0 | 200→200 | 7,7→18,6 | 86→86 |
| `preco_presente` | 1→1 | 1→1 | 1→1 | 200→200 | 6,7→7,3 | 47→47 |
| `referencia_curta` | **0→1** | **0→1** | **0→1** | 200→327 | 6,5→24,3 | 47→80 |
| `ambiguidade` | **0→1** | **0→1** | **0→1** | 200→200 | 7,7→7,2 | 86→47 |
| `resolvida` | 1→1 | 1→1 | 1→1 | 346→400 | 15,3→35,1 | 0→0 |

Agregado: candidatos, quadro e final **5/8 → 7/8**; **2 melhoraram, 0
pioraram**. Custo extra: pontos médios **254,9 → 307,2** (+52,3) e busca
mediana **10,1 → 20,3 ms** por consulta; memória **71,0 → 70,2 tokens**. A
soma da busca nas 8 consultas foi **86,1 → 169,2 ms**. A shortlist pode passar
de 200 itens quando a expansão por vínculo adiciona pais/respostas; os pontos
acima são os medidos, não o limite nominal.

Restou `correcao`: `t5` continua no rank 7215, sem vínculo e fora do corte.
A próxima hipótese é uma busca específica de correção/diversidade, não um novo
ajuste de pesos.

Famílias novas com distrações do mesmo tema: **3/3 cobertas antes e depois**.
`novo_quantidade_aurora` e `novo_quantidade_pack` empataram; `novo_preco_curto`
já era coberta, mas passou de **1 para 200 candidatos examinados**, com busca
de 0,04 → 6,9 ms. No agregado novo, pontos **133,7 → 200,0**, busca mediana
**7,2 → 6,7 ms** (soma 14,6 → 19,9 ms) e memória **58,0 → 58,0**.

### Revisão humana

Permanecem **23 pares cegos pendentes** de revisão humana:

- 12 pares da avaliação por propósito;
- 5 pares da rodada de remoções;
- 6 pares da rodada de dependências condicionais.

Os arquivos de revisão preservam respostas A/B/C sem rótulo de condição. Não
há revisão humana registrada; todos os números acima são automáticos.

### Generalização

A generalização para famílias de conversa realmente inéditas e para arquivos
muito maiores que 100 mil mensagens ainda precisa de avaliação com o DeepSeek
real. Os resultados de 100k são sintéticos e locais; os pares reais são poucos
e não formam uma estimativa estatística.

## Comandos

```powershell
python -m unittest discover -s tests -v

# Núcleo textual
python -m sinapse demo
python -m sinapse ask "Por que eu escolhi o Serum 2?" --context "Estávamos falando do Serum 2 e do bass."
python -m sinapse groups
python -m sinapse group g:o_que:1
python -m sinapse replies m1

# Agente com quadro branco e atenção temática
python -m sinapse agent-demo

# DeepSeek (real; lê DEEPSEEK_API_KEY)
python -m sinapse chat "Por que eu escolhi o Serum 2?" --context "Estávamos falando do Serum 2 e do bass."
python -m sinapse chat "Por que eu escolhi o Serum 2?" --context "Estávamos falando do Serum 2 e do bass." --simulate

# Benchmarks e experimentos
python -m sinapse benchmark
python -m sinapse scale-benchmark
python -m sinapse scale-recovery
python -m sinapse resumption-benchmark
python -m sinapse resumption-benchmark --deepseek
python -m sinapse resumption-validate
python -m sinapse resumption-real-pairs
python -m sinapse resumption-review-prepare
python -m sinapse resumption-review-consolidate
python -m sinapse resumption-rubric-v2
python -m sinapse instruction-experiment
python -m sinapse instruction-review-consolidate
python -m sinapse project create "Sinapse"
python -m sinapse project say "Comecei a Sinapse com o preset Lume no bass."
python -m sinapse project resume
python -m sinapse project show m2
python -m sinapse curiosity-learn
python -m sinapse curiosity-round2
python -m sinapse curiosity-round3
python -m sinapse paired-answer --deepseek
python -m sinapse purpose-benchmark
python -m sinapse purpose-answer --deepseek
python -m sinapse purpose-report
python -m sinapse removal-round --deepseek
python -m sinapse dependency-round --deepseek
python -m sinapse genetic-policy
python -m sinapse policy-answer --deepseek
```

Os comandos `--deepseek` fazem chamadas reais e exigem
`DEEPSEEK_API_KEY`; sem a flag, usam cliente simulado. Os comandos de
relatório (`purpose-report`) leem resultados salvos e não fazem chamadas.

`ask` aceita `--window`, `--output-reserve`, `--radius`, `--top-k`,
`--threshold`, `--focus`, `--seed`, `--base-message-id`, `--no-links`,
`--no-seed` e `--json`.

`chat` aceita os mesmos controles de evocação e orçamento, além de `--model`,
`--base-url`, `--max-tokens`, `--thinking`, `--simulate` e `--json`.

## Integração com OpenCode

A memória pode acompanhar o agente de código no próprio OpenCode, por um plugin
local que reutiliza o `ProjectStore` e o bridge `sinapse bridge`. O plugin fica
em `.opencode/plugins/sinapse/index.ts` e carrega automaticamente neste
repositório. Em cada sessão nova ele:

1. injeta no primeiro pedido o objetivo, a tarefa em andamento, a decisão
   vigente, as correções, o último resultado verificado e o próximo passo, com
   os IDs de fonte;
2. expõe `sinapse_resume`, `sinapse_search`, `sinapse_record` e
   `sinapse_status` como ferramentas locais.

O projeto é resolvido pela raiz do repositório (`repo_path` salvo no JSON do
projeto), por `.sinapse.json` (`{"project": "<id>"}`) ou por `SINAPSE_PROJECT`;
o projeto ativo global do copiloto não é usado. O bridge também funciona como
fallback manual:

```powershell
python -m sinapse bridge --repo . status --json
python -m sinapse bridge --repo . resume --json
python -m sinapse bridge --repo . search "configuração DeepSeek" --limit 5 --json
python -m sinapse bridge --repo . record --kind decision --text "..." --entry-id dec-exemplo --json
python -m sinapse bridge --repo . record --kind correction --text "..." --supersedes dec-exemplo --json
python -m sinapse bridge projects --json
```

Registros usam `entry_id` estável e repetir a gravação não duplica. Correções
exigem `supersedes` e mantêm o vínculo com a decisão anterior e com a fonte.
Textos com cara de credencial são recusados antes de serem gravados. Para usar
o plugin em outros repositórios, registre o diretório em `plugins` no
`opencode.json` global; o passo a passo está no `AGENTS.md`.

## Farol para projetos de software

O Farol usa a mesma memória por projeto, agora com tipos técnicos:
`architecture`, `decision`, `correction`, `bug`, `hypothesis`, `change`,
`test`, `blocker`, `pending`, `summary`, `goal` e `task`. Cada registro pode
carregar `files`, `issues`, `prs`, `commits`, `command`, `symptom`,
`restrictions` e a origem do resultado (`reported_by`, `report_source`).

- Uma pergunta sobre bug prioriza sintoma, tentativas, arquivos e testes.
- Uma pergunta de arquitetura usa a decisão vigente, motivo e restrições; a
  versão substituída não entra no estado atual.
- Uma saudação ou ideia casual não carrega o bloco técnico.
- “Onde paramos?” mostra tarefa atual, decisão técnica, último teste observado
  e próximo passo, com IDs de fonte.
- O painel permite associar o projeto a um repositório (`repo_path`), o mesmo
  campo usado pela ponte do OpenCode.
- “Briefing OpenCode” gera bullets com objetivo, contexto, restrições e
  critérios de aceitação. Um teste sugerido fica pendente; um teste reportado
  pelo OpenCode guarda relatório e comando.

O payload final é redigido antes de ir ao provedor: mensagem atual, histórico,
estado e memória passam pelo mesmo filtro de credenciais. Se o contexto
obrigatório não couber na janela, o Farol reduz primeiro o contexto opcional e,
se ainda não couber, retorna erro explícito sem chamar o modelo.

A busca semântica local fica em `sinapse/semantic.py` e é opcional. O padrão
global continua desligado em `data/config.json` (`"semantic": {"enabled": false}`);
cada projeto pode ser ligado no painel (“Busca semântica neste projeto”) ou
pelo campo `semantic` do JSON. Ela usa representações por hashing de palavras,
radicais, conceitos de software e trigramas, não um modelo treinado; serve para
lacunas com vocabulário diferente, enquanto nomes exatos de arquivos, funções,
commits e códigos de erro continuam na busca lexical. O seletor valida projeto,
vigência, sensibilidade e orçamento antes de enviar.

O painel também importa um repositório Git local para um índice por projeto
(`<id>.code.json`, fora do versionamento). A importação lê apenas arquivos
rastreados no commit, ignora binários, gerados e credenciais, e registra
caminho, intervalo de linhas, símbolo, commit e licença identificada. O índice
guarda código, comentários/docstrings, testes e documentação. A busca exata
prioriza símbolos e caminhos; a busca por comportamento usa o índice semântico
local. Comentários são fontes próprias: quando um comentário contradiz a
implementação (por exemplo, diz “maiúsculas” e o código usa minúsculas), o
Farol aponta a divergência. Ao trocar de commit, os arquivos removidos deixam a
busca vigente e as referências antigas permanecem ligadas ao commit original.

## Dados

`data/demo_conversation.json` tem seis mensagens usadas pelo núcleo textual.

`data/agent_conversation.json` tem a conversa longa do agente:

- m1 escolhe o preset Serum 2 para o bass;
- m2, resposta curta ligada a m1, registra o motivo (licença já comprada);
- m3, ligada a m1, registra o procedimento (dois envelopes);
- m4 fala de xadrez;
- m5/m6 são Piano V e seu motivo;
- m7/m8/m9 tratam do pack de presets, da divulgação e do pack do bass;
- p1/p2 formam uma segunda conversa (`c2`) sobre uma bicicleta.

`sent_at` é a **data de envio**, metadado de identidade. A dimensão `quando`
representa a **data do acontecimento** e só existe quando anotada. Consulta
temporal não usa `sent_at` como evidência.

## Representação

Sete dimensões: `o_que`, `quem`, `onde`, `quando`, `como`, `por_que`,
`para_que`. Cada uma aceita vários valores. A anotação é manual, em JSON.
Ausência é desconhecida (`null` no vetor), não zero. Cada mensagem permanece
um ponto inteiro, com ID, texto, autor, conversa, posição, `sent_at` e
`reply_to` opcional.

Grupos: por dimensão, Jaccard dos tokens dos valores; união dos pares com
similaridade ≥ limiar (single-linkage). Limiar padrão `0.33`, configurável em
`group_threshold` ou `groups --threshold`. Grupos de dimensões diferentes se
sobrepõem; uma mensagem pode estar em vários.

## Semente e foco

- Foco: regra lexical, nesta ordem: `para que`, `por que/porque`, `o que`,
  `como`, `quem`, `onde`, `quando`; fallback `o_que`.
- Referências: pronomes (`ele`, `isso`, `aquilo`, …) registram o turno anterior
  mais próximo como antecedente.
- Semente:
  `score(m) = |tokens(m) ∩ âncora| / sqrt(|tokens(m)| · |âncora|)`, onde
  `tokens(m)` inclui texto e coordenadas. Desempate: posição, depois ID.
- O agente multiplica esse score por um viés temático
  `1 + 0.5 · afinidade(m, temas)`; em turno sem dimensão explícita, soma até
  `0.8 · sobreposição(m, turno atual)` para desempatar referências curtas.
- Pesos de distância: `o_que .25, quem .10, onde .05, quando .05, como .10,
  por_que .10, para_que .15`; a dimensão pedida recebe `focus_weight = 1.0`.

## Distância

```
sim_d(p, s) = Jaccard(valores_p[d], valores_s[d])
distance(p) = 0                                            se p == s
distance(p) = 1 - Σ_d(w_d · sim_d) / Σ_d(w_d)              caso contrário
```

Ausência de um lado vale `0` (nunca correspondência). Exceção explícita: se
`p` e `s` têm vínculo direto de resposta, o lado sem valor herda o outro
**apenas para esse par**, pois o vínculo declara o mesmo acontecimento. Raio
padrão `0.70`.

## Associação e atenção

```
content(p) = |tokens(p) ∩ âncora| / sqrt(|tokens(p)| · |âncora|)
link(p)    = 0.90 se há vínculo direto com a semente, senão 0
association(p) = max(content(p), link(p))
attention(p)   = association(p) × (evidence[p][focus] ?? 0)
```

`0.90` é heurístico e configurável em `link_association_floor`. O diagnóstico
mostra `association_content`, `association_link`, `association` e a origem
(`content`/`link`/`none`).

Vetor de evidência, por dimensão:

- `null` se a mensagem não tem anotação própria na dimensão;
- `0..1` = Jaccard entre o conteúdo próprio e o valor do evento da semente;
- `0.0` significa informação presente que não ajuda; `null` significa ausência.

O vínculo só fornece contexto de assunto para distância/associação. Motivo,
procedimento e objetivo sempre precisam do conteúdo da própria mensagem.
Atenção usa `null` como zero **somente na multiplicação**; o vetor preserva
`null`.

## Atenção temática por turno

Antes da evocação, o agente produz um relatório estruturado:

- **temas ativos** e pesos normalizados (o maior vira `1.000`);
- **origem** de cada peso: `turno_atual`, `historico_recente` ou `decaimento`;
- **termos observáveis** que acionaram o tema;
- **dimensão pedida** (`por_que`, `como`, `o_que`, …) e a regra textual;
- **ambiguidade** e seu motivo.

Regras determinísticas:

```
menção explícita no turno        -> peso 1.0 (reativa o tema)
tema não mencionado              -> peso anterior × 0.55
tema com apoio no histórico/quadro -> piso 0.40 (sem ultrapassar o explícito)
peso abaixo de 0.20              -> descartado
ambiguidade                      -> 2º >= 0.85 × 1º e sem menção explícita
```

Consequências:

- **menção explícita manda**: nesse turno a semente usa apenas o turno atual
  como âncora; histórico e quadro não sequestram o assunto novo;
- **referência curta** (“E como eu testei?”) usa histórico + quadro quando não
  há menção explícita;
- **vários temas** podem ficar ativos ao mesmo tempo (ex.: `bass` e `pack`);
- **reativação**: um tema decaído volta a `1.0` com nova menção;
- **ambíguo**: o quadro aceita no máximo **1 evidência** e registra o motivo,
  em vez de fingir certeza.

A afinidade temática reordena as evidências:
`atenção_efetiva = atenção × (0.6 + 0.4 × afinidade)`. O fator nunca zera uma
evidência; apenas reduz o peso de quem não cobre os temas ativos. Uma resposta
curta sem termo temático próprio herda o tema do pai via `reply_to` (m2 herda
Serum/bass de m1).

## Quadro branco por conversa

`MemoryAgent` mantém um `ConversationWhiteboard` por `conversation_id`:

- **assunto**: a semente do turno;
- **evidências**: mensagens com evidência não nula no foco, dentro do raio e
  com atenção acima do limiar, ordenadas por atenção;
- **limite**: 5 evidências; 1 em turno ambíguo;
- **IDs e textos originais**: o quadro guarda apenas IDs e metadados; o texto
  vem do arquivo de memória e nunca é apagado ao sair do quadro;
- **reavaliação**: a cada turno todos os itens são reavaliados contra a nova
  semente; entradas, permanências e saídas viram diagnóstico com scores,
  motivos e resultado do orçamento;
- **retorno**: um item que saiu pode voltar se uma nova menção o tornar
  relevante (m2 volta ao retomar o preset).

Orçamento a cada turno, aplicado ao **payload de chat**:

```
capacidade_entrada = janela - reserva_de_saída
tokens_base        = contador(mensagens sem memória)
orçamento_memória  = max(0, capacidade_entrada - tokens_base)
```

A contagem inclui instruções (system), histórico recente, turno atual,
memórias renderizadas com IDs/autoria/posição/vínculo e a reserva de saída.
Distância, vetor, atenção, temas e motivos ficam **somente** no diagnóstico
local. Mensagens nunca são truncadas; candidato que não cabe é pulado.

Casos especiais: `base_overflow` (entrada-base maior que a capacidade),
`no_budget` (orçamento zero), `outside_radius`, `no_focus_evidence`,
`below_threshold`, `already_in_base`, `duplicate`, `top_k_full`,
`does_not_fit`, `requires_seed`, `selected`, `board_limit`.

## Utilidade marginal

Antes de entrar no orçamento, cada candidato recebe um ganho estimado:

```
base    = evidência[focus] × associação
novidade = 1 − cobertura máxima no histórico recente + evidências já escolhidas
ganho   = base × (0.4 + 0.6 × novidade)
```

- afirmação idêntica já presente → ganho × 0.1 (`same_assertion_in_context`);
- conteúdo já coberto pelo contexto → novidade baixa e candidato descartado;
- negação, número, data ou acontecimento diferente preserva novidade (piso
  `0.65` quando a cobertura é alta), porque a diferença pode mudar a resposta;
- o cálculo é refeito depois de cada inclusão;
- a seleção encerra com espaço sobrando quando os restantes têm
  `novidade < 0.55` ou ganho abaixo de `0.05`;
- `marginal_gain`, `selection_reason`, `stopped_early` e `stop_reason` ficam no
  diagnóstico local.

## Sensibilidade

Duas camadas separadas:

- `Preference` **declarada** pelo usuário: prioridade máxima, pode ser
  atualizada; escopo `assunto`, `mensagem` ou `detalhe`;
- inferência do **avaliador temático**: desconforto indireto
  (“isso me deixa desconfortável”, “estou evitando…”), bloqueia evocação
  espontânea, mas perde para uma declaração explícita.

Ações distintas:

- `restringir`: a mensagem fica no arquivo, mas não entra na requisição;
- `excluir`: remoção permanente do conjunto ativo; não volta nem com pedido
  claro;
- `permitir`: atualiza/remove uma restrição declarada.

Uma memória restrita só é liberada quando o turno atual tem **pedido claro**
(marcador como “lembra”, “pode contar”, “retomar”, “detalhe”) **e** menciona o
alvo específico. Referência ambígua (“E aquilo?”) ou associação temática não
liberam. A liberação vale só para aquele turno e só para o detalhe pedido.

A filtragem acontece antes da atenção e do orçamento, sobre:

- candidatos evocados e itens do quadro anterior;
- resumos derivados;
- mensagens antigas do histórico recente (o turno atual é sempre preservado);
- a requisição final ao DeepSeek.

Diagnóstico: apenas IDs, ações, escopos e motivos genéricos
(`detalhe restrito`, `mensagem restrita`, `exclusão solicitada`); o alvo e o
texto restrito não são reproduzidos.

## Curiosidade aprendida

`python -m sinapse curiosity-learn` executa uma rodada extra de curiosidade
com quatro ações: `stop`, `follow_links`, `other_theme` e `correction`. Para
cada caso de treino, todas as ações disponíveis partem do mesmo resultado
inicial e passam pelo mesmo seletor e orçamento. O rótulo é a ação com maior
ganho em evidências esperadas; empates preferem menos inclusões irrelevantes,
menos candidatos extras e, por fim, `stop`.

Sinais usados antes da busca extra: ambiguidade, evidência faltante, conflito,
vínculos, quantidade de candidatos, atributo disponível, atributo pedido e
restrição de sensibilidade. O modelo é um softmax linear; os pesos são
congelados antes da avaliação. Treino e avaliação são separados por família.

Resultado local (7 casos de treino, 6 de avaliação):

```
condição   evidências  buscas_sem_ganho  irrelevantes  extras  latência  tokens
desligada       0              0              5          0      0.25 ms  142.0
fixa            3              3              5          4      0.68 ms  170.5
aprendida       3              0              5          3      0.62 ms  160.8
```

A política aprendida melhorou `ev_correction` (parou de explorar o tema errado
e buscou a correção) e regrediu `ev_follow` (parou em vez de seguir vínculos);
eliminou buscas extras sem ganho e reduziu tokens, mas o saldo de evidências
ficou empatado com a regra fixa. A restrição de licença foi aplicada em todas
as ações.

## Benchmark

> Esta seção descreve o benchmark inicial de 15 casos e seus resultados
> históricos. A síntese do estágio atual está em **Estágio atual (2026-11-20)**,
> no topo do documento.

`benchmark/cases_eval.json` tem 15 casos; `benchmark/cases_tuning.json` tem 2
casos separados para calibração. Cada caso declara:

- arquivos de mensagens, histórico recente, pergunta;
- `expected_ids`, `dependency_ids`, `forbidden_ids`, `noise_ids`;
- limite de memória, fatos exigidos, afirmações proibidas, necessidade de
  reconhecer ausência e rubrica;
- preferências de sensibilidade e resumos derivados, quando aplicável.

Quatro condições, com o mesmo modelo, instruções, histórico, pergunta, reserva
de saída e limite de memória:

1. `history`: só histórico recente;
2. `lexical`: busca por similaridade lexical, sem quadro/tema/utilidade;
3. `sinapse`: evocação + quadro + temas + sensibilidade + utilidade;
4. `manual`: evidências corretas + dependências, escolhidas à mão.

Comandos:

```powershell
python -m sinapse benchmark
python -m sinapse benchmark --cases benchmark/cases_tuning.json --out benchmark_out_tuning
python -m sinapse benchmark --deepseek --limit 4 --out benchmark_out_deepseek
```

O modo padrão usa um cliente **simulado** que responde conforme a evidência
presente na requisição. Ele valida o mecanismo de avaliação; os números **não**
são atribuídos ao DeepSeek. `--deepseek` faz chamadas reais e é identificado
separadamente.

Métricas de seleção sobre o contexto efetivamente enviado: `hit` (alguma
evidência correta), `full` (toda a evidência necessária), tokens adicionais,
violações de conteúdo restrito, repetição de evidência já no histórico e
candidatos irrelevantes em casos sem evidência. Métricas de resposta: fatos
exigidos, afirmações proibidas, escopo sensível e reconhecimento de informação
ausente. O arquivo `review.json` exporta respostas e rubricas sem identificar a
condição; `results.json` mantém o mapeamento completo.

Resultado local medido (15 casos, cliente simulado):

```
condição         hit    full   tokens  violações  repet.  irrelev.  resposta
history          0.10   0.10     0.0         0       0         0     0.40
lexical          0.70   0.60   103.3         0       0         1     0.73
sinapse          1.00   1.00    70.9         0       0         1     1.00
manual           1.00   1.00    69.3         0       0         0     1.00

Delta Sinapse: +60.00 pp vs history, +26.67 pp vs lexical, +0.00 pp vs manual
```

Chamada real, subconjunto de 4 casos (16 requisições ao `deepseek-flash`),
separada do resultado simulado:

```
history       hit 0.00  full 0.00  resposta 0.00
lexical       hit 0.50  full 0.50  resposta 0.50
sinapse       hit 1.00  full 1.00  resposta 1.00
manual        hit 1.00  full 1.00  resposta 1.00
Delta Sinapse: +100.00 pp vs history, +50.00 pp vs lexical, +0.00 pp vs manual
```

Falha observada no modo real: sem memória, o modelo inventou um motivo
(“timbre de baixo encorpado”) que não está no histórico. Com a memória da
Sinapse, respondeu a licença correta (m2). O conjunto real é pequeno e não
substitui uma avaliação estatística.

## Integração DeepSeek

Valores verificados na documentação oficial em **2026-11-16**:

| item | valor |
| --- | --- |
| modelo | `deepseek-flash` = DeepSeek-V4.1-Flash-1017 |
| contexto | 1M tokens (orçado aqui como 1.000.000) |
| saída máxima | 384K = 393.216 tokens |
| endpoint | `POST https://api.deepseek.com/chat/completions` |
| chave | `DEEPSEEK_API_KEY` |

Padrões em `data/config.json`:

```json
"deepseek": {
  "model": "deepseek-flash",
  "base_url": "https://api.deepseek.com",
  "context_window": 1000000,
  "output_reserve": 4096,
  "max_tokens": 4096,
  "thinking": "disabled",
  "timeout": 60
}
```

A API liga o modo de raciocínio por padrão; aqui ele é desligado para respostas
de chat mais rápidas e baratas, e pode ser reativado com `thinking: enabled`.
`max_tokens` não pode exceder a reserva de saída nem o máximo oficial.

O corpo enviado é montado a partir das mensagens efetivamente orçadas:

```json
{
  "model": "deepseek-flash",
  "messages": [...],
  "max_tokens": 4096,
  "stream": false,
  "thinking": {"type": "disabled"},
  "temperature": 0.7
}
```

`SinapseChatbot` executa evocação → orçamento → `client.complete(body)`. O
cliente HTTP real usa `urllib` e aceita um `opener` injetável; o
`SimulatedDeepSeekClient` registra uma cópia serializada do corpo final e
devolve uma resposta fixa.

Em 2026-11-16, uma chamada real respondeu:

```
Você escolheu o Serum 2 porque já tinha comprado a licença (m2).
prompt_tokens=181 completion_tokens=21 total_tokens=202
estimativa local=184 | real informado=181 | diferença=-3
```

## Contador

A contagem padrão é **estimada** e assim identificada:

`simulated-words-punct-v1+chat-overhead-4`

Ela conta palavras/pontuação do conteúdo, marcadores de papel e 4 tokens por
mensagem. **Não** é o tokenizador oficial do DeepSeek; o `usage.prompt_tokens`
da API é a referência. A interface `ChatTokenCounter` permite injetar um
tokenizador real:

```python
class ModelCounter:
    name = "tokenizer-real"
    simulated = False
    def count_messages(self, messages):
        return len(self.tokenizer.encode(chat_template(messages)))
```

A garantia de encaixe vale para o contador injetado. Na integração real, conte
o template/chat completo, incluindo tokens especiais e overhead por mensagem.

## Estrutura

```
sinapse/text.py        tokens, foco, pronomes, assinaturas de dedup
sinapse/models.py      Message, Candidate, Group, RecallResult
sinapse/grouping.py    grupos sobrepostos
sinapse/memory.py      semente, distância, associação, evidência, atenção
sinapse/budget.py      contadores, orçamento textual, diagnóstico
sinapse/chat.py        orçamento no formato de mensagens de chat
sinapse/themes.py      atenção temática, decaimento, ambiguidade
sinapse/whiteboard.py  agente e quadro branco por conversa
sinapse/sensitivity.py preferências, inferências, restrição e exclusão
sinapse/utility.py     utilidade marginal por candidato
sinapse/attributes.py  atributo pedido (preço) e evidência específica
sinapse/deepseek.py    cliente HTTP, configuração, transporte simulado
sinapse/chatbot.py     orquestração evocação → orçamento → API
sinapse/benchmark.py   benchmark de 15 casos e exportação
sinapse/answer_benchmark.py  qualidade de resposta por condição
sinapse/scale_benchmark.py   escala sintética 1k/10k/100k
sinapse/scale_recovery.py    rastreamento de perdas e busca com histórico
sinapse/longitudinal.py      preferências e revisões ao longo de semanas
sinapse/resumption.py       retomada de projeto: histórico vs Sinapse
sinapse/resumption_validation.py validação local dos pares reais
sinapse/resumption_real.py  execução pareada real e revisão cega
sinapse/resumption_review.py revisão humana cega e consolidação
sinapse/resumption_rubric_v2.py rubrica v2 e calibração local
sinapse/instruction_experiment.py comparação de instruções de geração
sinapse/instruction_review.py revisão cega de instruções
sinapse/project_copilot.py MVP do copiloto de projetos (CLI)
sinapse/software_context.py seleção de estado técnico por tipo de pedido
sinapse/semantic.py    índice semântico local para lacunas (opcional)
sinapse/redaction.py   redação de credenciais antes do provedor/log
sinapse/context_audit.py linha de base de contexto (16 musicais + 10 de software)
sinapse/integration.py ponte local: retomada, busca e registros idempotentes
sinapse/farol_settings.py configuração global do DeepSeek no painel
sinapse/web_server.py  servidor local do Farol
.opencode/plugins/sinapse/ plugin do OpenCode (contexto e ferramentas)
AGENTS.md              fluxo do agente e comandos da integração
sinapse/correction_source.py fonte de candidatos para correções
sinapse/correction_round.py  rodada da fonte de correção
sinapse/curiosity_learning.py política de curiosidade
sinapse/curiosity_round2.py  força de vínculo
sinapse/curiosity_round3.py  vínculo compatível com dimensão/atributo
sinapse/purpose_selection.py seleção por tema e propósito
sinapse/purpose_answer.py    respostas reais por propósito
sinapse/purpose_report.py    reprocessamento sem novas chamadas
sinapse/removal_round.py     remoções dirigidas
sinapse/dependency_round.py  dependências condicionais
sinapse/attribution_round.py atribuição e ambiguidade
sinapse/genetic_policy.py    otimização genética da política
sinapse/context_reduction.py redução marginal do quadro
sinapse/cli.py         interface de todos os experimentos
benchmark/             casos de avaliação, calibração e planos
data/                  conversas e configuração
docs/                  arquitetura do prompt do copiloto
tests/                 núcleo, orçamento, temas, quadro, sensibilidade,
                       utilidade, DeepSeek, propósito, remoções, dependências
```

## Limitações

- Léxicos temáticos e regras de foco são determinísticos e manuais; não são
  aprendidos. “Como” preposicional pode enganar.
- Resolução de referências é de um salto, sem concordância de gênero/número.
- A herança temática por `reply_to` cobre apenas um salto.
- Ambiguidade é detectada por proximidade de pesos; a redução para 1 evidência
  é conservadora, não uma desambiguação semântica.
- Distância, evidência, associação, piso `0.90` e dedup são heurísticas.
- O contador da demo é estimado; a contagem real depende do tokenizador e do
  template do modelo.
- Mensagens nunca são truncadas; candidato que não cabe é pulado.
- O quadro vive em memória durante o processo; a persistência durável é o
  arquivo de mensagens originais. Sair do quadro não apaga nada dele.
- Utilidade marginal é uma heurística de novidade lexical; não entende
  paráfrases profundas nem contradições semânticas.
- A detecção de preferências e de pedidos claros é lexical; um pedido
  indireto pode não liberar a memória (falso negativo conservador).
- Exclusão é permanente no conjunto ativo desta execução; a remoção física do
  arquivo-fonte é uma operação separada.
- O benchmark simulado valida o mecanismo, não a qualidade do DeepSeek. O
  subconjunto real é pequeno e as rubricas ainda pedem revisão humana.
- A generalização para famílias de conversa realmente inéditas e arquivos
  muito maiores que 100 mil mensagens ainda não foi medida com o DeepSeek real.
- Há **23 pares cegos pendentes** de revisão humana; nenhum resultado humano
  foi incorporado às métricas. As **6 respostas anteriores de retomada**, as
  **18 da rodada 2**, as **30 simuladas desta rodada** e os **10 pares cegos
  reais** também aguardam revisão, contabilizados separadamente.
- A retomada real foi executada com `deepseek-flash`, mas a rubrica congelada
  rejeitou paráfrases válidas de decisão (“está valendo”, “é o”). O diagnóstico
  separado encontrou o fato em 9/9 respostas; a rubrica não foi alterada.
  A nota simulada de 1,00 mede seleção e rubrica, não a qualidade do modelo.
- O copiloto de projetos é **planejamento de produto**; não há implementação
  do fluxo de produção nesta fase.
- A seleção por propósito e a omissão condicional de dependências usam regras
  lexicais e heurísticas; os ganhos de tokens não garantem ganho de resposta.
- Os números de 100k mensagens são de um benchmark sintético local de seleção,
  não de qualidade de resposta. A melhoria de busca elevou a cobertura de 5/8
  para 7/8 ao custo de mais candidatos e latência; `correcao` continua
  descoberta.
