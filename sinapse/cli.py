"""Interface de linha de comando da Sinapse."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from .budget import (
    BaseInput,
    BudgetResult,
    RecentTurn,
    SimulatedCounter,
    render_seed_block,
    select_context,
    wrap_memory,
)
from .benchmark import export, load_cases, run_benchmark, summarize
from .answer_benchmark import export_answers, run_answer_benchmark, summarize_answers
from .learning import run_experiment
from .curiosity_learning import run_experiment as run_curiosity_experiment
from .curiosity_round2 import run_round2
from .curiosity_round3 import run_round3
from .context_reduction import run_context_reduction
from .paired_answer import export_paired, run_paired
from .genetic_policy import run_genetic_experiment
from .policy_answer import export_policy_answer, run_policy_answer
from .purpose_selection import run_purpose_benchmark
from .purpose_answer import export_purpose_answer, run_purpose_answer
from .purpose_report import export_human_review, reprocess
from .removal_round import export_removal_round, run_removal_round
from .dependency_round import export_dependency_review, run_dependency_round
from .attribution_round import export_attribution_review, run_attribution_round
from .scale_benchmark import run_scale
from .scale_recovery import export_recovery, run_recovery
from .resumption import export_resumption, run_resumption_benchmark
from .resumption_validation import export_validation, validate_real_comparison
from .resumption_real import export_real_pairs, run_real_pairs
from .project_copilot import (
    DEFAULT_ROOT,
    ProjectStore,
    chat_with_deepseek,
    format_resume,
    local_answer,
)
from .integration import (
    DEFAULT_QUERY,
    ENTRY_KINDS,
    ProjectMemory,
    list_projects,
)
from .context_audit import format_report as format_context_audit, run_audit
from .diagnostics import DiagnosticLog
from .web_server import run_server
from .resumption_review import consolidate_review, prepare_blind_review
from .resumption_rubric_v2 import calibrate, export_calibration
from .instruction_experiment import (
    export_instruction_experiment,
    run_instruction_experiment,
)
from .instruction_review import consolidate_instruction_review
from .anticipation_benchmark import (
    export_anticipation_benchmark,
    run_anticipation_benchmark,
)
from .chat import EstimatedChatCounter, build_base_messages
from .chatbot import SinapseChatbot
from .config import load_config
from .deepseek import (
    DeepSeekClient,
    MissingApiKey,
    SimulatedDeepSeekClient,
    load_deepseek_settings,
)
from .grouping import build_groups
from .memory import recall
from .models import DIMENSIONS, Candidate, Message, load_demo, load_messages
from .themes import ThemeReport
from .whiteboard import AgentTurn, MemoryAgent


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "demo_conversation.json"
DEFAULT_CONFIG = ROOT / "data" / "config.json"
AGENT_DATA = ROOT / "data" / "agent_conversation.json"

AGENT_TURNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "c1",
        "Por que eu escolhi o preset Serum 2?",
        ("Falamos do Serum 2 e do bass.",),
    ),
    ("c1", "E como eu testei?", ("Por que eu escolhi o preset Serum 2?",)),
    ("c1", "Passei a tarde jogando xadrez.", ("E como eu testei?",)),
    (
        "c1",
        "Voltando ao preset: por que eu escolhi ele?",
        ("Passei a tarde jogando xadrez.",),
    ),
    (
        "c1",
        "Como eu divulguei o pack do bass?",
        ("Voltando ao preset: por que eu escolhi ele?",),
    ),
    ("c1", "E aquilo que eu preparei?", ("Como eu divulguei o pack do bass?",)),
)


def _load(args: argparse.Namespace) -> tuple[list[Message], dict[str, Any]]:
    data_path = Path(getattr(args, "data", DEFAULT_DATA))
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG))
    return load_messages(data_path), load_config(config_path)


def _base_from_args(args: argparse.Namespace, config: dict[str, Any]) -> BaseInput:
    turns = tuple(
        RecentTurn(text=text, author="user")
        for text in (getattr(args, "context", None) or [])
    )
    return BaseInput(
        instructions=str(config.get("instructions", "")),
        recent_turns=turns,
        current_turn=args.query,
        explicit_message_ids=tuple(getattr(args, "base_message_id", None) or ()),
    )


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.3f}"


def _fmt_vector(evidence: dict[str, float | None]) -> str:
    return "{" + ", ".join(f"{d}:{_fmt_score(evidence.get(d))}" for d in DIMENSIONS) + "}"


def _print_recall(recall_result, config: dict[str, Any]) -> None:
    print("CONTADOR: simulado — simulated-words-punct-v1")
    print(f"SEMENTE: {recall_result.seed.id} — {recall_result.seed.text}")
    print(
        f"  enviado_em={recall_result.seed.sent_at} | "
        f"quando_anotado={list(recall_result.seed.values('quando')) or '(vazio)'}"
    )
    print(f"FOCO INFERIDO: {recall_result.focus} (regra: {recall_result.focus_rule})")
    print(f"REFERÊNCIAS: {json.dumps(recall_result.references.as_dict(), ensure_ascii=False)}")
    print("PESOS: " + ", ".join(
        f"{d}={recall_result.weights[d]:.2f}" for d in DIMENSIONS
    ))
    print(f"RAIO: {recall_result.radius:.2f}")
    print("CANDIDATOS (evocação):")
    print("  ID   dist   assoc  origem   atenção  raio   vetor")
    for candidate in sorted(
        recall_result.candidates,
        key=lambda c: (-c.attention, c.distance, c.message.position, c.message.id),
    ):
        print(
            f"  {candidate.message.id:<4} {candidate.distance:5.3f} "
            f"{candidate.association:5.3f}  {candidate.association_source:<7} "
            f"{candidate.attention:6.3f}  {'sim' if candidate.within_radius else 'não':<5} "
            f"{_fmt_vector(dict(candidate.evidence))}"
        )


def _print_budget(result: BudgetResult) -> None:
    d = result.diagnostics
    print("ORÇAMENTO E SELEÇÃO:")
    print(
        f"  contador={d.counter_name} (simulado={str(d.counter_simulated).lower()}) | "
        f"janela={d.window} | reserva_saída={d.output_reserve}"
    )
    print(
        f"  base={d.base_tokens} | capacidade_entrada={d.capacity} | "
        f"orçamento_memória={d.memory_budget}"
    )
    print(
        f"  memória_usada={d.memory_tokens} | entrada_usada={d.used_tokens} | "
        f"saldo={d.remaining_tokens} | entrada+reserva={d.total_with_reserve}/{d.window}"
    )
    if d.status == "base_overflow":
        print(
            f"  STATUS: base_overflow — entrada-base excede a capacidade em "
            f"{d.base_overflow_tokens} tokens; o contexto montado excede a janela"
        )
    else:
        print(f"  STATUS: {d.status}")
    print(
        f"  SEMENTE: incluída={str(d.seed_included).lower()} | origem={d.seed_source} | "
        f"custo={d.seed_tokens} tokens | já_na_base={str(d.seed_in_base).lower()}"
    )
    print(f"  EVIDÊNCIAS SELECIONADAS: {', '.join(d.selected_ids) if d.selected_ids else '(nenhuma)'}")
    print("  DECISÕES POR CANDIDATO:")
    for decision in d.decisions:
        rank = f" #{decision.rank}" if decision.rank else ""
        detail = f" — {decision.detail}" if decision.detail else ""
        print(
            f"    {decision.id:<4} dist={decision.distance:5.3f} "
            f"assoc={decision.association:5.3f}({decision.association_source}; "
            f"texto={decision.association_content:.3f}, vínculo={decision.association_link:.3f}) "
            f"atenção={decision.attention:5.3f} tokens={decision.tokens} "
            f"motivo={decision.reason}{rank}{detail}"
        )
        print(
            f"         vetor={_fmt_vector(dict(decision.evidence))} | "
            f"enviado_em={decision.sent_at} | "
            f"quando_anotado={list(decision.event_time) or '(vazio)'} | "
            f"resposta_a={decision.reply_to or '-'}"
        )
    print("CONTEXTO ENTREGUE AO CHATBOT:")
    print(result.context_text)


def _run_query(
    messages: Sequence[Message],
    config: dict[str, Any],
    query: str,
    recent_turns: Sequence[str],
    *,
    window: int,
    output_reserve: int,
    seed_id: str | None = None,
    focus: str | None = None,
    radius: float | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    use_links: bool = True,
    include_seed: bool = True,
    base_message_ids: Sequence[str] = (),
    counter: SimulatedCounter | None = None,
) -> tuple[Any, BudgetResult, BaseInput]:
    recall_result = recall(
        messages,
        query,
        recent_turns,
        config,
        seed_id=seed_id,
        focus=focus,
        radius=radius,
        use_links=use_links,
    )
    base = BaseInput(
        instructions=str(config.get("instructions", "")),
        recent_turns=tuple(RecentTurn(text=t, author="user") for t in recent_turns),
        current_turn=query,
        explicit_message_ids=tuple(base_message_ids),
    )
    result = select_context(
        base,
        recall_result.seed,
        recall_result.focus,
        recall_result.candidates,
        window=window,
        output_reserve=output_reserve,
        counter=counter or SimulatedCounter(),
        top_k=int(top_k if top_k is not None else config.get("top_k", 3)),
        threshold=float(threshold if threshold is not None else config.get("attention_threshold", 0.05)),
        include_seed=include_seed,
    )
    return recall_result, result, base


def cmd_ask(args: argparse.Namespace) -> int:
    messages, config = _load(args)
    recall_result, budget_result, _ = _run_query(
        messages,
        config,
        args.query,
        args.context or [],
        window=args.window if args.window is not None else int(config["window"]),
        output_reserve=(
            args.output_reserve if args.output_reserve is not None else int(config["output_reserve"])
        ),
        seed_id=args.seed,
        focus=args.focus,
        radius=args.radius,
        top_k=args.top_k,
        threshold=args.threshold,
        use_links=not args.no_links,
        include_seed=not args.no_seed,
        base_message_ids=args.base_message_id or (),
    )
    if args.json:
        payload = {
            "recall": recall_result.as_dict(),
            "budget": budget_result.as_dict(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_recall(recall_result, config)
    _print_budget(budget_result)
    return 0


def _demo_query(
    messages: Sequence[Message],
    config: dict[str, Any],
    spec: dict[str, Any],
    *,
    window: int,
    output_reserve: int,
) -> tuple[Any, BudgetResult]:
    recall_result, budget_result, _ = _run_query(
        messages,
        config,
        spec["text"],
        spec.get("recent_turns", []),
        window=window,
        output_reserve=output_reserve,
    )
    print("\n" + "=" * 78)
    print(f"CONSULTA [{spec['id']}]: {spec['text']}")
    _print_recall(recall_result, config)
    _print_budget(budget_result)
    return recall_result, budget_result


def cmd_demo(args: argparse.Namespace) -> int:
    data = load_demo(Path(args.data))
    messages = load_messages(Path(args.data))
    config = load_config(Path(args.config))
    window = int(config["window"])
    reserve = int(config["output_reserve"])
    queries = data.get("queries", [])
    runs: dict[str, tuple[Any, BudgetResult]] = {}
    for spec in queries:
        runs[spec["id"]] = _demo_query(
            messages, config, spec, window=window, output_reserve=reserve
        )

    por_que = next((q for q in queries if q["id"] == "porque"), None)
    if por_que:
        recall_result, _, base = _run_query(
            messages,
            config,
            por_que["text"],
            por_que.get("recent_turns", []),
            window=window,
            output_reserve=reserve,
        )
        counter = SimulatedCounter()
        base_text = base.render()
        base_tokens = counter.count(base_text)
        seed_only = counter.count(
            base_text + "\n\n" + wrap_memory([render_seed_block(recall_result.seed)])
        ) - base_tokens
        tight_window = base_tokens + reserve + seed_only
        print("\n" + "=" * 78)
        print("DOIS ORÇAMENTOS PARA A MESMA CONSULTA (porque)")
        print(
            f"orçamento largo: janela={window}, reserva={reserve}; "
            f"orçamento estreito: janela={tight_window}, reserva={reserve}; "
            f"custo da semente sozinha={seed_only} tokens"
        )
        wide_recall, wide_budget, _ = _run_query(
            messages,
            config,
            por_que["text"],
            por_que.get("recent_turns", []),
            window=window,
            output_reserve=reserve,
        )
        tight_recall, tight_budget, _ = _run_query(
            messages,
            config,
            por_que["text"],
            por_que.get("recent_turns", []),
            window=tight_window,
            output_reserve=reserve,
        )
        print("\n--- ORÇAMENTO LARGO ---")
        _print_recall(wide_recall, config)
        _print_budget(wide_budget)
        print("\n--- ORÇAMENTO ESTREITO ---")
        _print_recall(tight_recall, config)
        _print_budget(tight_budget)
        wide_attention = {c.message.id: c.attention for c in wide_recall.candidates}
        tight_attention = {c.message.id: c.attention for c in tight_recall.candidates}
        print("\nCOMPARAÇÃO:")
        print(f"  evidências largas={wide_budget.diagnostics.selected_ids}")
        print(f"  evidências estreitas={tight_budget.diagnostics.selected_ids}")
        print(f"  atenções iguais={wide_attention == tight_attention}")
        for message_id in sorted(wide_attention):
            print(
                f"    {message_id}: larga={wide_attention[message_id]:.6f} "
                f"estreita={tight_attention[message_id]:.6f}"
            )
    return 0


def cmd_groups(args: argparse.Namespace) -> int:
    messages, config = _load(args)
    threshold = args.threshold if args.threshold is not None else float(config["group_threshold"])
    index = build_groups(messages, threshold=threshold)
    if args.json:
        print(json.dumps(
            {"threshold": threshold, "groups": [g.as_dict() for g in index.groups]},
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    print(f"CONTADOR: simulado — simulated-words-punct-v1")
    print(f"LIMIAR DE GRUPO: {threshold}")
    for group in index.groups:
        print(
            f"{group.id:<16} dim={group.dimension:<8} label={group.label!r} "
            f"mensagens={', '.join(group.message_ids)}"
        )
    return 0


def cmd_group(args: argparse.Namespace) -> int:
    messages, config = _load(args)
    threshold = float(config["group_threshold"])
    index = build_groups(messages, threshold=threshold)
    group = index.by_id(args.group_id)
    if group is None:
        print(f"Grupo não encontrado: {args.group_id}", file=sys.stderr)
        return 2
    by_id = {m.id: m for m in messages}
    print("CONTADOR: simulado — simulated-words-punct-v1")
    print(f"GRUPO {group.id} | dimensão={group.dimension} | limiar={group.threshold}")
    print(f"RÓTULO: {group.label}")
    for message_id in group.message_ids:
        message = by_id[message_id]
        print(f"  {message.id}: {message.text}")
    return 0


def cmd_replies(args: argparse.Namespace) -> int:
    messages, _ = _load(args)
    replies = [m for m in messages if m.reply_to == args.message_id]
    print("CONTADOR: simulado — simulated-words-punct-v1")
    print(f"RESPOSTAS DIRETAS A {args.message_id}:")
    if not replies:
        print("  (nenhuma)")
    for message in replies:
        print(f"  {message.id}: {message.text}")
    return 0


def _print_theme_report(turn: AgentTurn) -> None:
    report: ThemeReport = turn.themes
    print("TEMAS ATIVOS:")
    for signal in report.themes:
        terms = ", ".join(signal.matched_terms) or "-"
        print(
            f"  {signal.theme:<7} peso={signal.weight:.3f} "
            f"anterior={signal.previous_weight:.3f} origem={signal.support:<18} "
            f"termos={terms}"
        )
    print(
        f"DOMINANTE: {report.dominant or '(nenhum)'} | "
        f"DIMENSÃO PEDIDA: {report.requested_dimension} "
        f"(regra: {report.dimension_rule}) | AMBÍGUO: {report.ambiguous}"
    )
    if report.ambiguous:
        print(
            f"  MOTIVO DA AMBIGUIDADE: {report.ambiguity_reason}; "
            "quadro limitado a 1 evidência"
        )
    print("SINAIS OBSERVÁVEIS:")
    for signal in report.signals:
        print(f"  - {signal}")


def _print_agent_turn(turn: AgentTurn, settings, title: str | None = None) -> None:
    if title:
        print("\n" + "=" * 78)
        print(title)
    print("=" * 78)
    print(
        f"TURNO {turn.turn_index} | conversa={turn.conversation_id} | "
        f"pergunta={turn.query!r}"
    )
    _print_theme_report(turn)
    print(f"SEMENTE: {turn.seed.id} — {turn.seed.text}")
    delta = turn.delta
    print(
        "QUADRO: "
        f"entraram={list(delta.entered) or '-'} | "
        f"permaneceram={list(delta.stayed) or '-'} | "
        f"saíram={list(delta.left) or '-'}"
    )
    for message_id, reason in delta.left_reasons.items():
        print(f"  saiu {message_id}: {reason}")
    for decision in turn.decisions:
        focus = (
            "null" if decision.evidence_focus is None else f"{decision.evidence_focus:.3f}"
        )
        themes = ",".join(decision.themes) or "-"
        sent = "sim" if decision.sent else "não"
        print(
            f"  {decision.message_id:<4} {decision.role:<9} ação={decision.action:<7} "
            f"rel={decision.association:.3f} dist={decision.distance:.3f} "
            f"atenção={decision.attention:.3f} (base {decision.base_attention:.3f}) "
            f"evid[{turn.focus}]={focus} temas={themes} enviado={sent} "
            f"motivo={decision.reason}"
        )
        if decision.budget_reason:
            print(f"       orçamento: {decision.budget_reason}")

    d = turn.budget.diagnostics
    if d.seed_included:
        evidence_tokens = d.memory_tokens - d.seed_tokens
    else:
        evidence_tokens = d.memory_tokens
    print(
        f"ORÇAMENTO: base={d.base_tokens} capacidade_entrada={d.capacity} "
        f"orçamento_memória={d.memory_budget} memória_usada={d.memory_tokens} "
        f"entrada_usada={d.used_tokens} saldo={d.remaining_tokens} "
        f"entrada+reserva={d.total_with_reserve}/{d.window} "
        f"status={d.status} contagem={'estimada' if d.counter_simulated else 'real'}"
    )
    print(
        f"CUSTO DO QUADRO: {d.memory_tokens} tokens "
        f"(semente={d.seed_tokens}, evidências={evidence_tokens}, "
        f"contador={d.counter_name})"
    )
    print("ENVIADO AO CHATBOT:")
    for index, message in enumerate(turn.messages, start=1):
        print(f"  [{index}] {message['role']}:")
        for line in str(message["content"]).splitlines():
            print(f"      {line}")


def cmd_agent_demo(args: argparse.Namespace) -> int:
    messages = load_messages(Path(args.agent_data))
    config = load_config(Path(args.config))
    settings = load_deepseek_settings(config)
    agent = MemoryAgent(messages, config)
    turns: list[AgentTurn] = []
    for _, query, recent in AGENT_TURNS:
        turns.append(agent.before_response("c1", query, recent))

    if args.json:
        payload = {
            "settings": {
                "model": settings.model,
                "context_window": settings.context_window,
                "output_reserve": settings.output_reserve,
            },
            "turns": [turn.as_dict() for turn in turns],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for turn in turns:
        _print_agent_turn(turn, settings)

    # Orçamento zero: mesma pergunta, agente novo, nenhuma memória adicional.
    fresh = MemoryAgent(messages, config)
    _, query, recent = AGENT_TURNS[0]
    base = build_base_messages(
        str(config.get("instructions", "")),
        tuple(RecentTurn(text=text) for text in recent),
        query,
    )
    base_tokens = fresh.counter.count_messages(base)
    zero_window = base_tokens + settings.output_reserve
    zero_turn = fresh.before_response(
        "c1", query, recent, window=zero_window, output_reserve=settings.output_reserve
    )
    _print_agent_turn(
        zero_turn,
        settings,
        title=(
            "ORÇAMENTO ZERO (agente novo, mesma pergunta; "
            f"janela={zero_window}, reserva={settings.output_reserve})"
        ),
    )
    print(
        "  OBS.: o quadro continua com "
        f"{list(fresh.board_for('c1').ids())}; nada foi enviado além do turno."
    )

    # Isolamento: o mesmo agente atende c2 sem tocar no quadro de c1.
    c2_turn = agent.before_response(
        "c2", "Por que comprei a bicicleta?", ("Falamos de bicicleta.",)
    )
    _print_agent_turn(
        c2_turn, settings, title="ISOLAMENTO ENTRE CONVERSAS (c2 no mesmo agente)"
    )
    c1_ids = set(agent.board_for("c1").ids())
    c2_ids = set(agent.board_for("c2").ids())
    print(f"  c1={sorted(c1_ids)}")
    print(f"  c2={sorted(c2_ids)}")
    print(f"  interseção={sorted(c1_ids & c2_ids)}")
    return 0


def cmd_scale_benchmark(args: argparse.Namespace) -> int:
    scales = [int(value) for value in args.scales.split(",") if value.strip()]
    config = load_config(Path(args.config))
    results = [
        run_scale(scale, args.seed, args.candidate_k, config) for scale in scales
    ]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("BENCHMARK DE ESCALA SINAPSE")
    print(
        "Dados sintéticos determinísticos | sem chamadas ao DeepSeek | "
        f"semente={args.seed} | candidate_k={args.candidate_k}"
    )
    print("Consultas com evidência esperada: 8/11 (sem sensibilidade e sem resposta).")
    for result in results:
        summary = result["summary"]
        print("\n" + "=" * 78)
        print(
            f"ESCALA {result['scale']} | mensagens={result['messages']} | "
            f"geração={result['build_ms']} ms | índice={result['index_ms']} ms"
        )
        print(
            "condição        cand_all   final_all  busca med/p95   seleção med/p95   "
            "pontos  tokens"
        )
        for condition in ("sinapse", "sinapse_curioso", "lexical"):
            item = summary[condition]
            print(
                f"{condition:<14} {item['candidate_all']}/{item['denominator']:<7} "
                f"{item['final_all']}/{item['denominator']:<9} "
                f"{item['search_median_ms']:>7.1f}/{item['search_p95_ms']:<7.1f} "
                f"{item['select_median_ms']:>8.1f}/{item['select_p95_ms']:<7.1f} "
                f"{item['avg_points']:>7.1f} {item['avg_tokens']:>7.1f}"
            )
        comparison = summary["curiosity_comparison"]
        print(
            "curiosidade: "
            f"evidências corretas adicionais={comparison['additional_correct_evidence']} | "
            f"candidatos extras={comparison['extra_candidates_evaluated']} | "
            f"latência Δmediana={comparison['select_ms_delta_median']} ms | "
            f"tokens Δmédio={comparison['tokens_delta_avg']} | "
            f"piorou={comparison['worsened_cases'] or '-'}"
        )
        print(
            "sem resposta: "
            f"sinapse={summary['sinapse']['no_answer_empty']}/"
            f"{summary['sinapse']['no_answer_total']} | "
            f"curiosa={summary['sinapse_curioso']['no_answer_empty']}/"
            f"{summary['sinapse_curioso']['no_answer_total']} | "
            f"lexical={summary['lexical']['no_answer_empty']}/"
            f"{summary['lexical']['no_answer_total']} | "
            f"exclusão por sensibilidade={summary['sinapse']['sensitivity_excluded']}/1"
        )
        errors = [
            outcome
            for outcome in result["outcomes"]
            if outcome["condition"] == "sinapse" and outcome["missing_stage"]
        ]
        print("primeiros erros (Sinapse desligada):")
        for outcome in errors[:3]:
            print(
                f"  {outcome['query']}: perdeu em {outcome['missing_stage']} | "
                f"candidatos={outcome['candidate_ids']} "
                f"pontuados={outcome['scored_ids']} finais={outcome['final_ids']}"
            )
        if not errors:
            print("  (nenhum)")
        curious_errors = [
            outcome
            for outcome in result["outcomes"]
            if outcome["condition"] == "sinapse_curioso" and outcome["missing_stage"]
        ]
        if curious_errors:
            print("primeiros erros (Sinapse com curiosidade):")
            for outcome in curious_errors[:3]:
                print(
                    f"  {outcome['query']}: perdeu em {outcome['missing_stage']} | "
                    f"extras={outcome['extra_candidates']} novos={outcome['new_ids']}"
                )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scale_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nArquivo: {output / 'scale_results.json'}")
    return 0


def cmd_scale_recovery(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_recovery(args.scale, args.seed, args.candidate_k, config)
    path = export_recovery(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("RECUPERAÇÃO EM ESCALA SINAPSE")
    print(
        f"escala={result['scale']} | semente={result['seed']} | "
        f"candidate_k={result['candidate_k']} | orçamento={result['budget']} | "
        "sem chamadas ao DeepSeek"
    )
    print(
        "Consultas com evidência esperada: 8 regressões + 3 famílias novas; "
        "as 3 consultas sem evidência ficam fora do agregado."
    )
    print("\nRASTREAMENTO DAS PERDAS (antes da melhoria)")
    for trace in result["traces"]:
        print(
            f"  {trace['query']}/{trace['expected']}: "
            f"indexado={trace['indexed']} fonte={trace['source']} "
            f"rank_antes={trace['rank_before_cut']} "
            f"pos_depois={trace['position_after_cut_and_expansion']} "
            f"quadro={trace['in_board']} final={trace['in_final']} "
            f"pai={trace['expected_reply_to'] or '-'} "
            f"pai_no_corte={trace['parent_in_shortlist']} "
            f"expansão_alcançou={trace['link_expansion_reached']} "
            f"primeira_perda={trace['first_loss_stage']}"
        )
    for label, group in (("REGRESSÕES", "regression"), ("NOVAS FAMÍLIAS", "new")):
        rows = [row for row in result["rows"] if row["group"] == group]
        print(f"\n{label} ({len(rows)} consultas)")
        for row in rows:
            print(
                f"  {row['query']:<22} cand {int(row['candidate_before'])}→"
                f"{int(row['candidate_after'])} | quadro {int(row['board_before'])}→"
                f"{int(row['board_after'])} | final {int(row['final_before'])}→"
                f"{int(row['final_after'])} | pontos {row['points_before']}→"
                f"{row['points_after']} | busca {row['search_ms_before']}→"
                f"{row['search_ms_after']} ms | memória "
                f"{row['memory_tokens_before']}→{row['memory_tokens_after']} tokens"
            )
    for name, label in (("regression", "regressões"), ("new", "novas famílias")):
        item = result["aggregates"][name]
        print(
            f"\nAGREGADO ({label}): candidatos {item['candidate_before']}/"
            f"{item['queries']}→{item['candidate_after']}/{item['queries']} | "
            f"quadro {item['board_before']}/{item['queries']}→"
            f"{item['board_after']}/{item['queries']} | final "
            f"{item['final_before']}/{item['queries']}→"
            f"{item['final_after']}/{item['queries']} | melhoraram="
            f"{item['improved']} pioraram={item['regressed']} | pontos "
            f"{item['points_before']}→{item['points_after']} | busca "
            f"{item['search_sum_before']}→{item['search_sum_after']} ms | "
            f"memória {item['memory_before']}→{item['memory_after']} tokens"
        )
    print(f"\nArquivo: {path}")
    return 0


def cmd_resumption_benchmark(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_resumption_benchmark(config, settings, client)
    paths = export_resumption(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("RETOMADA DE PROJETO")
    print(
        f"MODO: {result['mode']} | modelo={result['model']} | "
        f"casos={result['cases']} | 6 conversas, 2 a 4 sessões"
    )
    for row in result["rows"]:
        checks = row["checks"]
        print("\n" + "=" * 78)
        print(
            f"{row['case']} [{row['condition']}] nota={row['score']:.2f} | "
            f"decisão={checks['decision']} motivo={checks['reason']} "
            f"concluído={checks['completed']} pendência={checks['pending']}"
        )
        print(
            f"  evidência={row['evidence_present']} cobertura={row['evidence_coverage']:.2f} | "
            f"refs={row['references']} | seleção={row['selected_ids']}"
        )
        print(
            f"  memória={row['memory_tokens']} tokens | "
            f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
            f"({row['token_source']}) | seleção={row['select_ms']} ms"
        )
        if checks["completed_repetition"]:
            print(f"  repetiu concluído: {checks['completed_repetition']}")
        if checks["unsupported_claims"]:
            print(f"  sem suporte: {checks['unsupported_claims']}")
        if checks["conditions_omitted"]:
            print(f"  condição omitida: {checks['conditions_omitted']}")
        if row["old_tasks_as_current"]:
            print(f"  tarefa antiga como atual: {row['old_tasks_as_current']}")
        if row["wrong_project"]:
            print(f"  projeto errado: {row['wrong_project']}")
        if row["errors"]:
            print(f"  erros: {row['errors']}")
        print(f"  resposta: {row['answer']}")
    print("\nRASTREAMENTO")
    for trace in result["traces"]:
        if trace["condition"] == "sinapse":
            pending = [
                item
                for item in trace["candidates"]
                if item["id"] in {"pr6", "pr7"}
            ]
            print(
                f"  {trace['case']} [sinapse] foco={trace['focus']} "
                f"semente={trace['seed']} propósito={trace['purpose_inferred']} "
                f"seleção={trace['selected']}"
            )
            for item in pending:
                print(
                    f"    {item['id']}: candidata=True raio={item['within_radius']} "
                    f"atenção={item['attention']} razão={item['reason'] or '-'}"
                )
        elif trace.get("passes"):
            print(
                f"  {trace['case']} [multi-propósito] "
                f"propósito={trace['purpose_inferred']} "
                f"seleção={trace['selected']}"
            )
            for item in trace["passes"]:
                print(
                    f"    {item['purpose']}: semente={item['seed']} "
                    f"ids={item['ids']}"
                )
    print("\nAGREGADO")
    for condition, item in result["aggregates"].items():
        print(
            f"  {condition:<20} decisão={item['decision_ok']}/{item['decision_total']} "
            f"motivo={item['reason_ok']}/{item['reason_total']} "
            f"concluído={item['completed_ok']}/{item['completed_total']} "
            f"pendência={item['pending_ok']}/{item['pending_total']} "
            f"cobertura={item['coverage']:.2f} refs={item['references_all']} "
            f"suporte={item['reference_support_all']} "
            f"condição={item['conditions_ok']}/{item['conditions_total']} "
            f"antigas={item['old_tasks_as_current']} "
            f"projeto_errado={item['wrong_project']} "
            f"erros={item['errors']} "
            f"nota={item['avg_score']:.2f} repetições={item['completed_repetitions']} "
            f"sem_suporte={item['unsupported_claims']} "
            f"prompt={item['prompt_tokens_total']} completion={item['completion_tokens_total']} "
            f"memória_média={item['memory_tokens_avg']}"
        )
    print(
        f"\nSinapse > histórico={result['sinapse_better_history']} | "
        f"multi > Sinapse={result['multipurpose_better']} | "
        f"multi pior={result['multipurpose_worse']} | "
        f"avaliação real disponível={result['real_evaluation']['available']} | "
        f"pareamento real disponível={result['real_pairing']['available']}"
    )
    print(
        f"Arquivos: {paths['results']} | {paths['review']} | "
        f"{paths['real_comparison']}"
    )
    return 0


def cmd_resumption_review_prepare(args: argparse.Namespace) -> int:
    path = prepare_blind_review(
        Path(args.review), Path(args.results), Path(args.out)
    )
    print(f"Revisão cega preparada: {path}")
    print("Campos humanos deixados em branco; condições A/B não reveladas.")
    return 0


def cmd_resumption_review_consolidate(args: argparse.Namespace) -> int:
    result = consolidate_review(
        Path(args.review), Path(args.results), Path(args.out)
    )
    if not result["complete"]:
        print("Revisão incompleta; condições não reveladas.")
        for item in result["missing"]:
            print(f"  falta: {item}")
        print("Preencha os campos humanos e rode novamente.")
        return 2
    print("REVISÃO HUMANA CONSOLIDADA")
    for item in result["revealed_conditions"]:
        print(f"  {item['review_id']} ({item['case']}): {item['conditions']}")
    for condition, item in result["human_summary"].items():
        elements = " ".join(
            f"{name}={data['hits']}/{data['total']}"
            for name, data in item["elements"].items()
            if data["total"]
        )
        print(
            f"  {condition}: {elements} | completos={item['complete']}/"
            f"{item['rows']} | automáticos={item['automatic_complete']} | "
            f"diagnóstico={item['diagnostic_complete']}"
        )
    print(f"  divergências={len(result['divergences'])}")
    print(f"Arquivo: {args.out}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    try:
        run_server(Path(args.root), args.host, args.port, args.settings)
    except OSError as error:
        print(f"Não foi possível iniciar o Farol: {error}", file=sys.stderr)
        return 2
    return 0


def cmd_diagnostics(args: argparse.Namespace) -> int:
    log = DiagnosticLog(Path(args.root) / "diagnostics")
    if args.diagnostics_command != "project":
        print("Comando de diagnóstico não implementado", file=sys.stderr)
        return 2
    events = log.events(args.project, since=args.since)
    if args.json:
        print(json.dumps(events, ensure_ascii=False, indent=2))
        return 0
    if not events:
        print(f"Nenhum evento para o projeto {args.project}.")
        return 0
    for event in events:
        print(
            f"{event['ts']} | {event['project_id']} | "
            f"{event['stage']} | {event['status']} | {event['event_id']}"
        )
        details = event.get("details") or {}
        if details:
            print("  " + json.dumps(details, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_anticipation_benchmark(args: argparse.Namespace) -> int:
    result = run_anticipation_benchmark()
    path = export_anticipation_benchmark(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("ANTECIPAÇÃO — COMPARAÇÃO LOCAL")
    for condition in ("off", "on"):
        item = result[condition]
        print(
            f"  {condition:<3} atual={item['current_ms']}ms "
            f"seguinte={item['next_ms']}ms "
            f"reservadas={item['candidates_reserved']} "
            f"usadas={item['memories_used']} "
            f"cobertura={item['evidence_coverage']} "
            f"expiradas={item['expired_predictions']} "
            f"busca={item['anticipation_elapsed_ms']}ms"
        )
    print(f"Arquivo: {path}")
    return 0


def cmd_context_audit(args: argparse.Namespace) -> int:
    result = run_audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_context_audit(result))
        print(f"Arquivo: {out}")
    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    command = args.bridge_command
    try:
        if command == "projects":
            payload = list_projects(args.root)
        else:
            memory = ProjectMemory(
                args.repo, root=args.root, project_id=args.project
            )
            if command == "resume":
                payload = memory.resume(args.query)
            elif command == "search":
                payload = memory.search(args.query, args.limit)
            elif command == "status":
                payload = memory.status()
            elif command == "link":
                payload = memory.link(args.project)
            elif command == "record":
                payload = memory.record(
                    args.kind,
                    args.text,
                    entry_id=args.entry_id,
                    supersedes=args.supersedes,
                    reason=args.reason,
                    resolves=args.resolves,
                    author=args.author,
                    component=args.component,
                    files=args.file,
                    issues=args.issue,
                    prs=args.pr,
                    commits=args.commit,
                    command=args.command,
                    symptom=args.symptom,
                    cause=args.cause,
                    restrictions=args.restriction,
                    alternatives=args.alternative,
                    report_source=args.report_source,
                    reported_by=args.reported_by,
                    status=args.status,
                    related_to=args.related_to,
                )
            elif command == "brief":
                payload = memory.brief(args.query)
            elif command == "code-status":
                payload = memory.code_status()
            elif command == "code-import":
                payload = memory.code_import(args.commit)
            elif command == "code-search":
                payload = memory.code_search(args.query, args.limit)
            else:
                raise ValueError(f"Comando de ponte desconhecido: {command}")
    except ValueError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        else:
            print(str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload.get("message") or payload.get("formatted") or "")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    store = ProjectStore(Path(args.root))
    command = args.project_command
    try:
        if command == "create":
            project = store.create(args.name)
            print(f"Projeto criado: {project['name']} ({project['id']})")
            return 0
        if command == "list":
            for item in store.list():
                marker = "*" if item["active"] else " "
                print(f"{marker} {item['id']:<24} {item['name']}")
            return 0
        if command == "select":
            project = store.select(args.project)
            print(f"Projeto ativo: {project['name']} ({project['id']})")
            return 0
        project = store.load()
        if command == "say":
            store.add_message(project, args.text, author="user")
            if args.deepseek:
                try:
                    answer = chat_with_deepseek(project, args.text)
                except Exception as error:
                    print(f"Aviso: DeepSeek indisponível: {error}", file=sys.stderr)
                    answer = local_answer(store, project, args.text)
            else:
                answer = local_answer(store, project, args.text)
            store.add_message(project, answer, author="assistant")
            print(answer)
            return 0
        if command == "resume":
            result = store.resume(project, args.query)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(format_resume(result))
            return 0
        if command == "show":
            message = store.show(args.message_id, project)
            print(
                f"[{message['id']}] {message['project_name']} "
                f"({message['project_id']}) | {message['author']} | "
                f"{message['created_at']}"
            )
            print(message["text"])
            return 0
        if command == "restrict":
            store.restrict(args.text, project)
            print(f"Restrição registrada no projeto {project['id']}.")
            return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 2


def cmd_instruction_review_consolidate(args: argparse.Namespace) -> int:
    result = consolidate_instruction_review(
        Path(args.review), Path(args.results), Path(args.out)
    )
    if not result["complete"]:
        print("Revisão incompleta; instruções não reveladas.")
        for item in result["missing"]:
            print(f"  falta: {item}")
        return 2
    print("REVISÃO CEGA DE INSTRUÇÃO CONSOLIDADA")
    for item in result["revealed_conditions"]:
        print(
            f"  {item['review_id']} ({item['case']}): "
            f"{item['conditions']} | preferida={item['preferred']}"
        )
    print(
        f"preferências: {result['preferences']} | "
        f"divergências={result['divergences_by_criterion']}"
    )
    for condition, item in result["human_summary"].items():
        human = " ".join(
            f"{name}={value['hits']}/{value['total']}"
            for name, value in item["human_criteria"].items()
            if value["total"]
        )
        v2 = " ".join(
            f"{name}={value['hits']}/{value['total']}"
            for name, value in item["v2_criteria"].items()
            if value["total"]
        )
        print(
            f"  {condition:<14} humano: {human} | completos={item['human_complete']}/"
            f"{item['rows']} | v2 completos={item['v2_complete']} | "
            f"prompt={item['prompt_tokens']} completion={item['completion_tokens']} "
            f"memória_média={item['memory_tokens_avg']}"
        )
        print(f"    v2: {v2}")
    print(
        f"custo real: Δprompt={result['cost_delta_prompt']} "
        f"Δcompletion={result['cost_delta_completion']} | "
        f"instrução={result['instruction_cost']}"
    )
    print(f"Arquivo: {args.out}")
    return 0


def cmd_instruction_experiment(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    settings = load_deepseek_settings(config)
    if args.model:
        settings = replace(settings, model=args.model)
    try:
        settings.validate()
        client = DeepSeekClient(base_url=settings.base_url)
    except (ValueError, MissingApiKey) as error:
        print(f"Bloqueio: {error}", file=sys.stderr)
        return 2
    result = run_instruction_experiment(
        config, settings, client, Path(args.plan)
    )
    paths = export_instruction_experiment(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("EXPERIMENTO DE INSTRUÇÃO")
    print(
        f"modelo={result['model']} | pares={result['pairs']} | "
        f"chamadas={result['calls']} | tokens={result['real_tokens']}"
    )
    for name, item in result["aggregates"].items():
        criteria = " ".join(
            f"{criterion}={data['hits']}/{data['total']}"
            for criterion, data in item["criteria"].items()
            if data["total"]
        )
        print(
            f"  {name:<14} completos={item['complete']}/{item['rows']} "
            f"{criteria} prompt={item['prompt_tokens']} "
            f"completion={item['completion_tokens']} "
            f"memória_média={item['memory_tokens_avg']} erros={item['errors']}"
        )
    print(
        f"  pares: melhoraram={result['improved']} "
        f"empataram={result['tied']} pioraram={result['worse']}"
    )
    print(f"  custo da instrução: {result['instruction_cost']}")
    print(
        f"  RP02: {result['rp02']} | Dunas: {result['dunas']}"
    )
    print(f"Arquivos: {paths['results']} | {paths['review']}")
    return 0


def cmd_resumption_rubric_v2(args: argparse.Namespace) -> int:
    result = calibrate()
    path = export_calibration(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("RUBRICA V2 — CALIBRAÇÃO LOCAL")
    print(f"pares={result['rows']}")
    for name, item in result["criteria"].items():
        print(
            f"  {name:<12} tp={item['tp']} tn={item['tn']} "
            f"fp={item['fp']} fn={item['fn']}"
        )
    passed = sum(
        1
        for item in result["synthetic"]
        if all(item["v2"][name] == gold for name, gold in item["gold"].items())
    )
    print(f"casos sintéticos={len(result['synthetic'])} | corretos={passed}")
    print("chamadas ao DeepSeek: 0 (usa respostas já salvas)")
    print(f"Arquivo: {path}")
    return 0


def cmd_resumption_real_pairs(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    settings = load_deepseek_settings(config)
    if args.model:
        settings = replace(settings, model=args.model)
    try:
        settings.validate()
        client = DeepSeekClient(base_url=settings.base_url)
    except (ValueError, MissingApiKey) as error:
        print(f"Bloqueio: {error}", file=sys.stderr)
        print(
            "Configure DEEPSEEK_API_KEY e rode: "
            "python -m sinapse resumption-real-pairs",
            file=sys.stderr,
        )
        return 2
    result = run_real_pairs(
        Path(args.pairs), config, settings, client
    )
    paths = export_real_pairs(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("RETOMADA REAL — PARES PAREADOS")
    print(
        f"modelo={result['model']} | pares={result['pairs']} | "
        f"chamadas={result['calls']} | tokens={result['real_tokens']}"
    )
    for condition, item in result["aggregates"].items():
        print(
            f"  {condition:<20} completos={item['complete']}/{item['rows']} "
            f"(diagnóstico={item['complete_diagnostic']}/{item['rows']}) "
            f"decisão={item['decision_ok']}/{item['decision_total']} "
            f"(fato={item['decision_fact_only']}) "
            f"motivo={item['reason_ok']}/{item['reason_total']} "
            f"concluído={item['completed_ok']}/{item['completed_total']} "
            f"pendência={item['pending_ok']}/{item['pending_total']} "
            f"refs_sem_falha={item['reference_support']}/{item['rows']} "
            f"condições={item['conditions_ok']}/{item['conditions_total']} "
            f"projeto={item['project_ok']}/{item['project_total']} "
            f"erros={item['errors']} "
            f"prompt={item['prompt_tokens']} completion={item['completion_tokens']}"
        )
    print(
        "\nRevisão cega: "
        + paths["review"]
        + " | resultados: "
        + paths["results"]
    )
    return 0


def cmd_resumption_validate(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    settings = load_deepseek_settings(config)
    result = validate_real_comparison(
        Path(args.pairs), config, settings
    )
    path = export_validation(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    summary = result["summary"]
    print("VALIDAÇÃO LOCAL DOS PARES DE RETOMADA")
    print(
        f"pares={summary['pairs']} | pares_válidos={summary['pair_valid']} | "
        f"seleção_congelada={summary['selection_matches_frozen']}/{summary['pairs']}"
    )
    print(
        f"completos: foco único={summary['single_complete']}/{summary['pairs']} | "
        f"multi-propósito={summary['multipurpose_complete']}/{summary['pairs']}"
    )
    print(
        f"integridade das referências: foco único={summary['single_integrity']} | "
        f"multi={summary['multipurpose_integrity']}"
    )
    print(
        f"Dunas: condição={summary['dunas_condition_ok']}/1 multi e "
        f"{summary['dunas_single_condition_ok']}/1 foco único; "
        f"dois caminhos={summary['dunas_paths_ok']}/1 multi e "
        f"{summary['dunas_single_paths_ok']}/1 foco único"
    )
    print(
        f"Norte/Sul: projeto correto={summary['project_total_ok']}/"
        f"{summary['project_total']} (resposta e referências)"
    )
    for pair in result["pairs"]:
        print(
            f"  {pair['case']}: par={'ok' if pair['pair_valid'] else 'falha'} | "
            f"único={'ok' if pair['single']['complete'] else 'faltou:' + ','.join(pair['single']['missing_elements'])} | "
            f"multi={'ok' if pair['multipurpose']['complete'] else 'faltou:' + ','.join(pair['multipurpose']['missing_elements'])}"
        )
    print(
        "\nDeepSeek real: indisponível; validação apenas local. "
        "Arquivos de revisão existentes foram preservados."
    )
    print(f"Arquivo: {path}")
    return 0


def cmd_genetic_policy(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_genetic_experiment(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "genetic_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("OTIMIZAÇÃO GENÉTICA DA POLÍTICA DE MEMÓRIA")
    print(
        f"calibração={result['calibration_cases']} | regressões conhecidas={result['known_cases']} | "
        f"famílias inéditas={result['unseen_cases']}"
    )
    print(
        f"sementes={result['seeds']} | avaliações por semente={result['evaluations_per_seed']} "
        f"(aleatória e genética com o mesmo total)"
    )
    print(f"limites={result['bounds']}")
    print("\nMELHORES POLÍTICAS (calibração)")
    for name in ("reference", "best_random", "best_genetic"):
        item = result[name]
        print(
            f"  {name:<13} política={item['policy']} chave={item['key']} "
            f"full={item['full_cases']}/{item['cases']} deps_perdidas={item['dependencies_lost']} "
            f"violações={item['violations']} sem_função={item['functionless']} "
            f"tokens={item['tokens']} candidatos={item['candidates']}"
        )
    print("\nPOLÍTICAS CONGELADAS AVALIADAS")
    for name, evaluation in result["evaluations"].items():
        print(f"  {name}: {result['frozen_policies'][name]}")
        for split in ("known", "unseen"):
            item = evaluation[split]
            print(
                f"    {split:<7} full={item['full_cases']}/{item['cases']} "
                f"deps_perdidas={item['dependencies_lost']} violações={item['violations']} "
                f"sem_função={item['functionless']} tokens={item['tokens']} "
                f"candidatos={item['candidates']} buscas_extras={item['extra_searches']}"
            )
    print("\nMUDANÇAS POR CASO (vs referência)")
    for name in ("random", "genetic"):
        changes = []
        for ref, other in zip(
            result["evaluations"]["reference"]["known"]["details"],
            result["evaluations"][name]["known"]["details"],
        ):
            if ref["full"] != other["full"] or ref["tokens"] != other["tokens"]:
                changes.append(
                    f"{ref['case']}: full {ref['full']}->{other['full']} "
                    f"tokens {ref['tokens']}->{other['tokens']}"
                )
        for ref, other in zip(
            result["evaluations"]["reference"]["unseen"]["details"],
            result["evaluations"][name]["unseen"]["details"],
        ):
            if ref["full"] != other["full"] or ref["tokens"] != other["tokens"]:
                changes.append(
                    f"[inédito] {ref['case']}: full {ref['full']}->{other['full']} "
                    f"tokens {ref['tokens']}->{other['tokens']}"
                )
        print(f"  {name}: {changes or '(nenhuma diferença)'}")
    print(f"\nArquivo: {output / 'genetic_results.json'}")
    return 0


def cmd_policy_answer(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_policy_answer(config, settings, client)
    paths = export_policy_answer(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("TRÊS POLÍTICAS CONGELADAS — QUALIDADE DE RESPOSTA")
    print(
        f"MODO: {'DeepSeek real' if args.deepseek else 'SIMULADO'} | "
        f"casos={result['cases']} | chamadas={result['calls']}"
    )
    print("política   fatos  item  sem_suporte  sensível  ambiguidade  nota  mem_tokens  prompt  completion")
    for name in ("reference", "random", "genetic"):
        item = result["summary"][name]
        print(
            f"{name:<10} {item['facts']:>5} {item['item']:>5} {item['unsupported']:>11} "
            f"{item['sensitive']:>9} {item['ambiguity']:>11} {item['avg_score']:>5.2f} "
            f"{item['avg_memory_tokens']:>10.1f} {item['prompt_tokens']:>7} {item['completion_tokens']:>11}"
        )
    print("\nCOMPARAÇÕES COM A REFERÊNCIA")
    for name in ("random", "genetic"):
        comp = result["comparisons"][name]
        print(
            f"  {name}: melhorou={comp['better_than_reference']} "
            f"empatou={comp['tied_with_reference']} piorou={comp['worse_than_reference']}"
        )
    print("\nPolíticas:")
    for name, policy in result["policies"].items():
        print(f"  {name}: {policy}")
    print("Arquivos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_purpose_answer(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_purpose_answer(config, settings, client)
    paths = export_purpose_answer(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("RESPOSTAS POR PROPÓSITO — REFERÊNCIA vs PROPÓSITO")
    print(
        f"MODO: {'DeepSeek real' if args.deepseek else 'SIMULADO'} | "
        f"casos={result['cases']} | chamadas={result['calls']} | "
        f"subconjunto de remoção={result['removal_subset']}"
    )
    print("condição   casos  todos_ok  propósitos_ok  deps_perdidas  prioridade_ok  sem_suporte  sensível  nota  mem  prompt  completion")
    for name in ("reference", "purpose", "removal"):
        item = result["conditions"][name]
        print(
            f"{name:<10} {item['cases']:>5} {item['all_ok']:>9} "
            f"{item['purposes_ok']}/{item['purposes_total']:<11} "
            f"{item['dependencies_lost']:>13} {item['priority_ok']:>13} "
            f"{item['unsupported']:>12} {item['sensitive']:>8} "
            f"{item['avg_score']:>5.2f} {item['avg_memory_tokens']:>4.0f} "
            f"{item['prompt_tokens']:>7} {item['completion_tokens']:>11}"
        )
    print("\nCOMPARAÇÕES COM A REFERÊNCIA")
    for name in ("purpose", "removal"):
        pair = result["pairs"][name]
        print(
            f"  {name}: melhorou={pair['better_than_reference']} "
            f"empatou={pair['tied_with_reference']} piorou={pair['worse_than_reference']}"
        )
    print("\nPOR CASO")
    for entry in result["entries"]:
        ref = entry["conditions"]["reference"]
        purpose = entry["conditions"]["purpose"]
        print(
            f"  {entry['case']:<20} ref={ref['score']:.2f} propósito={purpose['score']:.2f} "
            f"ids={purpose['selected']} removed={purpose['removed'] or '-'}"
        )
        if "removal" in entry["conditions"]:
            removal = entry["conditions"]["removal"]
            print(
                f"    remoção: nota={removal['score']:.2f} "
                f"removidos={removal['removed']} partes={removal['checks']['purposes']}"
            )
    print("\nArquivos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_attribution_round(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_attribution_round(config, settings, client)
    paths = export_attribution_review(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("AVALIAÇÃO DE ATRIBUIÇÃO")
    print(
        f"MODO: {'DeepSeek real' if args.deepseek else 'SIMULADO'} | "
        f"casos={result['cases']} | chamadas_executadas={result['calls']}"
    )
    for entry in result["entries"]:
        ref = entry["conditions"]["reference"]
        con = entry["conditions"]["conservative"]
        forced = entry["conditions"]["forced"]
        print("\n" + "=" * 78)
        print(f"CASO {entry['case']} ({entry['kind']}) | pai={entry['parent']}")
        print(
            f"  completo ids={entry['reference_ids']} "
            f"tokens={ref['usage'].get('prompt_tokens')}/{ref['usage'].get('completion_tokens')} "
            f"score={ref['checks']['score']:.2f}"
        )
        print(
            f"  conservador ids={entry['conservative_ids']} omitiu={entry['conservative_verdict']} "
            f"tokens={con['usage'].get('prompt_tokens')}/{con['usage'].get('completion_tokens')} "
            f"score={con['checks']['score']:.2f}"
        )
        print(
            f"  forçado ids={entry['forced_ids']} omitiu={entry['forced_verdict']} "
            f"tokens={forced['usage'].get('prompt_tokens')}/{forced['usage'].get('completion_tokens')} "
            f"score={forced['checks']['score']:.2f} sem_pista_equivalente={entry['no_equivalent_clue']}"
        )
        print(
            f"  conservador: fato={con['checks']['fact_ok']} entidade={con['checks']['entity_ok']} "
            f"decisão={con['checks']['decision_ok']} ambiguidade={con['checks']['ambiguity_ok']}"
        )
    print("\nRevisão cega:", paths["review"])
    print("23 casos anteriores preservados como pendentes.")
    return 0


def cmd_dependency_round(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_dependency_round(config, settings, client)
    paths = export_dependency_review(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("DEPENDÊNCIAS CONDICIONAIS")
    print(
        f"MODO: {'DeepSeek real' if args.deepseek else 'SIMULADO'} | "
        f"casos={result['cases']} | chamadas_executadas={result['calls']} | "
        f"comparações={result['comparisons']}"
    )
    for entry in result["entries"]:
        ref = entry["conditions"]["reference"]
        con = entry["conditions"]["conservative"]
        forced = entry["conditions"]["forced"]
        print("\n" + "=" * 78)
        print(f"CASO {entry['case']} ({entry['group']}) | dependência={entry['planned_dependency']}")
        print(
            f"  referência ids={entry['reference_ids']} "
            f"tokens={ref['usage'].get('prompt_tokens')}/{ref['usage'].get('completion_tokens')} "
            f"nota={ref['checks']['score']:.2f}"
        )
        print(
            f"  conservadora ids={entry['conservative_ids']} omitiu={entry['conservative_omitted']} "
            f"tokens={con['usage'].get('prompt_tokens')}/{con['usage'].get('completion_tokens')} "
            f"nota={con['checks']['score']:.2f} veredito={entry['verdict']}"
        )
        print(
            f"  forçada ids={entry['forced_ids']} omitiu={entry['forced_omitted']} "
            f"tokens={forced['usage'].get('prompt_tokens')}/{forced['usage'].get('completion_tokens')} "
            f"nota={forced['checks']['score']:.2f}"
        )
        print(
            f"  partes conservadora={con['checks']['purposes']} "
            f"decisão={con['checks']['decision_ok']} ambiguidade={con['checks']['ambiguity_ok']} "
            f"preservação={con['checks']['preserve_ok']}"
        )
    print("\nRevisão cega:", paths["review"])
    print("17 casos anteriores preservados como pendentes.")
    return 0


def cmd_removal_round(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_removal_round(config, settings, client)
    paths = export_removal_round(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("RODADA DE REMOÇÕES DIRIGIDAS")
    print(
        f"MODO: {'DeepSeek real' if args.deepseek else 'SIMULADO'} | "
        f"casos={result['cases']} | teto_previsto={result['planned_calls']} | "
        f"chamadas_executadas={result['executed_calls']}"
    )
    for entry in result["entries"]:
        print("\n" + "=" * 78)
        print(f"CASO {entry['case']}")
        purpose = entry["conditions"]["purpose"]
        reference = entry["conditions"]["reference"]
        print(
            f"  referência ids={reference['selected']} "
            f"tokens={reference['usage'].get('prompt_tokens')}/"
            f"{reference['usage'].get('completion_tokens')}"
        )
        print(
            f"  propósito ids={purpose['selected']} "
            f"tokens={purpose['usage'].get('prompt_tokens')}/"
            f"{purpose['usage'].get('completion_tokens')} "
            f"partes={purpose['checks']['purposes']}"
        )
        for comparison in entry["comparisons"]:
            removal = comparison["result"]
            print(
                f"  REMOÇÃO {comparison['item']} ({comparison['role']}): "
                f"presente={comparison['present_in_purpose']} "
                f"mesma_requisição={comparison['same_request']} "
                f"veredito={comparison['verdict']} "
                f"ids={removal['selected']} "
                f"tokens={removal['usage'].get('prompt_tokens')}/"
                f"{removal['usage'].get('completion_tokens')} "
                f"partes={removal['checks']['purposes']}"
            )
    print(
        "\nRevisão humana: pendente | pares cegos anteriores preservados | "
        f"arquivo={paths['review']}"
    )
    return 0


def cmd_purpose_report(args: argparse.Namespace) -> int:
    summary = reprocess(args.results)
    review_info = export_human_review(args.review, args.human_review_out)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "purpose_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps({"summary": summary, "human_review": review_info}, ensure_ascii=False, indent=2))
        return 0
    print("REPROCESSAMENTO — AVALIAÇÃO POR PROPÓSITO")
    print("Sem novas chamadas; dados reais já salvos.")
    print(
        "condição   casos  rubrica_ok  todos_propósitos  propósitos  prompt  completion"
    )
    for name in ("reference", "purpose", "removal"):
        item = summary["overall"][name]
        print(
            f"{name:<10} {item['cases']:>5} {item['rubric_approved']:>10} "
            f"{item['all_purposes']:>16} {item['purposes_ok']}/{item['purposes_total']:<10} "
            f"{item['prompt']:>7} {item['completion']:>11}"
        )
    print("\nGRUPOS")
    for group in ("regression", "new"):
        print(f"  {group}:")
        for condition, item in summary["groups"][group].items():
            if condition == "paired":
                continue
            print(
                f"    {condition:<10} rubrica={item['rubric_approved']}/{item['cases']} "
                f"todos={item['all_purposes']}/{item['cases']} "
                f"propósitos={item['purposes_ok']}/{item['purposes_total']} "
                f"Δprompt={item['delta_prompt_vs_reference']:+d} "
                f"Δcompletion={item['delta_completion_vs_reference']:+d}"
            )
        print(f"    pareado={summary['groups'][group].get('paired')}")
    priority = summary.get("priority_case")
    if priority:
        purpose = priority["conditions"]["purpose"]
        print("\nnew_priority:")
        print(
            f"  rubrica_aprovada={purpose['checks']['all_ok']} "
            f"todos_propósitos={all(v['ok'] for v in purpose['checks']['purposes'].values())} "
            f"prioridade_ok={purpose['checks']['priority_ok']} "
            f"partes={purpose['checks']['purposes']}"
        )
    print("\nOMISSÕES CLASSIFICADAS:")
    for omission in summary["omissions"]:
        print(f"  {omission}")
    print("PARTES OMITIDAS CORRETAMENTE:")
    priority_omission = [
        omission
        for omission in summary["omissions"]
        if omission["reason"] == "prioridade_explicita"
    ]
    for omission in priority_omission:
        print(f"  prioridade: {omission}")
    for omission in summary.get("accepted_unavailable", []):
        if omission["condition"] == "purpose":
            print(f"  restrição: {omission}")
    if not priority_omission and not summary.get("accepted_unavailable"):
        print("  (nenhuma)")
    print("\nREMOÇÕES:")
    for entry in summary.get("entries", []):
        if "removal" in entry["conditions"]:
            removal = entry["conditions"]["removal"]
            print(
                f"  {entry['case']}: removidos={removal['removed']} "
                f"partes={removal['checks']['purposes']} "
                f"tokens={removal['usage'].get('prompt_tokens')}/"
                f"{removal['usage'].get('completion_tokens')}"
            )
    print(
        f"\nRevisão humana: pares={review_info['pairs']} "
        f"pendentes={review_info['pending']} | arquivo={args.human_review_out}"
    )
    print("\nPRÓXIMA RODADA DIRIGIDA")
    print("  plano=benchmark/removal_plan_v2.json")
    print("  casos=5 | condições=4 | chamadas reais previstas=20")
    print(f"Arquivo: {output / 'purpose_report.json'}")
    return 0


def cmd_purpose_benchmark(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_purpose_benchmark(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "purpose_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("SELEÇÃO POR TEMA E PROPÓSITO")
    print(
        f"treino={result['train_cases']} | avaliação={result['eval_cases']} | "
        "mesmo orçamento e renderizador"
    )
    print(
        "condição   cobertura  turnos_completos  deps_perdidas  distrações  tokens"
    )
    for key, label in (("reference", "referência"), ("purpose", "propósito")):
        item = result[key]
        print(
            f"{label:<10} {item['purpose_coverage']:>9.2f} "
            f"{item['all_purposes_cases']}/{item['cases']:<14} "
            f"{item['deps_lost']:>13} {item['distractions']:>10} {item['memory_tokens']:>7}"
        )
    print("\nPOR CASO")
    for entry in result["entries"]:
        print(
            f"  {entry['case']:<18} referência={entry['reference']['purposes']} "
            f"seleção={entry['reference']['selected']} | "
            f"propósito={entry['purpose']['purposes']} seleção={entry['purpose']['selected']}"
        )
    print(f"\nMelhoraram: {result['improvements'] or '-'} | "
          f"Regrediram: {result['regressions'] or '-'}")
    print(f"\nArquivo: {output / 'purpose_results.json'}")
    return 0


def cmd_paired_answer(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2
    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    result = run_paired(config, settings, client)
    paths = export_paired(result, Path(args.out))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("PARES ATUAL vs REDUZIDO — QUALIDADE DE RESPOSTA")
    if args.deepseek:
        print(
            f"MODO: DeepSeek real | modelo={settings.model} | casos={result['cases']} | "
            f"chamadas={result['calls']}"
        )
    else:
        print(
            "MODO: SIMULADO — valida o mecanismo; não representa o DeepSeek."
        )
        print(f"casos={result['cases']} | chamadas simuladas={result['calls']}")
    print(
        f"condição   fatos  item  sem_suporte  sensível  ambiguidade  nota  mem_tokens  req_estimados  tokens_api"
    )
    for key, label in (("current", "atual"), ("reduced", "reduzido")):
        item = result["conditions"][key]
        print(
            f"{label:<10} {item['facts']:>5} {item['item']:>5} {item['unsupported']:>11} "
            f"{item['sensitive']:>9} {item['ambiguity']:>11} {item['avg_score']:>5.2f} "
            f"{item['avg_memory_tokens']:>10.1f} {item['avg_request_tokens']:>14.1f} "
            f"{item['api_prompt_tokens']:>10}"
        )
    print(
        f"\nPares: melhorou={result['counts']['melhorou']} "
        f"empatou={result['counts']['empatou']} piorou={result['counts']['piorou']}"
    )
    for entry in result["entries"]:
        current = entry["current"]
        reduced = entry["reduced"]
        print(
            f"  {entry['case']:<20} {entry['verdict']:<8} "
            f"nota {current['score']:.2f}->{reduced['score']:.2f} | "
            f"mem {current['memory_tokens']}->{reduced['memory_tokens']} | "
            f"removidos={reduced['removed']} | "
            f"fonte={current['token_source']}/{reduced['token_source']}"
        )
        if entry["verdict"] == "piorou":
            print(
                f"    PIOROU: respostas atuais/reduzidas acima; "
                f"IDs removidos={reduced['removed']}"
            )
    print("\nArquivos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_context_reduction(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_context_reduction(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "context_reduction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("REDUÇÃO DE CONTEXTO — SELEÇÃO DE MEMÓRIA")
    print("Referência: mais_treino congelado; busca/ação/sensibilidade/orçamento iguais.")
    print(
        "condição   evidências  dependências  sem_função  tokens_mem  tokens_totais  casos"
    )
    for key, label in (("current", "atual"), ("reduced", "reduzida")):
        item = result[key]
        print(
            f"{label:<9} {item['expected']:>9} {item['context']:>13} "
            f"{item['functionless']:>11} {item['memory_tokens']:>11} "
            f"{item['tokens']:>13} {item['cases']:>6}"
        )
    print("\nPOR CASO")
    for entry in result["entries"]:
        current = entry["current"]
        reduced = entry["reduced"]
        print(
            f"  {entry['case']:<20} ação={entry['action']:<12} "
            f"atual={current['selected']} -> reduzida={reduced['selected']} | "
            f"removidos={reduced['removed']} | "
            f"esperadas {len(current['expected'])}->{len(reduced['expected'])} | "
            f"contexto {len(current['context'])}->{len(reduced['context'])} | "
            f"sem_função {len(current['functionless'])}->{len(reduced['functionless'])} | "
            f"mem_tokens {current['memory_tokens']}->{reduced['memory_tokens']}"
        )
    print("\nCASOS COM PERDA DE EVIDÊNCIA OU CONTEXTO:")
    if result["tradeoffs"]:
        for case_id in result["tradeoffs"]:
            print(f"  {case_id}")
    else:
        print("  (nenhum)")
    print(f"\nArquivo: {output / 'context_reduction.json'}")
    return 0


def cmd_curiosity_round3(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_round3(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "curiosity_round3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("CURIOSIDADE APRENDIDA — RODADA 3")
    print(
        f"treino={result['train_cases']} | avaliação={result['eval_cases']} | "
        f"índice: build={result['index_avg_build_ms']:.4f} ms, "
        f"consulta={result['index_avg_query_ms']:.4f} ms, "
        f"entradas={result['index_avg_entries']}"
    )
    diag = result["diag_piano"]
    print("\nREPRODUÇÃO ev_follow_piano")
    print(f"  sinais antigos: {diag['features_old']}")
    print(f"  sinal compatível: {diag['features_compatible']}")
    print(f"  ações elegíveis: {diag['eligible_actions']}")
    print(f"  scores mais_treino: {diag['scores_more']}")
    print(f"  scores compatível: {diag['scores_compatible']}")
    print("  IDs por ação, do mesmo estado inicial:")
    for action, values in diag["all_actions"].items():
        print(
            f"    {action:<12} esperadas={values['expected']} "
            f"contexto={values['context']} sem_função={values['functionless']} "
            f"extras={values['extras']} tokens_mem={values['memory_tokens']}"
        )
    print(
        "  Contradição resolvida: follow_links recupera m6; other_theme não "
        "recupera nada. O texto anterior estava errado: este caso é uma "
        "resposta curta vinculada, não um caso de outro tema."
    )

    print("\nRESUMO")
    print(
        "condição    esperadas  contexto  sem_função  sem_ganho/extras  consultas  "
        "candidatos  tokens_mem  tokens  lat_med/p95"
    )
    for key, label in (("off", "off"), ("fixed", "fixa"), ("more", "mais_treino"), ("compatible", "compatível")):
        item = result["summary"][key]
        print(
            f"{label:<11} {item['expected']:>9} {item['context']:>9} "
            f"{item['functionless']:>11} "
            f"{item['no_gain_searches']}/{item['extra_searches']:<16} "
            f"{item['queries_triggered']}/{item['cases']:<8} {item['candidates']:>10} "
            f"{item['memory_tokens']:>10} {item['tokens']:>8} "
            f"{item['latency_median_ms']:>7.2f}/{item['latency_p95_ms']:<7.2f}"
        )
    print("\nEVIDÊNCIAS RECUPERADAS POR FAMÍLIA")
    print("família          off  fixa  mais_treino  compatível")
    for family, values in result["families"].items():
        print(
            f"{family:<16} {values['off']:>3} {values['fixed']:>5} "
            f"{values['more']:>12} {values['compatible']:>11}"
        )
    print("\nMUDANÇAS DE AÇÃO (compatível vs mais_treino)")
    for entry in result["entries"]:
        action_more = entry["more"]["action"]
        action_compatible = entry["compatible"]["action"]
        if action_more != action_compatible:
            effect = (
                "melhorou"
                if len(entry["compatible"]["expected"]) > len(entry["more"]["expected"])
                else "piorou"
                if len(entry["compatible"]["expected"]) < len(entry["more"]["expected"])
                else "igual"
            )
            print(
                f"  {entry['case']}: mais_treino={action_more} "
                f"compatível={action_compatible} -> {effect} | "
                f"esperadas {entry['more']['expected']} -> {entry['compatible']['expected']}"
            )
    print(
        f"\nRegressões: ev_follow={result['entries'][0]['compatible']['action']} "
        f"esperadas={result['entries'][0]['compatible']['expected']}; "
        f"ev_correction={result['entries'][2]['compatible']['action']} "
        f"esperadas={result['entries'][2]['compatible']['expected']}"
    )
    print(f"Arquivo: {output / 'curiosity_round3.json'}")
    return 0


def cmd_curiosity_round2(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_round2(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "curiosity_round2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("CURIOSIDADE APRENDIDA — RODADA 2")
    print(
        f"treino base={result['train_base']} casos | treino estendido={result['train_extended']} | "
        f"avaliação={result['eval_cases']} famílias novas | "
        f"custo do sinal de vínculo={result['signal_avg_cost_ms']:.3f} ms/caso"
    )
    diag = result["diag_follow"]
    print("\nREPRODUÇÃO ev_follow")
    print(f"  sinais antigos: {diag['features_old']}")
    print(f"  sinal novo: {diag['features_link']}")
    print(f"  ações elegíveis: {diag['eligible_actions']}")
    print("  probabilidades do modelo anterior:", diag["scores_old"])
    print("  probabilidades com força de vínculo:", diag["scores_link"])
    print("  resultados de cada ação a partir do mesmo estado:")
    for action, values in diag["all_actions"].items():
        print(
            f"    {action:<12} esperadas={values['expected']} "
            f"contexto={values['context']} sem_função={values['functionless']} "
            f"extras={values['extras']} tokens_mem={values['memory_tokens']} "
            f"total={values['tokens']}"
        )

    print("\nRESUMO (categorias de inclusão)")
    print(
        "condição  esperadas  contexto  sem_função  buscas_sem_ganho  consultas_acionadas  "
        "candidatos  tokens_mem  tokens  lat_med/p95"
    )
    for key, label in (("off", "off"), ("fixed", "fixa"), ("old", "anterior"), ("more", "mais_treino"), ("link", "vinculo")):
        item = result["summary"][key]
        print(
            f"{label:<9} {item['expected']:>9} {item['context']:>9} "
            f"{item['functionless']:>11} "
            f"{item['no_gain_searches']}/{item['extra_searches']:<11} "
            f"{item['queries_triggered']}/{item['cases']:<12} {item['candidates']:>10} "
            f"{item['memory_tokens']:>10} {item['tokens']:>8} "
            f"{item['latency_median_ms']:>7.2f}/{item['latency_p95_ms']:<7.2f}"
        )
    print("\nEVIDÊNCIAS RECUPERADAS POR FAMÍLIA")
    print("família        off  fixa  anterior  mais_treino  vínculo")
    for family, values in result["families"].items():
        print(
            f"{family:<14} {values['off']:>3} {values['fixed']:>5} {values['old']:>9} "
            f"{values['more']:>11} {values['link']:>8}"
        )
    print("\nESCOLHAS QUE MUDARAM")
    for entry in result["entries"]:
        old_action = entry["old"]["action"]
        link_action = entry["link"]["action"]
        if old_action != link_action or entry["fixed"]["action"] != link_action:
            print(
                f"  {entry['case']}: fixa={entry['fixed']['action']} "
                f"anterior={old_action} vínculo={link_action} | "
                f"esperadas fixa={entry['fixed']['expected']} "
                f"anterior={entry['old']['expected']} vínculo={entry['link']['expected']}"
            )
    print(f"\nArquivo: {output / 'curiosity_round2.json'}")
    return 0


def cmd_curiosity_learn(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_curiosity_experiment(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "curiosity_learning_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("CURIOSIDADE APRENDIDA")
    print(
        f"treino={result['train_cases']} casos ({result['training_rows']} linhas) | "
        f"avaliação={result['eval_cases']} casos"
    )
    print("PESOS POR AÇÃO (8 sinais: ambiguidade, faltante, conflito, vínculos, candidatos, atributo disponível, atributo pedido, restrição):")
    for index, action in enumerate(("stop", "follow_links", "other_theme", "correction")):
        weights = ", ".join(f"{value:+.3f}" for value in result["model"]["weights"][index])
        print(f"  {action:<12} [{weights}] bias={result['model']['biases'][index]:+.3f}")
    print("\ncondição   evidências  buscas_sem_ganho  irrelevantes  extras  latência_mediana  tokens")
    for key, label in (("off", "desligada"), ("fixed_rules", "fixa"), ("learned", "aprendida")):
        item = result[key]
        print(
            f"{label:<10} {item['full']:>10} {item['no_gain_searches']:>17} "
            f"{item['irrelevant']:>12} {item['extras']:>7} "
            f"{item['latency_median_ms']:>16.2f} {item['avg_tokens']:>7.1f}"
        )
    print("\ncasos em que a ação aprendida mudou:")
    for entry in result["evaluation"]:
        if entry["learned_action"] != entry["fixed_action"]:
            verdict = "melhorou" if entry["improved"] else "piorou" if entry["worsened"] else "igual"
            print(
                f"  {entry['case']}: fixa={entry['fixed_action']} "
                f"aprendida={entry['learned_action']} -> {verdict} "
                f"(full fixa={entry['fixed']['full']} aprendida={entry['learned']['full']})"
            )
    print(f"\nArquivo: {output / 'curiosity_learning_results.json'}")
    return 0


def cmd_learn_benchmark(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = run_experiment(config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "learning_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("APRENDIZADO SUPERVISIONADO DE PESOS")
    print(
        f"treino={result['train_cases']} casos ({', '.join(result['train_families'])}) | "
        f"avaliação={result['eval_cases']} casos ({', '.join(result['eval_families'])})"
    )
    model = result["model"]
    print(
        f"exemplos={model['examples']} | positivos={model['positives']} | "
        f"bias={model['bias']}"
    )
    print("PESOS APRENDIDOS (congelados antes da avaliação):")
    for name, weight in model["weights"].items():
        print(f"  {name:<18} {weight:+.6f}")
    print("\ncondição   full       irrelevantes  tokens_adicionais  sem_candidato")
    for side in ("fixed", "learned"):
        item = result[side]
        print(
            f"{side:<10} {item['full']}/{item['cases']:<7} "
            f"{item['irrelevant']:>10} {item['avg_additional_tokens']:>16.2f} "
            f"{item['missing_candidates']:>14}"
        )
    print(
        f"\nconsultas que mudaram: {result['changed'] or '-'} | "
        f"melhoraram: {result['improved'] or '-'} | "
        f"regrediram: {result['regressed'] or '-'}"
    )
    for entry in result["evaluation"]:
        print(
            f"  {entry['case']:<20} fixo={entry['fixed']['selected']} "
            f"aprendido={entry['learned']['selected']} "
            f"full fixo={entry['fixed']['full']} aprendido={entry['learned']['full']}"
        )
    print(f"\nArquivo: {output / 'learning_results.json'}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2

    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2

    results = run_benchmark(cases, config, settings, client)
    summary = summarize(results)
    paths = export(results, summary, Path(args.out))

    if args.json:
        print(
            json.dumps(
                {
                    "mode": "deepseek" if args.deepseek else "simulado",
                    "summary": summary,
                    "cases": [result.as_dict() for result in results],
                    "files": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("BENCHMARK SINAPSE")
    if args.deepseek:
        print(
            f"MODO: DeepSeek real | modelo={settings.model} | "
            f"casos={len(results)} | chamadas={len(results) * 4}"
        )
    else:
        print(
            "MODO: cliente SIMULADO — valida o mecanismo de avaliação, "
            "não representa o DeepSeek."
        )
        print(
            f"modelo configurado={settings.model} | casos={len(results)} | "
            f"chamadas simuladas={len(results) * 4}"
        )
    print(
        f"reserva_saída={settings.output_reserve} | "
        f"limite de memória por caso=variável (120–170 tokens)"
    )

    for result in results:
        case = result.case
        print("\n" + "=" * 78)
        print(f"CASO {case.id} ({case.kind}) | {case.question!r}")
        print(
            f"  esperados={list(case.expected_ids) or '-'} | "
            f"proibidos={list(case.forbidden_ids) or '-'} | "
            f"limite_memória={case.memory_token_limit}"
        )
        for condition in ("history", "lexical", "sinapse", "manual"):
            outcome = result.outcomes[condition]
            print(
                f"  {condition:<8} ids={str(list(outcome.selected_ids) or '-'):<18} "
                f"tokens={outcome.additional_tokens:>3} "
                f"hit={'S' if outcome.hit_any else 'n'} "
                f"full={'S' if outcome.full_coverage else 'n'} "
                f"viol={len(outcome.restricted_violations)} "
                f"rep={'S' if outcome.repetition else 'n'} "
                f"irr={'S' if outcome.irrelevant_attached else 'n'} "
                f"nota={outcome.answer_score:.2f}"
            )
            if outcome.error:
                print(f"    erro: {outcome.error[:160]}")
        scores = {c: result.outcomes[c].answer_score for c in summary["conditions"]}
        better = [
            baseline
            for baseline in ("history", "lexical", "manual")
            if scores["sinapse"] > scores[baseline]
        ]
        worse = [
            baseline
            for baseline in ("history", "lexical", "manual")
            if scores["sinapse"] < scores[baseline]
        ]
        print(
            f"  Sinapse melhor que: {better or '-'} | pior que: {worse or '-'}"
        )

    print("\n" + "=" * 78)
    print("RESUMO")
    print("condição         hit (n/den)  full (n/den)  tokens  violações  repet.  irrelev.  resposta")
    for condition in ("history", "lexical", "sinapse", "manual"):
        item = summary["conditions"][condition]
        hit_label = f"{item['hit_count']}/{item['hit_denominator']}={item['hit_rate']:.2f}"
        full_label = f"{item['full_count']}/{item['hit_denominator']}={item['full_rate']:.2f}"
        print(
            f"{condition:<15} {hit_label:<12} {full_label:<12} "
            f"{item['avg_additional_tokens']:>7.1f} {item['restricted_violations']:>9} "
            f"{item['repetitions']:>7} {item['irrelevant_attachments']:>9} "
            f"{item['avg_answer_score']:>8.2f}"
        )
    print("\nDelta Sinapse (pontos percentuais):")
    for baseline, delta in summary["sinapse_delta_pp"].items():
        print(f"  vs {baseline}: {delta:+.2f} pp")
    print("\nArquivos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_answer_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    config = load_config(Path(args.config))
    try:
        settings = load_deepseek_settings(config)
        if args.model:
            settings = replace(settings, model=args.model)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2

    client = None
    if args.deepseek:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2
    results = run_answer_benchmark(cases, config, settings, client)
    summary = summarize_answers(results)
    paths = export_answers(results, summary, Path(args.out))

    if args.json:
        print(
            json.dumps(
                {
                    "mode": "deepseek" if args.deepseek else "simulado",
                    "summary": summary,
                    "cases": [result.as_dict() for result in results],
                    "files": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print("QUALIDADE DE RESPOSTA POR CONDIÇÃO")
    if args.deepseek:
        print(
            f"MODO: DeepSeek real | modelo={settings.model} | casos={len(results)} | "
            f"chamadas={len(results) * 4}"
        )
    else:
        print(
            "MODO: SIMULADO — valida o mecanismo; não representa o DeepSeek."
        )
        print(
            f"modelo configurado={settings.model} | casos={len(results)} | "
            f"chamadas simuladas={len(results) * 4}"
        )
    print(
        f"instruções/histórico/pergunta iguais | reserva={settings.output_reserve} | "
        f"temperatura={settings.temperature} | teto de memória por caso"
    )
    for result in results:
        print("\n" + "=" * 78)
        print(f"CASO {result.case.id} ({result.case.kind}) | {result.case.question!r}")
        print(
            f"  esperados={list(result.case.expected_ids) or '-'} | "
            f"distratores={list(result.case.distractor_ids) or '-'}"
        )
        for condition in ("history", "distractors", "sinapse", "manual"):
            outcome = result.outcomes[condition]
            print(
                f"  {condition:<11} ids={str(list(outcome.selected_ids) or '-'):<18} "
                f"tokens={outcome.additional_tokens:>3} "
                f"enviada={'S' if outcome.evidence_sent else 'n'} "
                f"usada={'S' if outcome.evidence_used else 'n'} "
                f"nota={outcome.score:.2f}"
            )
            if outcome.error:
                print(f"    erro: {outcome.error[:160]}")
    print("\n" + "=" * 78)
    print("RESUMO")
    print("condição      evid_enviada  evid_usada  nota  tokens_adicionais")
    for condition in ("history", "distractors", "sinapse", "manual"):
        item = summary[condition]
        print(
            f"{condition:<12} {item['evidence_sent']}/{item['cases']:<10} "
            f"{item['evidence_used']}/{item['cases']:<9} "
            f"{item['avg_score']:>5.2f} {item['avg_additional_tokens']:>10.1f}"
        )
    comp = summary["comparisons"]
    print(
        f"\ncasos em que trocar a memória mudou a nota: {comp['changed_cases']}"
    )
    print(
        f"Sinapse melhor que histórico: {comp['sinapse_better_than_history']} | "
        f"pior: {comp['sinapse_worse_than_history']} | "
        f"distratores piores que histórico: {comp['distractors_worse_than_history']}"
    )
    print(f"melhoraram com Sinapse: {comp['improved'] or '-'}")
    print(f"pioraram com distratores: {comp['worsened_by_distractors'] or '-'}")
    print(
        "acertaram a evidência mas erraram a resposta: "
        f"{comp['correct_evidence_but_error'] or '-'}"
    )
    print("\nArquivos:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    messages, config = _load(args)
    try:
        settings = load_deepseek_settings(config)
        overrides: dict[str, Any] = {}
        if args.model:
            overrides["model"] = args.model
        if args.base_url:
            overrides["base_url"] = args.base_url
        if args.window is not None:
            overrides["context_window"] = args.window
        if args.output_reserve is not None:
            overrides["output_reserve"] = args.output_reserve
        if args.max_tokens is not None:
            overrides["max_tokens"] = args.max_tokens
        if args.thinking:
            overrides["thinking"] = args.thinking
        if overrides:
            settings = replace(settings, **overrides)
        settings.validate()
    except ValueError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        return 2

    if args.simulate:
        client = SimulatedDeepSeekClient(reply=args.simulate_reply)
    else:
        try:
            client = DeepSeekClient(base_url=settings.base_url)
        except MissingApiKey as error:
            print(str(error), file=sys.stderr)
            return 2

    chatbot = SinapseChatbot(messages, config, client, settings=settings)
    try:
        turn = chatbot.ask(
            args.query,
            args.context or [],
            window=args.window,
            output_reserve=args.output_reserve,
            seed_id=args.seed,
            focus=args.focus,
            radius=args.radius,
            top_k=args.top_k,
            threshold=args.threshold,
            include_seed=not args.no_seed,
            base_message_ids=args.base_message_id or (),
        )
    except Exception as error:  # erro da API já vem tipado; aqui só reportamos
        print(f"Falha ao executar o chat: {error}", file=sys.stderr)
        return 3

    if args.json:
        payload = {
            "settings": {**asdict(settings), "model_version": settings.model_version},
            "transport": {"name": client.name, "simulated": bool(client.simulated)},
            "turn": turn.as_dict(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    d = turn.budget.diagnostics
    print("INTEGRAÇÃO DEEPSEEK")
    print(f"  modelo={settings.model} ({settings.model_version})")
    print(f"  base_url={settings.base_url}")
    print(
        f"  janela_contexto={settings.context_window} | reserva_saída={settings.output_reserve} "
        f"| max_tokens={settings.max_tokens} | thinking={settings.thinking}"
    )
    print(
        f"  transporte={client.name} (simulado={str(bool(client.simulated)).lower()}) | "
        f"contagem_estimada={str(bool(d.counter_simulated)).lower()}"
    )
    print(
        f"  contador={d.counter_name} | formato={d.context_format}"
    )
    print("ORÇAMENTO SOBRE O PAYLOAD DE CHAT:")
    print(
        f"  base={d.base_tokens} | capacidade_entrada={d.capacity} | "
        f"orçamento_memória={d.memory_budget}"
    )
    print(
        f"  memória_usada={d.memory_tokens} | entrada_usada={d.used_tokens} | "
        f"saldo={d.remaining_tokens} | entrada+reserva={d.total_with_reserve}/{d.window}"
    )
    print(f"  STATUS: {d.status}")
    print(
        f"  SEMENTE: incluída={str(d.seed_included).lower()} | origem={d.seed_source} | "
        f"custo={d.seed_tokens} tokens | já_na_base={str(d.seed_in_base).lower()}"
    )
    print(
        "  EVIDÊNCIAS SELECIONADAS: "
        + (", ".join(d.selected_ids) if d.selected_ids else "(nenhuma)")
    )
    print("  DECISÕES POR CANDIDATO:")
    for decision in d.decisions:
        rank = f" #{decision.rank}" if decision.rank else ""
        detail = f" — {decision.detail}" if decision.detail else ""
        print(
            f"    {decision.id:<4} dist={decision.distance:5.3f} "
            f"atenção={decision.attention:5.3f} tokens={decision.tokens} "
            f"motivo={decision.reason}{rank}{detail}"
        )
    print("REQUISIÇÃO FINAL (mensagens enviadas):")
    for index, message in enumerate(turn.request_body["messages"], start=1):
        content = str(message["content"])
        if len(content) > 600:
            content = content[:600] + " [... truncado só na exibição]"
        print(f"  [{index}] {message['role']}: {content}")
    print("RESPOSTA RECEBIDA:")
    print(turn.response.content or "(vazia)")
    if turn.response.usage:
        usage = turn.response.usage
        print(
            "USO INFORMADO PELA API: "
            f"prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_tokens={usage.get('total_tokens')}"
        )
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            print(
                f"COMPARAÇÃO: estimativa local={d.used_tokens} tokens | "
                f"real informado={prompt_tokens} tokens | "
                f"diferença={prompt_tokens - d.used_tokens:+d}"
            )
    else:
        print("USO: não informado (transporte simulado)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sinapse", description="Memória inspecionável para chatbots.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="JSON com as mensagens")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON de configuração")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Executa uma consulta e aplica o orçamento")
    ask.add_argument("query")
    ask.add_argument("--context", action="append", default=[], help="Turno recente (repetível)")
    ask.add_argument("--window", type=int)
    ask.add_argument("--output-reserve", type=int)
    ask.add_argument("--seed")
    ask.add_argument("--focus", choices=DIMENSIONS)
    ask.add_argument("--radius", type=float)
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--threshold", type=float)
    ask.add_argument("--no-links", action="store_true")
    ask.add_argument("--no-seed", action="store_true")
    ask.add_argument("--base-message-id", action="append")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    demo = sub.add_parser("demo", help="Executa as consultas de demonstração")
    demo.set_defaults(func=cmd_demo)

    agent_demo = sub.add_parser(
        "agent-demo",
        help="Executa a sequência do agente com quadro branco e atenção temática",
    )
    agent_demo.add_argument("--agent-data", default=str(AGENT_DATA))
    agent_demo.add_argument("--json", action="store_true")
    agent_demo.set_defaults(func=cmd_agent_demo)

    benchmark = sub.add_parser(
        "benchmark",
        help="Compara histórico, busca lexical, Sinapse e seleção manual",
    )
    benchmark.add_argument("--cases", default="benchmark/cases_eval.json")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--deepseek", action="store_true", help="Faz chamadas reais")
    benchmark.add_argument("--model")
    benchmark.add_argument("--out", default="benchmark_out")
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    scale = sub.add_parser(
        "scale-benchmark",
        help="Teste de escala com 1k/10k/100k mensagens sintéticas (sem DeepSeek)",
    )
    scale.add_argument("--scales", default="1000,10000,100000")
    scale.add_argument("--seed", type=int, default=42)
    scale.add_argument("--candidate-k", type=int, default=200)
    scale.add_argument("--out", default="benchmark_out_scale")
    scale.add_argument("--json", action="store_true")
    scale.set_defaults(func=cmd_scale_benchmark)

    recovery = sub.add_parser(
        "scale-recovery",
        help="Rastreia perdas em 100k e compara a busca com histórico recente",
    )
    recovery.add_argument("--scale", type=int, default=100000)
    recovery.add_argument("--seed", type=int, default=42)
    recovery.add_argument("--candidate-k", type=int, default=200)
    recovery.add_argument("--out", default="benchmark_out_scale_recovery")
    recovery.add_argument("--json", action="store_true")
    recovery.set_defaults(func=cmd_scale_recovery)

    resumption = sub.add_parser(
        "resumption-benchmark",
        help="Retomada de projeto: histórico recente vs Sinapse (simulado por padrão)",
    )
    resumption.add_argument("--deepseek", action="store_true")
    resumption.add_argument("--model")
    resumption.add_argument("--out", default="benchmark_out_resumption_round3")
    resumption.add_argument("--json", action="store_true")
    resumption.set_defaults(func=cmd_resumption_benchmark)

    real = sub.add_parser(
        "resumption-real-pairs",
        help="Executa os dez pares pareados com o deepseek-flash real",
    )
    real.add_argument(
        "--pairs",
        default="benchmark_out_resumption_round3/real_comparison.json",
    )
    real.add_argument("--out", default="benchmark_out_resumption_real")
    real.add_argument("--model")
    real.add_argument("--json", action="store_true")
    real.set_defaults(func=cmd_resumption_real_pairs)

    review_prepare = sub.add_parser(
        "resumption-review-prepare",
        help="Preenche o formulário cego de revisão humana sem revelar condições",
    )
    review_prepare.add_argument(
        "--review",
        default="benchmark_out_resumption_real/real_review_blind.json",
    )
    review_prepare.add_argument(
        "--results",
        default="benchmark_out_resumption_real/real_results.json",
    )
    review_prepare.add_argument(
        "--out",
        default="benchmark_out_resumption_real/real_review_blind.json",
    )
    review_prepare.set_defaults(func=cmd_resumption_review_prepare)

    review_consolidate = sub.add_parser(
        "resumption-review-consolidate",
        help="Consolida a revisão humana e revela A/B após as notas",
    )
    review_consolidate.add_argument(
        "--review",
        default="benchmark_out_resumption_real/real_review_blind.json",
    )
    review_consolidate.add_argument(
        "--results",
        default="benchmark_out_resumption_real/real_results.json",
    )
    review_consolidate.add_argument(
        "--out",
        default="benchmark_out_resumption_real/human_consolidated.json",
    )
    review_consolidate.set_defaults(func=cmd_resumption_review_consolidate)

    rubric_v2 = sub.add_parser(
        "resumption-rubric-v2",
        help="Calibra a rubrica v2 nos pares revisados e nos casos sintéticos",
    )
    rubric_v2.add_argument(
        "--out",
        default="benchmark_out_resumption_real/rubric_v2_calibration.json",
    )
    rubric_v2.add_argument("--json", action="store_true")
    rubric_v2.set_defaults(func=cmd_resumption_rubric_v2)

    instruction = sub.add_parser(
        "instruction-experiment",
        help="Compara duas instruções de geração com memória e orçamento fixos",
    )
    instruction.add_argument(
        "--plan",
        default="benchmark_out_resumption_real/instruction_experiment_plan.json",
    )
    instruction.add_argument(
        "--out", default="benchmark_out_instruction_experiment"
    )
    instruction.add_argument("--model")
    instruction.add_argument("--json", action="store_true")
    instruction.set_defaults(func=cmd_instruction_experiment)

    instruction_review = sub.add_parser(
        "instruction-review-consolidate",
        help="Consolida a revisão cega de instruções e revela A/B",
    )
    instruction_review.add_argument(
        "--review",
        default="benchmark_out_instruction_experiment/instruction_review_blind.json",
    )
    instruction_review.add_argument(
        "--results",
        default="benchmark_out_instruction_experiment/instruction_results.json",
    )
    instruction_review.add_argument(
        "--out",
        default="benchmark_out_instruction_experiment/instruction_consolidated.json",
    )
    instruction_review.set_defaults(func=cmd_instruction_review_consolidate)

    validation = sub.add_parser(
        "resumption-validate",
        help="Valida localmente os pares preparados para a comparação real",
    )
    validation.add_argument(
        "--pairs",
        default="benchmark_out_resumption_round3/real_comparison.json",
    )
    validation.add_argument(
        "--out",
        default="benchmark_out_resumption_round3/local_validation.json",
    )
    validation.add_argument("--json", action="store_true")
    validation.set_defaults(func=cmd_resumption_validate)

    answer = sub.add_parser(
        "answer-benchmark",
        help="Compara qualidade de resposta por condição de memória",
    )
    answer.add_argument("--cases", default="benchmark/cases_answer.json")
    answer.add_argument("--limit", type=int)
    answer.add_argument("--deepseek", action="store_true")
    answer.add_argument("--model")
    answer.add_argument("--out", default="benchmark_out_answer")
    answer.add_argument("--json", action="store_true")
    answer.set_defaults(func=cmd_answer_benchmark)

    learn = sub.add_parser(
        "learn-benchmark",
        help="Aprende pesos de relevância e compara fixo vs aprendido",
    )
    learn.add_argument("--out", default="benchmark_out_learning")
    learn.add_argument("--json", action="store_true")
    learn.set_defaults(func=cmd_learn_benchmark)

    curiosity_learn = sub.add_parser(
        "curiosity-learn",
        help="Aprende a política de curiosidade (parar, vínculos, tema, correção)",
    )
    curiosity_learn.add_argument("--out", default="benchmark_out_curiosity")
    curiosity_learn.add_argument("--json", action="store_true")
    curiosity_learn.set_defaults(func=cmd_curiosity_learn)

    curiosity_round2 = sub.add_parser(
        "curiosity-round2",
        help="Rodada 2: força de vínculo e correção da métrica de inclusões",
    )
    curiosity_round2.add_argument("--out", default="benchmark_out_curiosity_round2")
    curiosity_round2.add_argument("--json", action="store_true")
    curiosity_round2.set_defaults(func=cmd_curiosity_round2)

    curiosity_round3 = sub.add_parser(
        "curiosity-round3",
        help="Rodada 3: vínculo compatível com dimensão/atributo",
    )
    curiosity_round3.add_argument("--out", default="benchmark_out_curiosity_round3")
    curiosity_round3.add_argument("--json", action="store_true")
    curiosity_round3.set_defaults(func=cmd_curiosity_round3)

    reduction = sub.add_parser(
        "context-reduction",
        help="Compara seletor atual com redução marginal de contexto",
    )
    reduction.add_argument("--out", default="benchmark_out_context_reduction")
    reduction.add_argument("--json", action="store_true")
    reduction.set_defaults(func=cmd_context_reduction)

    paired = sub.add_parser(
        "paired-answer",
        help="Compara resposta atual vs reduzida em pares cegos",
    )
    paired.add_argument("--deepseek", action="store_true")
    paired.add_argument("--model")
    paired.add_argument("--out", default="benchmark_out_paired")
    paired.add_argument("--json", action="store_true")
    paired.set_defaults(func=cmd_paired_answer)

    genetic = sub.add_parser(
        "genetic-policy",
        help="Otimiza parâmetros da política com busca aleatória e genética",
    )
    genetic.add_argument("--out", default="benchmark_out_genetic")
    genetic.add_argument("--json", action="store_true")
    genetic.set_defaults(func=cmd_genetic_policy)

    policy_answer = sub.add_parser(
        "policy-answer",
        help="Compara as três políticas congeladas com respostas",
    )
    policy_answer.add_argument("--deepseek", action="store_true")
    policy_answer.add_argument("--model")
    policy_answer.add_argument("--out", default="benchmark_out_policy_answer")
    policy_answer.add_argument("--json", action="store_true")
    policy_answer.set_defaults(func=cmd_policy_answer)

    purpose = sub.add_parser(
        "purpose-benchmark",
        help="Compara seleção por tema e propósito com a referência",
    )
    purpose.add_argument("--out", default="benchmark_out_purpose")
    purpose.add_argument("--json", action="store_true")
    purpose.set_defaults(func=cmd_purpose_benchmark)

    purpose_answer = sub.add_parser(
        "purpose-answer",
        help="Compara respostas da referência e da seleção por propósito",
    )
    purpose_answer.add_argument("--deepseek", action="store_true")
    purpose_answer.add_argument("--model")
    purpose_answer.add_argument("--out", default="benchmark_out_purpose_answer")
    purpose_answer.add_argument("--json", action="store_true")
    purpose_answer.set_defaults(func=cmd_purpose_answer)

    purpose_report = sub.add_parser(
        "purpose-report",
        help="Reprocessa resultados salvos por propósito, sem novas chamadas",
    )
    purpose_report.add_argument(
        "--results",
        default="benchmark_out_purpose_answer_deepseek/purpose_answer_results.json",
    )
    purpose_report.add_argument(
        "--review",
        default="benchmark_out_purpose_answer_deepseek/purpose_answer_review.json",
    )
    purpose_report.add_argument(
        "--human-review-out",
        default="benchmark_out_purpose_answer_deepseek/purpose_review_human.json",
    )
    purpose_report.add_argument("--out", default="benchmark_out_purpose_report")
    purpose_report.add_argument("--json", action="store_true")
    purpose_report.set_defaults(func=cmd_purpose_report)

    removal_round = sub.add_parser(
        "removal-round",
        help="Executa o plano de remoções dirigidas v2",
    )
    removal_round.add_argument("--deepseek", action="store_true")
    removal_round.add_argument("--model")
    removal_round.add_argument("--out", default="benchmark_out_removal_round")
    removal_round.add_argument("--json", action="store_true")
    removal_round.set_defaults(func=cmd_removal_round)

    dependency_round = sub.add_parser(
        "dependency-round",
        help="Testa omissão conservadora de dependências condicionais",
    )
    dependency_round.add_argument("--deepseek", action="store_true")
    dependency_round.add_argument("--model")
    dependency_round.add_argument("--out", default="benchmark_out_dependency_round")
    dependency_round.add_argument("--json", action="store_true")
    dependency_round.set_defaults(func=cmd_dependency_round)

    attribution_round = sub.add_parser(
        "attribution-round",
        help="Avalia atribuição de entidade/decisão e ambiguidade",
    )
    attribution_round.add_argument("--deepseek", action="store_true")
    attribution_round.add_argument("--model")
    attribution_round.add_argument("--out", default="benchmark_out_attribution_round")
    attribution_round.add_argument("--json", action="store_true")
    attribution_round.set_defaults(func=cmd_attribution_round)

    chat = sub.add_parser("chat", help="Executa o chatbot com memória e API DeepSeek")
    chat.add_argument("query")
    chat.add_argument("--context", action="append", default=[], help="Turno recente (repetível)")
    chat.add_argument("--window", type=int)
    chat.add_argument("--output-reserve", type=int)
    chat.add_argument("--model")
    chat.add_argument("--base-url")
    chat.add_argument("--max-tokens", type=int)
    chat.add_argument("--thinking", choices=["enabled", "disabled"])
    chat.add_argument("--simulate", action="store_true", help="Não chama a rede")
    chat.add_argument(
        "--simulate-reply",
        default="Resposta simulada: nenhuma chamada de rede foi feita.",
    )
    chat.add_argument("--seed")
    chat.add_argument("--focus", choices=DIMENSIONS)
    chat.add_argument("--radius", type=float)
    chat.add_argument("--top-k", type=int)
    chat.add_argument("--threshold", type=float)
    chat.add_argument("--no-seed", action="store_true")
    chat.add_argument("--base-message-id", action="append")
    chat.add_argument("--json", action="store_true")
    chat.set_defaults(func=cmd_chat)

    groups = sub.add_parser("groups", help="Lista grupos sobrepostos")
    groups.add_argument("--threshold", type=float)
    groups.add_argument("--json", action="store_true")
    groups.set_defaults(func=cmd_groups)

    group = sub.add_parser("group", help="Mostra um grupo")
    group.add_argument("group_id")
    group.set_defaults(func=cmd_group)

    replies = sub.add_parser("replies", help="Mostra respostas diretas")
    replies.add_argument("message_id")
    replies.set_defaults(func=cmd_replies)

    project = sub.add_parser(
        "project", help="Copiloto de projetos: criar, conversar e retomar"
    )
    project.add_argument("--root", default=DEFAULT_ROOT)
    project_sub = project.add_subparsers(
        dest="project_command", required=True
    )
    project_create = project_sub.add_parser("create", help="Cria um projeto")
    project_create.add_argument("name")
    project_create.set_defaults(func=cmd_project)
    project_list = project_sub.add_parser("list", help="Lista projetos")
    project_list.set_defaults(func=cmd_project)
    project_select = project_sub.add_parser("select", help="Seleciona um projeto")
    project_select.add_argument("project")
    project_select.set_defaults(func=cmd_project)
    project_say = project_sub.add_parser("say", help="Envia uma mensagem")
    project_say.add_argument("text")
    project_say.add_argument("--deepseek", action="store_true")
    project_say.set_defaults(func=cmd_project)
    project_resume = project_sub.add_parser(
        "resume", help="Retoma o projeto com decisão, motivo, teste e pendência"
    )
    project_resume.add_argument("--query", default="Onde paramos?")
    project_resume.add_argument("--json", action="store_true")
    project_resume.set_defaults(func=cmd_project)
    project_show = project_sub.add_parser(
        "show", help="Mostra a mensagem original citada"
    )
    project_show.add_argument("message_id")
    project_show.set_defaults(func=cmd_project)
    project_restrict = project_sub.add_parser(
        "restrict", help="Declara uma restrição de sensibilidade"
    )
    project_restrict.add_argument("text")
    project_restrict.set_defaults(func=cmd_project)

    bridge = sub.add_parser(
        "bridge", help="Ponte local para agentes: retomar, buscar e registrar"
    )
    bridge.add_argument("--root", default=DEFAULT_ROOT)
    bridge.add_argument("--repo", default=".")
    bridge.add_argument("--project")
    bridge_sub = bridge.add_subparsers(dest="bridge_command", required=True)

    def bridge_parser(name: str, help_text: str):
        parser = bridge_sub.add_parser(name, help=help_text)
        parser.add_argument("--json", action="store_true")
        return parser

    bridge_resume = bridge_parser(
        "resume", "Retoma objetivo, tarefa, decisões, teste e próximo passo"
    )
    bridge_resume.add_argument("--query", default=DEFAULT_QUERY)
    bridge_resume.set_defaults(func=cmd_bridge)

    bridge_search = bridge_parser(
        "search", "Busca evidências pelo assunto da tarefa"
    )
    bridge_search.add_argument("query")
    bridge_search.add_argument("--limit", type=int, default=8)
    bridge_search.set_defaults(func=cmd_bridge)

    bridge_record = bridge_parser(
        "record", "Registra progresso com ID estável e idempotente"
    )
    bridge_record.add_argument("--kind", required=True, choices=ENTRY_KINDS)
    bridge_record.add_argument("--text", required=True)
    bridge_record.add_argument("--entry-id")
    bridge_record.add_argument("--supersedes")
    bridge_record.add_argument("--reason")
    bridge_record.add_argument("--resolves")
    bridge_record.add_argument("--author", default="agent")
    bridge_record.add_argument("--component")
    bridge_record.add_argument("--file", action="append")
    bridge_record.add_argument("--issue", action="append")
    bridge_record.add_argument("--pr", action="append")
    bridge_record.add_argument("--commit", action="append")
    bridge_record.add_argument("--command")
    bridge_record.add_argument("--symptom")
    bridge_record.add_argument("--cause")
    bridge_record.add_argument("--restriction", action="append")
    bridge_record.add_argument("--alternative", action="append")
    bridge_record.add_argument("--report-source")
    bridge_record.add_argument("--reported-by")
    bridge_record.add_argument("--status")
    bridge_record.add_argument("--related-to")
    bridge_record.set_defaults(func=cmd_bridge)

    bridge_brief = bridge_parser(
        "brief", "Gera bullets de implementação para o OpenCode"
    )
    bridge_brief.add_argument("--query", default=DEFAULT_QUERY)
    bridge_brief.set_defaults(func=cmd_bridge)

    bridge_code_status = bridge_parser(
        "code-status", "Mostra repositório, commit e índice de código"
    )
    bridge_code_status.set_defaults(func=cmd_bridge)

    bridge_code_import = bridge_parser(
        "code-import", "Indexa o código do commit atual ou informado"
    )
    bridge_code_import.add_argument("--commit")
    bridge_code_import.set_defaults(func=cmd_bridge)

    bridge_code_search = bridge_parser(
        "code-search", "Busca símbolos, caminhos e comportamento no código"
    )
    bridge_code_search.add_argument("query")
    bridge_code_search.add_argument("--limit", type=int, default=5)
    bridge_code_search.set_defaults(func=cmd_bridge)

    bridge_status = bridge_parser(
        "status", "Mostra o projeto vinculado a este repositório"
    )
    bridge_status.set_defaults(func=cmd_bridge)

    bridge_link = bridge_parser(
        "link", "Vincula o repositório e grava .sinapse.json"
    )
    bridge_link.set_defaults(func=cmd_bridge)

    bridge_projects = bridge_parser(
        "projects", "Lista projetos e repositórios vinculados"
    )
    bridge_projects.set_defaults(func=cmd_bridge)

    audit = sub.add_parser(
        "context-audit",
        help="Compara o contexto atual com histórico recente e seleção seletiva",
    )
    audit.add_argument(
        "--out",
        default="benchmark_out_context_audit/context_audit.json",
    )
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(func=cmd_context_audit)

    web = sub.add_parser("web", help="Inicia o Farol, interface web local")
    web.add_argument("--root", default=DEFAULT_ROOT)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--settings", default="data/farol_settings.json")
    web.set_defaults(func=cmd_web)

    anticipation = sub.add_parser(
        "anticipation-benchmark",
        help="Compara quadro antecipatório ligado e desligado",
    )
    anticipation.add_argument(
        "--out",
        default="benchmark_out_anticipation/anticipation_benchmark.json",
    )
    anticipation.add_argument("--json", action="store_true")
    anticipation.set_defaults(func=cmd_anticipation_benchmark)

    diagnostics = sub.add_parser(
        "diagnostics", help="Consulta somente leitura dos eventos técnicos"
    )
    diagnostics.add_argument("--root", default=DEFAULT_ROOT)
    diagnostics_sub = diagnostics.add_subparsers(
        dest="diagnostics_command", required=True
    )
    diagnostics_project = diagnostics_sub.add_parser(
        "project", help="Lista eventos de um projeto"
    )
    diagnostics_project.add_argument("project")
    diagnostics_project.add_argument("--since")
    diagnostics_project.add_argument("--json", action="store_true")
    diagnostics_project.set_defaults(func=cmd_diagnostics)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
