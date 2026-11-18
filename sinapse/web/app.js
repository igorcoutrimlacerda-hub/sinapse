(() => {
  "use strict";

  const state = {
    projects: [],
    active: null,
    data: null,
    busy: false,
    loadVersion: 0,
    deleteTarget: null,
  };

  const $ = (id) => document.getElementById(id);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  async function api(path, options = {}) {
    const { timeout, ...rest } = options;
    const controller = timeout ? new AbortController() : null;
    const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
    try {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...rest,
        ...(controller ? { signal: controller.signal } : {}),
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }
      if (!response.ok) throw new Error(payload.error || `Erro HTTP ${response.status}`);
      return payload;
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw new Error("Tempo esgotado ao falar com o servidor local.");
      }
      throw error;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  function showError(message) {
    const banner = $("error-banner");
    banner.textContent = message || "";
    banner.hidden = !message;
  }

  function showNotice(message) {
    const banner = $("notice-banner");
    banner.textContent = message || "";
    banner.hidden = !message;
  }

  function setBusy(busy, label) {
    state.busy = busy;
    const hasProject = Boolean(state.active);
    $("send-button").disabled = busy || !hasProject;
    $("resume-button").disabled = busy || !hasProject;
    $("brief-button").disabled = busy || !hasProject;
    $("repo-save").disabled = busy || !hasProject;
    $("repo-input").disabled = busy || !hasProject;
    $("semantic-toggle").disabled = busy || !hasProject;
    $("code-import").disabled = busy || !hasProject;
    $("message-input").disabled = busy || !hasProject;
    $("deepseek-toggle").disabled = busy || !hasProject;
    $("new-project-button").disabled = busy;
    $("project-select").disabled = busy || !state.projects.length;
    $("delete-project").disabled = busy || !hasProject;
    if (label) $("response-mode").textContent = label;
  }

  function chips(ids) {
    const box = el("div", "chips");
    (ids || []).forEach((id) => {
      if (!id) return;
      const button = el("button", "chip", id);
      button.type = "button";
      button.dataset.sourceId = id;
      button.title = `Abrir mensagem ${id}`;
      box.append(button);
    });
    return box;
  }

  function field(label, value, sourceIds = [], full = false, details = []) {
    const box = el("div", `resume-field${full ? " full" : ""}`);
    box.append(el("div", "label", label));
    box.append(el("div", `value${value ? "" : " empty"}`, value || "Ainda não definido"));
    (details || []).forEach((line) => box.append(el("div", "field-detail", line)));
    if (sourceIds && sourceIds.length) box.append(chips(sourceIds));
    return box;
  }

  function entryDetails(entry) {
    if (!entry) return [];
    const lines = [];
    if (entry.reason) lines.push(`Motivo: ${entry.reason}`);
    const labels = {
      restrictions: "Restrições",
      files: "Arquivos",
      issues: "Issues",
      prs: "PRs",
      commits: "Commits",
      command: "Comando",
      report_source: "Relatório",
      symptom: "Sintoma",
    };
    Object.entries(labels).forEach(([key, label]) => {
      const value = entry[key];
      if (!value) return;
      lines.push(`${label}: ${Array.isArray(value) ? value.join("; ") : value}`);
    });
    return lines;
  }

  function renderProjects() {
    const select = $("project-select");
    select.replaceChildren();
    state.projects.forEach((project) => {
      const option = el("option", "", project.name);
      option.value = project.id;
      select.append(option);
    });
    if (state.active) select.value = state.active;
    select.disabled = state.busy || !state.projects.length;
    $("empty-projects").hidden = state.projects.length > 0;
    $("delete-project").disabled = state.busy || !state.active;
  }

  function emptyConversation(text) {
    $("conversation").replaceChildren();
    const box = el("div", "empty-state");
    box.append(el("h3", "", "Sem mensagens ainda"));
    box.append(el("p", "", text));
    $("conversation").append(box);
  }

  function resetDashboard() {
    $("goal-value").textContent = "Ainda não definido";
    $("task-value").textContent = "Ainda não definido";
    $("decision-value").textContent = "Nenhuma decisão vigente";
    $("decision-note").textContent = "";
    $("decision-sources").replaceChildren();
    $("test-value").textContent = "Ainda não registrado";
    $("test-note").textContent = "";
    $("test-sources").replaceChildren();
    $("repo-path").textContent = "Ainda não associado";
    $("repo-input").value = "";
    $("repo-hint").textContent = "";
    $("semantic-toggle").checked = false;
    $("code-status").textContent = "Nenhum código indexado";
    $("code-meta").textContent = "";
    $("focus-status").className = "badge neutral";
    $("focus-status").textContent = "—";
    $("focus-action").textContent = "Ainda não definido";
    $("focus-why").textContent = "";
    $("focus-block").textContent = "";
    $("focus-sources").replaceChildren();
    $("checklist").replaceChildren(el("li", "muted", "Ainda não definido"));
    $("next-step").textContent = "Ainda não definido";
    $("next-sources").replaceChildren();
  }

  function renderEmptyAll() {
    state.active = null;
    state.data = null;
    $("project-title").textContent = "Selecione um projeto";
    $("project-subtitle").textContent = "O histórico e a memória são isolados por projeto.";
    $("resume-card").hidden = true;
    emptyConversation("Crie um projeto na barra lateral para começar.");
    resetDashboard();
    $("message-input").disabled = true;
    $("send-button").disabled = true;
    $("resume-button").disabled = true;
    $("brief-button").disabled = true;
    $("deepseek-toggle").disabled = true;
    renderRepo(null);
    renderProjects();
  }

  function renderMessages(messages) {
    $("conversation").replaceChildren();
    if (!messages.length) {
      emptyConversation("Registre uma decisão, um teste ou uma pendência para começar.");
      return;
    }
    messages.forEach((message) => {
      const wrapper = el("article", `message ${message.author === "user" ? "user" : "assistant"}`);
      const meta = el("div", "message-meta");
      meta.append(el("span", "", `[${message.id}] ${message.author === "user" ? "você" : "Farol"}`));
      if (message.source === "deepseek") meta.append(el("span", "badge deepseek", "DeepSeek"));
      if (message.source === "local") meta.append(el("span", "badge local", "Local"));
      if (message.code_sources && message.code_sources.length) {
        const codeBox = el("div", "chips");
        message.code_sources.forEach((source) => {
          const chip = el("button", "chip code-chip", `${source.path}:${source.start}-${source.end}`);
          chip.type = "button";
          chip.dataset.codeId = source.id;
          chip.title = `Abrir ${source.path}:${source.start}-${source.end}`;
          codeBox.append(chip);
        });
        wrapper.append(codeBox);
      }
      wrapper.append(meta);
      wrapper.append(el("div", "message-body", message.text));
      $("conversation").append(wrapper);
    });
    $("conversation").scrollTop = $("conversation").scrollHeight;
  }

  function renderResume(resume) {
    const card = $("resume-card");
    const content = $("resume-content");
    content.replaceChildren();
    if (!resume) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const decision = resume.decision;
    content.append(field(
      "Tarefa atual",
      resume.current_task || "Ainda não registrada",
      resume.current_task_source_ids || []
    ));
    if (resume.goal) {
      content.append(field("Objetivo", resume.goal, resume.goal_source_ids || []));
    }
    content.append(field(
      "Decisão técnica",
      decision ? decision.value : "Nenhuma decisão vigente sobre este assunto",
      decision ? decision.source_ids : [],
      false,
      entryDetails(decision)
    ));
    if (resume.reason && !(decision && decision.reason)) {
      content.append(field("Motivo", resume.reason, resume.reason_source_ids || []));
    }
    const test = resume.last_test || resume.completed;
    content.append(field(
      "Último teste observado",
      test ? test.text : "Ainda não registrado",
      test ? test.source_ids : [],
      false,
      entryDetails(test)
    ));
    if (resume.bugs && resume.bugs.length) {
      const bug = resume.bugs[resume.bugs.length - 1];
      const label = bug.status === "aberto" ? "Bug aberto" : "Bug corrigido";
      content.append(field(label, bug.text, bug.source_ids || [], false, entryDetails(bug)));
    }
    const pending = resume.pending || (resume.bugs && resume.bugs[0]);
    if (pending) content.append(field("Próximo passo", pending.text, pending.source_ids || []));
    else if (resume.no_pending) content.append(field("Próximo passo", "Sem próximo passo registrado", resume.no_pending.source_ids));
    else content.append(field("Próximo passo", "Ainda não definido", []));
    if (resume.gaps && resume.gaps.length) { content.append(field("Lacunas", resume.gaps.join("; "), [], true)); }
    if (resume.corrections && resume.corrections.length && resume.show_corrections) {
      const corrections = el("div", "corrections");
      corrections.append(el("strong", "", "Correções registradas: "));
      corrections.append(document.createTextNode(
        resume.corrections.map((item) => `${item.from} → ${item.to || "?"} [${(item.source_ids || []).concat(item.correction_source_ids || []).join(", ")}]`).join(" · ")
      ));
      content.append(corrections);
    }
    if (resume.restricted && resume.restricted.length) {
      content.append(field("Conteúdo restrito", resume.restricted.join("; "), [], true));
    }
  }

  function renderRepo(project) {
    const value = project && project.repo_path ? project.repo_path : "";
    $("repo-path").textContent = value || "Ainda não associado";
    $("repo-path").classList.toggle("empty", !value);
    $("repo-input").value = value;
    $("repo-hint").textContent = value
      ? "A ponte Sinapse usa este caminho para identificar o projeto."
      : "Associe o repositório para a ponte Sinapse reutilizar repo_path.";
    $("semantic-toggle").checked = Boolean(project && project.semantic);
  }

  async function saveSemantic() {
    if (!state.active) return;
    $("semantic-toggle").disabled = true;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(state.active)}/semantic`, {
        method: "POST",
        body: JSON.stringify({ enabled: $("semantic-toggle").checked }),
      });
      renderProject(data);
    } catch (error) {
      showError(error.message);
      $("semantic-toggle").checked = !$("semantic-toggle").checked;
    } finally {
      $("semantic-toggle").disabled = false;
    }
  }

  function renderCode(project) {
    const meta = project && project.code_index ? project.code_index : null;
    if (!meta || !meta.indexed) {
      $("code-status").textContent = "Nenhum código indexado";
      $("code-status").classList.add("empty");
      $("code-meta").textContent = project && project.repo_path
        ? "Use Indexar para ler o commit atual do repositório."
        : "Associe um repositório primeiro.";
      return;
    }
    $("code-status").textContent = `${meta.commit_short || meta.commit} · ${meta.files} arquivos`;
    $("code-status").classList.remove("empty");
    const license = meta.license && meta.license.name ? meta.license.name : "licença não identificada";
    $("code-meta").textContent = `${meta.repo_path} · ${meta.chunks} blocos · ${license}`;
  }

  async function importCode() {
    if (!state.active) return;
    $("code-import").disabled = true;
    showNotice("Indexando código do commit atual…");
    try {
      const data = await api(`/api/projects/${encodeURIComponent(state.active)}/code/import`, {
        method: "POST",
        body: "{}",
      });
      renderProject(data);
      showNotice("Código indexado.");
    } catch (error) {
      showError(error.message);
    } finally {
      $("code-import").disabled = false;
    }
  }

  function openCode(chunkId) {
    if (!state.active) return;
    api(`/api/projects/${encodeURIComponent(state.active)}/code/chunk/${encodeURIComponent(chunkId)}`)
      .then((chunk) => {
        $("code-title").textContent = chunk.symbol || chunk.path;
        $("code-meta-line").textContent = `${chunk.path}:${chunk.start}-${chunk.end} @ ${String(chunk.commit || "").slice(0, 12)}`;
        $("code-text").textContent = chunk.text || "";
        $("code-modal").hidden = false;
      })
      .catch((error) => showError(error.message));
  }

  function closeCode() {
    $("code-modal").hidden = true;
  }

  function renderDashboard(dashboard) {
    if (!dashboard) {
      resetDashboard();
      return;
    }
    $("goal-value").textContent = dashboard.goal || "Ainda não definido";
    $("goal-value").classList.toggle("empty", !dashboard.goal);
    $("task-value").textContent = dashboard.current_task || "Ainda não definido";
    $("task-value").classList.toggle("empty", !dashboard.current_task);

    const resume = state.data ? state.data.resume : null;
    const decision = resume && resume.decision;
    $("decision-value").textContent = decision ? decision.value : "Nenhuma decisão vigente";
    $("decision-value").classList.toggle("empty", !decision);
    $("decision-note").textContent = decision && decision.reason ? `Motivo: ${decision.reason}` : "";
    $("decision-sources").replaceChildren();
    if (decision) (decision.source_ids || []).forEach((id) => $("decision-sources").append(...chips([id]).childNodes));
    const test = resume && (resume.last_test || resume.completed);
    $("test-value").textContent = test ? test.text : "Ainda não registrado";
    $("test-value").classList.toggle("empty", !test);
    $("test-note").textContent = test && test.command ? `Comando: ${test.command}` : (test && test.report_source ? `Relatório: ${test.report_source}` : "");
    $("test-sources").replaceChildren();
    if (test) (test.source_ids || []).forEach((id) => $("test-sources").append(...chips([id]).childNodes));

    const focus = dashboard.focus || {};
    const status = $("focus-status");
    status.className = "badge neutral";
    status.textContent = "—";
    if (focus.status === "proposed") {
      status.textContent = "Proposta";
      status.className = "badge proposed";
    } else if (focus.status === "confirmed") {
      status.textContent = "Confirmado";
      status.className = "badge confirmed";
    } else if (focus.status === "indefinite") {
      status.textContent = "Indefinido";
      status.className = "badge indefinite";
    }
    $("focus-action").textContent = focus.action || "Ainda não definido";
    $("focus-action").classList.toggle("empty", !focus.action);
    $("focus-why").textContent = focus.why ? `Por quê: ${focus.why}` : "";
    $("focus-block").textContent = focus.blocked_by ? `Bloqueio/dependência: ${focus.blocked_by}` : "";
    $("focus-sources").replaceChildren();
    (focus.source_ids || []).forEach((id) => $("focus-sources").append(...chips([id]).childNodes));

    const checklist = $("checklist");
    checklist.replaceChildren();
    if (!dashboard.checklist || !dashboard.checklist.length) {
      checklist.append(el("li", "muted", "Ainda não definido"));
    } else {
      dashboard.checklist.forEach((item) => {
        const li = el("li", item.status === "done" ? "done" : "pending");
        li.append(el("span", "check", item.status === "done" ? "V" : "\t"));
        li.append(el("span", "item-text", item.text));
        const sourceBox = chips(item.source_ids || []);
        if (sourceBox.childNodes.length) li.append(sourceBox);
        checklist.append(li);
      });
    }

    const nextStep = dashboard.next_step;
    $("next-step").textContent = nextStep ? nextStep.text : "Sem próximo passo registrado";
    $("next-step").classList.toggle("empty", !nextStep);
    $("next-sources").replaceChildren();
    if (nextStep) (nextStep.source_ids || []).forEach((id) => $("next-sources").append(...chips([id]).childNodes));
  }

  function renderProject(data) {
    state.data = data;
    state.active = data.project.id;
    $("project-title").textContent = data.project.name;
    $("project-subtitle").textContent = `Projeto ${data.project.id} · histórico e estado isolados`;
    $("message-input").disabled = false;
    $("send-button").disabled = false;
    $("resume-button").disabled = false;
    $("brief-button").disabled = false;
    $("deepseek-toggle").disabled = false;
    $("project-select").value = data.project.id;
    renderRepo(data.project);
    renderCode(data.project);
    renderProjects();
    renderMessages(data.messages || []);
    renderResume(data.resume);
    renderDashboard(data.dashboard);
    showError("");
    showNotice("");
  }

  async function saveRepo() {
    if (!state.active) return;
    $("repo-save").disabled = true;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(state.active)}/repo`, {
        method: "POST",
        body: JSON.stringify({ repo_path: $("repo-input").value.trim() }),
      });
      renderProject(data);
    } catch (error) {
      showError(error.message);
    } finally {
      $("repo-save").disabled = false;
    }
  }

  async function loadProject(projectId) {
    const version = ++state.loadVersion;
    state.active = projectId;
    setBusy(true, "Carregando…");
    try {
      const data = await api(`/api/projects/${encodeURIComponent(projectId)}`);
      if (version !== state.loadVersion) return;
      renderProject(data);
    } catch (error) {
      if (version === state.loadVersion) showError(error.message);
    } finally {
      if (version === state.loadVersion) setBusy(false);
    }
  }

  async function loadProjects(preferredId = null) {
    try {
      const data = await api("/api/projects");
      state.projects = data.projects || [];
      const target = preferredId && state.projects.some((p) => p.id === preferredId)
        ? preferredId
        : data.active && state.projects.some((p) => p.id === data.active)
          ? data.active
          : state.projects[0]?.id || null;
      renderProjects();
      if (target) await loadProject(target);
      else renderEmptyAll();
    } catch (error) {
      showError(error.message);
    }
  }

  async function switchProject(projectId) {
    if (!projectId || state.busy) return;
    const version = ++state.loadVersion;
    setBusy(true, "Trocando projeto…");
    closeSource();
    closeDelete();
    try {
      await api(`/api/projects/${encodeURIComponent(projectId)}/select`, { method: "POST", body: "{}" });
      if (version !== state.loadVersion) return;
      await loadProject(projectId);
    } catch (error) {
      if (version === state.loadVersion) {
        showError(error.message);
        $("project-select").value = state.active || "";
      }
    } finally {
      if (version === state.loadVersion) setBusy(false);
    }
  }

  async function sendMessage(text) {
    if (!state.active || state.busy || !text.trim()) return;
    const projectId = state.active;
    const version = state.loadVersion;
    const deepseek = $("deepseek-toggle").checked;
    setBusy(true, "Registrando…");
    try {
      const data = await api(`/api/projects/${encodeURIComponent(projectId)}/messages`, {
        method: "POST",
        body: JSON.stringify({ text: text.trim(), deepseek }),
      });
      if (version !== state.loadVersion || projectId !== state.active) return;
      renderProject(data);
      $("message-input").value = "";
      $("response-mode").textContent = `Resposta ${data.source === "deepseek" ? "DeepSeek" : "local"}`;
      if (data.warning) showNotice(data.warning);
      else showNotice("");
    } catch (error) {
      if (version === state.loadVersion) showError(error.message);
    } finally {
      if (version === state.loadVersion) setBusy(false);
    }
  }

  async function openSource(messageId) {
    if (!state.active) return;
    const projectId = state.active;
    const version = state.loadVersion;
    const modal = $("source-modal");
    const error = $("source-error");
    error.hidden = true;
    $("source-title").textContent = `Mensagem ${messageId}`;
    $("source-meta").textContent = "";
    $("source-text").textContent = "Carregando…";
    modal.hidden = false;
    try {
      const message = await api(`/api/projects/${encodeURIComponent(projectId)}/messages/${encodeURIComponent(messageId)}`);
      if (version !== state.loadVersion || projectId !== state.active) return;
      $("source-meta").textContent = `${message.project_name} (${message.project_id}) · ${message.author} · ${message.created_at || ""}`;
      $("source-text").textContent = message.text;
    } catch (errorObject) {
      if (version === state.loadVersion) {
        $("source-text").textContent = "";
        error.textContent = errorObject.message;
        error.hidden = false;
      }
    }
  }

  function closeSource() {
    $("source-modal").hidden = true;
  }

  function openDelete() {
    if (!state.data) return;
    const project = state.data.project;
    state.deleteTarget = { id: project.id, name: project.name };
    $("delete-title").textContent = `Excluir “${project.name}”?`;
    $("delete-warning").textContent = `Projeto: ${project.name} (${project.id}).`;
    $("delete-error").hidden = true;
    $("delete-confirm").disabled = false;
    $("delete-cancel").disabled = false;
    $("delete-confirm").textContent = "Excluir";
    $("delete-modal").hidden = false;
  }

  function closeDelete() {
    $("delete-modal").hidden = true;
    state.deleteTarget = null;
  }

  function renderSettings(status) {
    const mode = $("settings-mode");
    if (status.mode === "deepseek") {
      mode.textContent = "DeepSeek";
      mode.className = "badge deepseek";
    } else {
      mode.textContent = "Modo local";
      mode.className = "badge local";
    }
    const source = status.source === "panel"
      ? "chave salva no painel"
      : status.source === "environment"
        ? "chave de DEEPSEEK_API_KEY"
        : "nenhuma chave configurada";
    const environment = status.environment_available && status.source !== "environment"
      ? ` · DEEPSEEK_API_KEY disponível como alternativa`
      : "";
    $("settings-source").textContent = source + environment;
    $("settings-model").value = status.model || "deepseek-flash";
    $("settings-model-version").textContent = status.model_version
      ? `Identificador: ${status.model} · ${status.model_version}`
      : `Identificador configurado: ${status.model || "deepseek-flash"}`;
    $("settings-remove").disabled = !status.has_saved_key;
    $("settings-key").value = "";
  }

  async function loadSettings() {
    try {
      const status = await api("/api/settings");
      renderSettings(status);
    } catch (error) {
      showError(error.message);
    }
  }

  function settingsMessage(message, isError = false) {
    const box = $("settings-message");
    box.textContent = message || "";
    box.className = `alert ${isError ? "error" : "notice"}`;
    box.hidden = !message;
  }

  function openSettings() {
    settingsMessage("");
    $("settings-modal").hidden = false;
    loadSettings();
    $("settings-key").focus();
  }

  function closeSettings() {
    $("settings-modal").hidden = true;
    $("settings-key").value = "";
  }

  async function saveSettings(event) {
    event.preventDefault();
    const key = $("settings-key").value.trim();
    if (!key) {
      settingsMessage("Informe uma chave para salvar ou substituir.", true);
      return;
    }
    $("settings-save").disabled = true;
    $("settings-test").disabled = true;
    try {
      const status = await api("/api/settings/key", {
        method: "POST",
        body: JSON.stringify({ api_key: key, model: $("settings-model").value.trim() }),
      });
      renderSettings(status);
      settingsMessage("Chave salva. O modo DeepSeek vale para as próximas mensagens.");
      await loadProjects();
    } catch (error) {
      settingsMessage(error.message, true);
    } finally {
      $("settings-save").disabled = false;
      $("settings-test").disabled = false;
    }
  }

  async function testSettings() {
    $("settings-test").disabled = true;
    settingsMessage("Testando conexão…");
    try {
      const result = await api("/api/settings/test", {
        method: "POST",
        body: "{}",
        timeout: 25000,
      });
      if (result.ok) {
        const origin = result.source === "panel"
          ? "chave do painel"
          : result.source === "environment"
            ? "DEEPSEEK_API_KEY"
            : "sem chave";
        settingsMessage(
          `${result.message || "Conexão OK."} (${origin}${result.model ? `, modelo ${result.model}` : ""})`
        );
      } else {
        settingsMessage(result.error || "Falha ao testar a conexão.", true);
      }
      await loadSettings();
    } catch (error) {
      settingsMessage(error.message, true);
    } finally {
      $("settings-test").disabled = false;
    }
  }

  async function removeSettings() {
    if (!confirm("Remover a chave salva? Se houver DEEPSEEK_API_KEY, ela volta a ser usada.")) {
      return;
    }
    $("settings-remove").disabled = true;
    try {
      const status = await api("/api/settings/remove", { method: "POST", body: "{}" });
      renderSettings(status);
      settingsMessage(
        status.source === "environment"
          ? "Chave salva removida. O Farol passou a usar DEEPSEEK_API_KEY."
          : "Chave salva removida. O Farol está em modo local."
      );
      await loadProjects();
    } catch (error) {
      settingsMessage(error.message, true);
      $("settings-remove").disabled = false;
    }
  }

  async function confirmDelete() {
    if (!state.deleteTarget || state.busy) return;
    const target = state.deleteTarget;
    $("delete-confirm").disabled = true;
    $("delete-cancel").disabled = true;
    $("delete-confirm").textContent = "Excluindo…";
    $("delete-error").hidden = true;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(target.id)}/delete`, {
        method: "POST",
        body: JSON.stringify({ confirm_id: target.id, confirm_name: target.name }),
      });
      closeDelete();
      await loadProjects(data.active || null);
    } catch (error) {
      $("delete-error").textContent = error.message;
      $("delete-error").hidden = false;
      $("delete-confirm").disabled = false;
      $("delete-cancel").disabled = false;
      $("delete-confirm").textContent = "Excluir";
    }
  }

  $("new-project-button").addEventListener("click", () => {
    const form = $("create-form");
    form.hidden = !form.hidden;
    if (!form.hidden) $("project-name").focus();
  });
  $("cancel-create").addEventListener("click", () => {
    $("create-form").hidden = true;
    $("project-name").value = "";
  });

  $("create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("project-name").value.trim();
    if (!name || state.busy) return;
    setBusy(true, "Criando…");
    try {
      const data = await api("/api/projects", { method: "POST", body: JSON.stringify({ name }) });
      const id = data.project.id;
      await api(`/api/projects/${encodeURIComponent(id)}/select`, { method: "POST", body: "{}" });
      $("project-name").value = "";
      $("create-form").hidden = true;
      await loadProjects(id);
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(false);
    }
  });

  $("project-select").addEventListener("change", (event) => switchProject(event.target.value));
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage($("message-input").value);
  });
  $("resume-button").addEventListener("click", () => sendMessage("Onde paramos?"));
  $("brief-button").addEventListener("click", () =>
    sendMessage("Prepare instruções para o OpenCode sobre a tarefa atual.")
  );
  $("repo-save").addEventListener("click", saveRepo);
  $("semantic-toggle").addEventListener("change", saveSemantic);
  $("code-import").addEventListener("click", importCode);
  $("code-close").addEventListener("click", closeCode);
  $("message-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage($("message-input").value);
    }
  });

  $("delete-project").addEventListener("click", openDelete);
  $("delete-confirm").addEventListener("click", confirmDelete);
  $("delete-cancel").addEventListener("click", closeDelete);
  $("delete-close").addEventListener("click", closeDelete);

  $("settings-button").addEventListener("click", openSettings);
  $("settings-form").addEventListener("submit", saveSettings);
  $("settings-test").addEventListener("click", testSettings);
  $("settings-remove").addEventListener("click", removeSettings);
  $("settings-close").addEventListener("click", closeSettings);

  document.addEventListener("click", (event) => {
    const source = event.target.closest("[data-source-id]");
    if (source) {
      event.preventDefault();
      openSource(source.dataset.sourceId);
      return;
    }
    if (event.target.closest("[data-close-modal]")) closeSource();
    if (event.target.closest("[data-close-delete]")) {
      if (!state.busy) closeDelete();
    }
    if (event.target.closest("[data-close-settings]")) closeSettings();
    if (event.target.closest("[data-close-code]")) closeCode();
    const code = event.target.closest("[data-code-id]");
    if (code) {
      event.preventDefault();
      openCode(code.dataset.codeId);
      return;
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSource();
      closeSettings();
      if (!state.busy) closeDelete();
    }
  });

  loadSettings();
  loadProjects();
})();
