"""Feynman — Gradio chat interface.

Run locally:
    python app.py

The layout is two columns:
  Left (70%)  — chat, action buttons, example topics
  Right (30%) — live sources panel + concept progress
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from feynman.agent.graph import feynman_graph
from feynman.agent.nodes import explain_node, set_api_key, summarize_node
from feynman.agent.state import AgentState

# ── Constants ──────────────────────────────────────────────────────────────────

EXAMPLE_TOPICS = [
    ["Vision Transformers"],
    ["Diffusion Models"],
    ["Reinforcement Learning from Human Feedback"],
    ["Graph Neural Networks"],
]

WELCOME = """👋 Welcome to **Feynman** — a research-paper-based tutor.

Type a topic below and I will:
1. Fetch and read the most relevant papers from ArXiv
2. Ask you a couple of questions to calibrate to your level
3. Teach you the topic from first principles, one concept at a time

I will check that you genuinely understood each concept before moving on — and re-explain with a different analogy if not.

**⏳ The first response takes 60–90 seconds** — the agent is downloading and parsing papers.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _papers_rows(state: Dict) -> List[List]:
    """Convert state.papers to rows for the Dataframe component."""
    papers = state.get("papers") or []
    rows = []
    for p in papers[:15]:
        year = p.published_date.year if getattr(p, "published_date", None) else "—"
        title = p.title if len(p.title) <= 52 else p.title[:49] + "..."
        link = f"https://arxiv.org/abs/{p.id}"
        rows.append([title, year, p.citation_count, link])
    return rows


def _progress_md(state: Dict) -> str:
    """Build the concept-progress markdown for the right panel."""
    ordered = state.get("concepts_ordered") or []
    covered = state.get("concepts_covered") or []
    current = state.get("current_concept") or ""
    level = state.get("user_level") or ""

    if not ordered:
        return "*Start a topic to see progress.*"

    filled = "█" * len(covered)
    empty = "░" * max(0, len(ordered) - len(covered))
    bar = f"`{filled}{empty}` {len(covered)}/{len(ordered)}"

    lines = [f"**Concepts** {bar}"]
    if level:
        lines.append(f"**Level** {level}")
    if current:
        lines.append(f"**Now** *{current}*")
    if covered:
        lines.append("**Done** " + " · ".join(f"~~{c}~~" for c in covered))
    return "\n\n".join(lines)


def _write_export(state: Dict) -> Optional[str]:
    """Serialize the session to a markdown file and return its path."""
    topic = state.get("topic", "session")
    history = state.get("conversation_history") or []
    papers = state.get("papers") or []
    covered = state.get("concepts_covered") or []

    lines = [
        f"# Feynman Session: {topic}",
        f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## Concepts covered",
        (", ".join(covered) if covered else "*(none)*") + "\n",
        "## Sources",
    ]
    for p in papers[:15]:
        lines.append(f"- [{p.title}](https://arxiv.org/abs/{p.id}) — {p.citation_count} citations")

    lines.append("\n## Conversation")
    for msg in history:
        label = "**You**" if msg["role"] == "user" else "**Feynman**"
        lines.append(f"\n{label}\n\n{msg['content']}\n")

    slug = topic.replace(" ", "_")[:40]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix=f"feynman_{slug}_"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


# ── Event handlers ─────────────────────────────────────────────────────────────

def respond(
    message: str,
    history: List,
    state: Dict,
    api_key: str = "",
) -> Generator[Tuple, None, None]:
    """Main chat handler.  Yields a loading placeholder, then the real response."""
    if api_key:
        set_api_key(api_key)

    message = (message or "").strip()
    if not message:
        yield history, state, _papers_rows(state), _progress_md(state), ""
        return

    phase = state.get("phase") or "start"

    # Immediate visual feedback while the graph runs
    if phase == "start":
        placeholder = (
            "🔍 *Fetching papers and building the knowledge base…*\n\n"
            "*(This takes 60–90 seconds — downloading and parsing PDFs)*"
        )
    else:
        placeholder = "💭 *Thinking…*"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": placeholder},
    ]
    yield history, state, _papers_rows(state), _progress_md(state), ""

    # Prepare state for this turn
    state = dict(state)
    state["user_message"] = message
    if phase == "start":
        state["topic"] = message
        state["phase"] = "start"

    try:
        result = feynman_graph.invoke(state)
        state.update(result)
        history[-1]["content"] = result.get("agent_response") or "*(no response)*"
    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ Something went wrong: `{exc}`\n\n"
            "Please try again or start a new topic."
        )

    yield history, state, _papers_rows(state), _progress_md(state), ""


def go_deeper(
    history: List, state: Dict
) -> Generator[Tuple, None, None]:
    """Trigger a drill on the current concept."""
    current = state.get("current_concept") or "the current concept"
    yield from respond(f"go deeper on {current}", history, state)


def skip_concept(history: List, state: Dict) -> Tuple:
    """Mark the current concept as covered and explain the next one."""
    current = state.get("current_concept") or ""
    covered = list(state.get("concepts_covered") or [])
    if current and current not in covered:
        covered.append(current)

    state = dict(state)
    state["concepts_covered"] = covered
    state["reexplain"] = False

    result = explain_node(state)

    # explain_node short-circuits to phase=summarizing when nothing is left
    if result.get("phase") == "summarizing":
        state.update(result)
        result = summarize_node(state)

    state.update(result)
    response = result.get("agent_response") or ""
    history = history + [
        {"role": "user", "content": "⏭ *Skipped to next concept*"},
        {"role": "assistant", "content": response},
    ]
    return history, state, _papers_rows(state), _progress_md(state)


def summarize_now(history: List, state: Dict) -> Tuple:
    """Generate a Feynman summary of everything covered so far."""
    state = dict(state)
    result = summarize_node(state)
    state.update(result)
    history = history + [
        {"role": "user", "content": "📋 *Summarize what I've learned so far*"},
        {"role": "assistant", "content": result.get("agent_response") or ""},
    ]
    return history, state, _papers_rows(state), _progress_md(state)


def new_topic() -> Tuple:
    """Reset to a fresh session."""
    return [{"role": "assistant", "content": WELCOME}], {}, [], "*Start a topic to see progress.*"


def export_session(state: Dict) -> gr.File:
    """Write session to disk and return a downloadable File component."""
    if not state.get("topic"):
        gr.Warning("Start a topic before exporting.")
        return gr.File(visible=False)
    path = _write_export(state)
    return gr.File(value=path, visible=True)


# ── Layout ─────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Feynman — Learn from Papers") as demo:

        agent_state = gr.State({})

        # Only shown when GOOGLE_API_KEY is not set in the environment (e.g. HuggingFace Spaces)
        _env_key_set = bool(os.getenv("GOOGLE_API_KEY"))
        api_key_input = gr.Textbox(
            label="Google API Key",
            placeholder="AIza... (get one free at aistudio.google.com/apikey)",
            type="password",
            visible=not _env_key_set,
        )

        gr.Markdown(
            "# Feynman\n"
            "*Learn any topic from first principles — knowledge sourced entirely from research papers*"
        )

        with gr.Row():

            # ── Left: chat ──────────────────────────────────────────────────
            with gr.Column(scale=7):

                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": WELCOME}],
                    height=520,
                    show_label=False,
                    render_markdown=True,
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a topic (e.g. 'Vision Transformers') or answer the question above…",
                        show_label=False,
                        scale=9,
                        autofocus=True,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=72)

                with gr.Row():
                    deeper_btn   = gr.Button("🔬 Go Deeper",       size="sm")
                    skip_btn     = gr.Button("⏭ Skip This",        size="sm")
                    summary_btn  = gr.Button("📋 Summarize So Far", size="sm")
                    new_btn      = gr.Button("🔄 New Topic",        size="sm", variant="stop")
                    export_btn   = gr.Button("💾 Save Session",     size="sm")

                export_file = gr.File(label="Session download", visible=False)

                gr.Examples(
                    examples=EXAMPLE_TOPICS,
                    inputs=[msg_box],
                    label="Example topics — click to load",
                )

            # ── Right: sources panel ────────────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### 📄 Sources")
                progress_md = gr.Markdown("*Start a topic to see progress.*")
                papers_table = gr.Dataframe(
                    headers=["Title", "Year", "Citations", "Link"],
                    datatype=["str", "number", "number", "str"],
                    wrap=True,
                    interactive=False,
                )

        # ── Wiring ────────────────────────────────────────────────────────────

        # Outputs shared by generator handlers (respond / go_deeper)
        stream_outs = [chatbot, agent_state, papers_table, progress_md, msg_box]
        # Outputs shared by non-generator handlers
        sync_outs   = [chatbot, agent_state, papers_table, progress_md]

        msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot, agent_state, api_key_input],
            outputs=stream_outs,
        )
        send_btn.click(
            fn=respond,
            inputs=[msg_box, chatbot, agent_state, api_key_input],
            outputs=stream_outs,
        )
        deeper_btn.click(
            fn=go_deeper,
            inputs=[chatbot, agent_state],
            outputs=stream_outs,
        )
        skip_btn.click(
            fn=skip_concept,
            inputs=[chatbot, agent_state],
            outputs=sync_outs,
        )
        summary_btn.click(
            fn=summarize_now,
            inputs=[chatbot, agent_state],
            outputs=sync_outs,
        )
        new_btn.click(
            fn=new_topic,
            outputs=sync_outs,
        )
        export_btn.click(
            fn=export_session,
            inputs=[agent_state],
            outputs=[export_file],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
