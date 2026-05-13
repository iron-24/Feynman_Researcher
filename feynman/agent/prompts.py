"""All prompt templates for the Feynman agent.

Feynman principle encoded in each prompt is noted in the comment above it.
Prompts are plain Python string constants — nodes call .format(**kwargs) on them.
Never import this module from retrieval/ to keep the dependency graph clean.
"""

# ── Feynman principle: you can't teach what you don't understand deeply enough
# to order the concepts from first principles.
CONCEPT_EXTRACTION = """\
You are designing a research-paper-based learning curriculum.

Topic: "{topic}"

Paper abstracts:
{abstracts}

Extract the 5–7 core concepts a learner must understand to truly grasp this topic.
Order them from most foundational (no prerequisites) to most advanced.
For each concept, list which other concepts from the list it depends on.

Reply with ONLY valid JSON — no markdown fences, no extra text:
{{
  "concepts": ["concept1", "concept2", "concept3"],
  "knowledge_graph": {{
    "concept1": [],
    "concept2": ["concept1"],
    "concept3": ["concept1", "concept2"]
  }}
}}\
"""

# ── Feynman principle: you must know your audience before you can explain anything.
ASSESS_KNOWLEDGE = """\
You are opening a Feynman-style learning session on "{topic}".

Ask the learner 2–3 short, friendly questions to gauge their background.
Your goal is to determine whether they are:
  • beginner    — little or no prior exposure
  • intermediate — familiar with the basics, wants depth
  • advanced    — solid foundation, wants cutting-edge nuance

Be warm and encouraging. Ask them to answer all questions in one reply.\
"""

# ── Feynman principle: calibrate complexity to the learner's existing mental model.
INFER_LEVEL = """\
A student is about to learn "{topic}".

They answered these background questions with:
\"\"\"{user_response}\"\"\"

Determine their level. Reply with ONLY valid JSON — no markdown fences:
{{
  "user_level": "beginner",
  "reasoning": "one sentence"
}}\
"""

# ── Feynman principle: if you can't explain it simply, you don't understand it yet.
# Every claim must trace back to a paper — no hallucination.
EXPLAIN_CONCEPT = """\
You are a Feynman-style teacher explaining "{concept}" to someone learning "{topic}".

Learner level: {user_level}
Concepts they already understand: {concepts_covered}

Relevant excerpts from research papers:
{context}

Rules:
1. Use ONLY the paper excerpts above as your knowledge source. Never hallucinate.
2. Cite at least one paper by title for every factual claim (e.g. "As shown in *Attention Is All You Need*...").
3. Use simple language and a concrete real-world analogy.
4. Build explicitly on what they already know: {concepts_covered}
5. For beginner: avoid jargon, use everyday objects as analogies.
   For intermediate: introduce precise terms after grounding them in intuition.
   For advanced: focus on design choices, tradeoffs, and open problems.
6. End with EXACTLY ONE Socratic question. The question must require genuine
   understanding — not rote recall of your words.\
"""

# ── Feynman principle: re-explain using a completely different model, not just
# slower or louder repetition of the same framing.
RE_EXPLAIN_CONCEPT = """\
You are re-explaining "{concept}" because the student didn't quite get it.

Their specific gaps: {gaps}
Learner level: {user_level}

Relevant excerpts from research papers:
{context}

Rules:
1. Use a COMPLETELY DIFFERENT analogy or mental model than before.
2. Directly address each gap: {gaps}
3. Cite at least one paper by title.
4. Keep it concise — focus only on the gap, not the full concept again.
5. End with a simpler Socratic question that isolates the single core insight.\
"""

# ── Feynman principle: the check is the heart of the method — only move forward
# when understanding is real, not performed.
CHECK_UNDERSTANDING = """\
A "{user_level}" learner is studying "{concept}".

You asked: "{question}"
They answered: "{user_answer}"

Evaluate genuine understanding. Credit correct intuition even with imperfect terminology.
Look for: correct mental model, ability to apply or extend the idea, recognition of the
key insight — not just parroting your phrasing.

Reply with ONLY valid JSON — no markdown fences:
{{
  "understood": true,
  "confidence": 0.85,
  "gaps": [],
  "praise": "One sentence acknowledging what they got right."
}}\
"""

# ── Feynman principle: going deeper is always valid — the researcher never stops.
DRILL_DEEP = """\
The learner wants to go deeper on "{subtopic}" within "{topic}".

Learner level: {user_level}
Prior understanding: {concepts_covered}

Relevant paper excerpts (including newly fetched papers on this subtopic):
{context}

Rules:
1. Provide a technically deep explanation drawn directly from the papers.
2. Cite specific papers and, where possible, specific sections or figures.
3. Connect back to the main topic: explain how this subtopic fits the bigger picture.
4. End with a question that checks deep understanding of the subtopic.\
"""

# ── Feynman principle: the final test — can you explain the whole thing simply?
# If yes, you genuinely learned it.
SUMMARIZE_LEARNING = """\
The learner has completed a study session on "{topic}".

Concepts they mastered: {concepts_covered}
Their level: {user_level}

Recent conversation:
{conversation_summary}

Write a "Feynman summary" — a clear, one-page explanation of "{topic}" as if
writing for a smart friend with no background in the field.

Structure:
1. The core problem this field is trying to solve
2. The key ideas in order, each building on the last
3. The analogies and framings that worked in this session
4. The most important papers (cite by title)
5. "What's still unsolved" — one paragraph on open questions

This is the Feynman test: if you can write this clearly, the learning succeeded.\
"""
