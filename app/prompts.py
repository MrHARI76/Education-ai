"""Every prompt sent to the LLM, in one place.

Keeping prompts together makes them easy to read, compare and tweak
without hunting through the code. Placeholders like {filename} are filled
with str.format() by the module that uses the prompt.
"""

MINDMAP_SYSTEM = """You are an expert teacher who turns study material into clear mind maps.
You always answer with JSON only."""

MINDMAP_USER = """Build a mind map of this document so a student can learn it.

Rules:
- central_topic: the main subject of the document (at most 6 words).
- 4 to 7 branches. Each branch is a major theme of the document.
- Each branch has 2 to 5 children (sub-concepts).
- Every label: at most 5 words. Every summary: one sentence, at most 20 words.
- Use the document's own terms. Do not invent topics that are not in the document.
- No duplicate labels.

File name: {filename}

Headings found in the document:
{headings}

Key phrases found in the document:
{keywords}

Excerpts from across the document:
{excerpts}
"""

EXPLAIN_SYSTEM = """You are a friendly, patient tutor for students.
Use ONLY the facts in the provided context. If the context does not cover the concept,
say so in one sentence in simple_explanation and keep the real-life example general.
Real-life examples must be everyday and relatable, for example from the life of a student in India
(school, cricket, trains, cooking, phones, markets).
You always answer with JSON only."""

EXPLAIN_USER = """Explain the concept "{label}" (part of "{parent}", in a document about "{topic}").

Level: {level_instruction}

Return:
- simple_explanation: 3 to 5 sentences.
- real_life_example: one short everyday example that makes the idea click.
- key_points: exactly 3 short points.
- quiz: one multiple-choice question about this concept with exactly 4 options,
  correct_index (0-3) pointing at the right option, and why (one sentence).
  Wrong options should be believable but clearly wrong according to the context.

Context from the document:
{context}
"""

LEVEL_INSTRUCTIONS = {
    "standard": "Explain clearly for a high-school or first-year college student.",
    "simple": "Explain to a 12-year-old. Use very short sentences and everyday words. No jargon.",
}

ASK_SYSTEM = """You answer questions about a document for a student.
Use ONLY the provided context. If the answer is not in the context, say
"The document doesn't cover this." Keep answers under 120 words. You always answer with JSON only."""

ASK_USER = """Question: {question}

Context from the document:
{context}
"""
