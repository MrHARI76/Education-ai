"""Build the mind map: with the LLM if possible, from headings + keywords if not.

The LLM produces a nested tree (central topic → branches → children).
The server then flattens it into a list of nodes with stable IDs
(n0, n1, …) and parent links, which is exactly what the graph library
needs. If the LLM fails, `fallback_mindmap` builds a simpler map without
any AI so the demo never shows a blank screen.
"""

import logging
import re
from pathlib import Path
from typing import Any

from app import prompts
from app.llm import chat_json
from app.models import LLMBranch, LLMChildNode, LLMMindMap, MindMap, MindMapNode

logger = logging.getLogger(__name__)

MAX_HEADINGS = 60
MAX_KEYWORDS = 30
NUM_EXCERPTS = 10
EXCERPT_CHARS = 400
MAX_BRANCHES = 7
MAX_CHILDREN = 5
MAX_LABEL_WORDS = 6


def pick_excerpts(chunks: list[dict[str, Any]], count: int = NUM_EXCERPTS) -> list[str]:
    """Pick `count` chunks evenly spread through the document (start, middle, end)."""
    if len(chunks) <= count:
        picked = chunks
    else:
        step = len(chunks) / count
        picked = [chunks[int(i * step)] for i in range(count)]
    return [c["text"][:EXCERPT_CHARS] for c in picked]


def build_prompt(filename: str, headings: list[str], keywords: list[str], chunks: list[dict[str, Any]]) -> str:
    """Fill the mind-map prompt, keeping it small enough for the context window."""
    excerpts = pick_excerpts(chunks)
    return prompts.MINDMAP_USER.format(
        filename=filename,
        headings="\n".join(f"- {h}" for h in headings[:MAX_HEADINGS]) or "(none)",
        keywords=", ".join(keywords[:MAX_KEYWORDS]) or "(none)",
        excerpts="\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(excerpts)),
    )


def short_label(text: str) -> str:
    """Trim a label to a few words so graph nodes stay readable."""
    words = text.strip().split()
    return " ".join(words[:MAX_LABEL_WORDS])


def flatten(tree: LLMMindMap, generated_by: str) -> dict[str, Any]:
    """Convert the nested tree into a flat node list with IDs and parent links.

    Also enforces size limits (max 7 branches, 5 children) and drops
    duplicate labels, because the LLM doesn't always follow instructions.
    """
    nodes = [MindMapNode(id="n0", label=short_label(tree.central_topic), summary="", parent_id=None, level=0)]
    seen = {nodes[0].label.lower()}

    def add(label: str, summary: str, parent_id: str, level: int) -> str | None:
        """Append a node unless its label is a duplicate; return the new ID."""
        label = short_label(label)
        if not label or label.lower() in seen:
            return None
        seen.add(label.lower())
        node_id = f"n{len(nodes)}"
        nodes.append(MindMapNode(id=node_id, label=label, summary=summary.strip(), parent_id=parent_id, level=level))
        return node_id

    for branch in tree.branches[:MAX_BRANCHES]:
        branch_id = add(branch.label, branch.summary, "n0", 1)
        if branch_id is None:
            continue
        for child in branch.children[:MAX_CHILDREN]:
            add(child.label, child.summary, branch_id, 2)
    return MindMap(central_topic=nodes[0].label, nodes=nodes, generated_by=generated_by).model_dump()


async def build_mindmap(
    filename: str, headings: list[str], keywords: list[str], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Ask the LLM for a mind map and return it flattened.

    Raises LLMError/LLMUnavailable on failure; the pipeline then uses the fallback.
    """
    tree = await chat_json(prompts.MINDMAP_SYSTEM, build_prompt(filename, headings, keywords, chunks), LLMMindMap)
    return flatten(tree, generated_by="llm")


def first_sentence_mentioning(term: str, chunks: list[dict[str, Any]], max_words: int = 20) -> str:
    """Find a sentence in the document containing `term`, to use as a summary."""
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            if pattern.search(sentence):
                words = sentence.split()
                return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")
    return "A key idea from the document."


def fallback_mindmap(
    filename: str, headings: list[str], keywords: list[str], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a mind map without the LLM.

    Branches = document headings (or top keywords if there are too few
    headings). Children = keywords that appear in that branch's text,
    then any leftovers shared round-robin. Always returns a usable map.
    """
    central = headings[0] if headings else Path(filename).stem.replace("_", " ").replace("-", " ")
    branch_labels = [h for h in headings[1:] if len(h.split()) <= 8][:MAX_BRANCHES]
    pool = [k for k in keywords if k.lower() not in {b.lower() for b in branch_labels}]
    if len(branch_labels) < 3:
        branch_labels = pool[:5]
        pool = pool[5:]

    children: dict[str, list[str]] = {b: [] for b in branch_labels}
    used: set[str] = set()
    for branch in branch_labels:
        text = " ".join(c["text"].lower() for c in chunks if c.get("heading") == branch)
        for kw in pool:
            if kw not in used and kw.lower() in text and len(children[branch]) < 3:
                children[branch].append(kw)
                used.add(kw)
    leftovers = iter(k for k in pool if k not in used)
    for branch in branch_labels:
        while len(children[branch]) < 2:
            kw = next(leftovers, None)
            if kw is None:
                break
            children[branch].append(kw)

    tree = LLMMindMap.model_construct(
        central_topic=central,
        branches=[
            LLMBranch.model_construct(
                label=b,
                summary=first_sentence_mentioning(b, chunks),
                children=[
                    LLMChildNode(label=k, summary=first_sentence_mentioning(k, chunks)) for k in children[b]
                ],
            )
            for b in branch_labels
        ],
    )
    return flatten(tree, generated_by="fallback")
