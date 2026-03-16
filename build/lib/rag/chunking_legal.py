from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Bắt articlie vd Điều
ARTICLE_RE = re.compile(r"(?m)^(Điều\s+\d+)\s*[.:]?\s*(.*)$")

# Khoản: "1. ..." hoặc "(1) ..." hoặc "8a." (khoản chữ số + chữ)
CLAUSE_RE = re.compile(r"^\(?(\d+[a-z]?)\)?\s*[\.\)]\s+.+$",re.IGNORECASE)

# Điểm: "a) ..." (có đ)
POINT_START_RE = re.compile(r"^([a-zđ])\)\s+.+$", re.IGNORECASE)

@dataclass
class LegalChunk:
    text: str
    article: Optional[str]=None
    clause: Optional[str]=None
    point: Optional[str]=None

def _split_by_article(text: str) -> List[tuple[Optional[str],str]]:
    line = text.split("\n")
    idxs=[]
    labels=[]
    for i,l in enumerate(line):
        m=ARTICLE_RE.match(l.strip())
        if m:
            idxs.append(i)
            labels.append(m.group(1).strip())
    if not idxs:
        return [(None,text)]
    segs: List[tuple[Optional[str],str]] = []
    for k,start in enumerate(idxs):
        end = idxs[k+1] if k+1 < len(idxs) else len(line)
        label = labels[k]
        body = "\n".join(line[start:end]).strip()
        segs.append((label,body))
    return segs

def _split_by_clause(block:str) -> List[tuple[Optional[str],str]]:
    lines = block.split("\n")
    starts = []
    nums = []
    for i,l in enumerate(lines):
        m = CLAUSE_RE.match(l.strip())
        if m:
            starts.append(i)
            nums.append(m.group(1).strip())
    if not starts:
        return [(None,block)]
    out : List[tuple[Optional[str],str]] = []
    for k,start in enumerate(starts):
        end = starts[k+1] if k+1 < len(starts) else len(lines)
        num = nums[k]
        label = f"Khoản {num}"
        out.append((label, "\n".join(lines[start:end]).strip()))
    return out

def _split_by_point(block: str) -> List[tuple[Optional[str], str]]:
    lines = block.split("\n")
    starts = []
    letters = []

    for i, line in enumerate(lines):
        if POINT_START_RE.match(line.strip()):
            m = re.match(r"^([a-zđ])\)", line.strip(), re.IGNORECASE)
            if m:
                starts.append(i)
                letters.append(m.group(1).lower())

    if not starts:
        return [(None, block.strip())]

    out: List[tuple[Optional[str], str]] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        letter = letters[k]
        label = f"Điểm {letter}"
        out.append((label, "\n".join(lines[start:end]).strip()))
    return out
# semantic + delimiter-based
def _fallback_split_by_size(text:str, max_chars: int, overlap:int) -> List[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    # Ưu tiên cắt theo câu
    sent = re.split(r"(?<=[\.\;\:])",text)
    chunks: List[str] = []
    curr = ""
    for s in sent:
        s= s.strip()
        if not s:
            continue
        if len(curr) + len(s) + 1 <= max_chars:
            curr = (curr + " " + s).strip()
        else:
            if curr:
                chunks.append(curr)
            curr = s
    if curr:
        chunks.append(curr)

    # Nế vẫn quá dài, ít dấu câu, cắt theo ký tự
    out = List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            out.append(chunk)
        else:
            i=0
            n=len(chunk)
            while i < n:
                end=min(i+max_chars, n)
                out.append(chunk[i:end].strip())
                i = end - overlap
                if i < 0:
                    i=0
                if i>=n:
                    break
    return [x for x in out if x]


def legal_chunk(text:str, max_chars: int = 1600, overlap: int=120) -> List[LegalChunk]:
    out: List[LegalChunk] = []
    for article_label, article_block in _split_by_article(text):
        for clause_label, clause_block in _split_by_clause(article_block):
            for point_label, point_block in _split_by_point(clause_block):
                segments = _fallback_split_by_size(point_block, max_chars, overlap)
                for seg in segments:
                    chunk = LegalChunk(
                        text=seg,
                        article=article_label,
                        clause=clause_label,
                        point=point_label
                    )
                    out.append(chunk)
    return [c for c in out if c.text.strip()]

# -------------------------
# Semantic merge "an toàn"
# - Chỉ merge nếu: cùng Điều + cùng Khoản
# -------------------------
def _cosine(a: List[float], b:List[float]) -> float:
    import math
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a ==0 or norm_b==0:
        return 0.0
    return dot / (norm_a * norm_b)

def semantic_merge_safe(chunks: List[LegalChunk], embeddings, min_chars: int = 350, sim_threshold: float = 0.88, max_merged_chars: int = 1600,) -> List[LegalChunk]:
    if len(chunks) <=1:
        return chunks
    texts = [c.text for c in chunks]
    vecs = embeddings.embed_documents(texts)

    merged = List[LegalChunk] = []
    i=0
    while i < len(chunks):
        curr = chunks[i]
        if i+1<len(chunks):
            next = chunks[i+1]
            same_unit = (curr.article == next.article) and (curr.clause == next.clause)
            both_short = (len(curr.text) < min_chars) and (len(next.text) < min_chars)
            fit = (len(curr.text) + len(next.text) <= max_merged_chars)
            if same_unit and both_short and fit:
                sim = _cosine(curr.text, next.text)
                if sim >= sim_threshold:
                    merged_chunk = LegalChunk(
                        text= curr.text + "\n" + next.text,
                        article= curr.article,
                        clause= curr.clause,
                        point= None
                    )
                    merged.append(merged_chunk)
                    i += 2
                    continue
        merged.append(curr)
        i+=1
    return merged



