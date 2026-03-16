import re
import uuid
from typing import List, Optional
from dataclasses import dataclass

# ==============================
# DATA STRUCTURE
# ==============================

@dataclass
class LegalNode:
    node_id: str
    node_type: str

    article: Optional[str]
    clause: Optional[str]
    point: Optional[str]

    text: str
    parent_id: Optional[str]

    # amendment metadata
    action: Optional[str] = None
    target_article: Optional[str] = None
    target_clause: Optional[str] = None
    target_point: Optional[str] = None

# ==============================
# REGEX
# ==============================

ARTICLE_RE = re.compile(r"^Điều\s+(\d+)[\.:]?", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^(\d+)\.\s+")
POINT_RE = re.compile(r"^([a-z])\)\s+")
# amendment
AMENDMENT_RE = re.compile(
    r"(Sửa đổi|Bổ sung|Bãi bỏ|Thay thế)", re.IGNORECASE
)
TARGET_ARTICLE_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
TARGET_CLAUSE_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
TARGET_POINT_RE = re.compile(r"điểm\s+([a-z])", re.IGNORECASE)

# ==============================
# UTILS
# ==============================

def new_id():
    return str(uuid.uuid4())

def detect_target(text: str):

    article = None
    clause = None
    point = None

    m = TARGET_ARTICLE_RE.search(text)
    if m:
        article = f"Điều {m.group(1)}"

    m = TARGET_CLAUSE_RE.search(text)
    if m:
        clause = f"Khoản {m.group(1)}"

    m = TARGET_POINT_RE.search(text)
    if m:
        point = f"Điểm {m.group(1)}"

    return article, clause, point

# ==============================
# MAIN PARSER
# ==============================

def parse_legal_text(text: str) -> List[LegalNode]:

    nodes: List[LegalNode] = []

    current_article = None
    current_clause = None
    current_point = None

    article_id = None
    clause_id = None
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:

        # ---------------------------
        # ARTICLE
        # ---------------------------
        m = ARTICLE_RE.match(line)
        if m:
            current_article = f"Điều {m.group(1)}"
            current_clause = None
            current_point = None

            article_id = new_id()

            nodes.append(
                LegalNode(
                    node_id=article_id,
                    node_type="article",
                    article=current_article,
                    clause=None,
                    point=None,
                    text=line,
                    parent_id=None,
                )
            )

            continue

        # ---------------------------
        # CLAUSE
        # ---------------------------
        m = CLAUSE_RE.match(line)
        if m:
            current_clause = f"Khoản {m.group(1)}"
            current_point = None

            clause_id = new_id()

            nodes.append(
                LegalNode(
                    node_id=clause_id,
                    node_type="clause",
                    article=current_article,
                    clause=current_clause,
                    point=None,
                    text=line,
                    parent_id=article_id,
                )
            )

            continue

        # ---------------------------
        # POINT
        # ---------------------------
        m = POINT_RE.match(line)
        if m:
            current_point = f"Điểm {m.group(1)}"

            nodes.append(
                LegalNode(
                    node_id=new_id(),
                    node_type="point",
                    article=current_article,
                    clause=current_clause,
                    point=current_point,
                    text=line,
                    parent_id=clause_id,
                )
            )

            continue

        # ---------------------------
        # AMENDMENT
        # ---------------------------
        if AMENDMENT_RE.search(line):
            action = AMENDMENT_RE.search(line).group(1)

            target_article, target_clause, target_point = detect_target(line)

            nodes.append(
                LegalNode(
                    node_id=new_id(),
                    node_type="amendment",
                    article=current_article,
                    clause=current_clause,
                    point=None,
                    text=line,
                    parent_id=clause_id,
                    action=action,
                    target_article=target_article,
                    target_clause=target_clause,
                    target_point=target_point,
                )
            )

            continue

        # ---------------------------
        # NORMAL TEXT
        # ---------------------------
        parent = clause_id if clause_id else article_id

        nodes.append(
            LegalNode(
                node_id=new_id(),
                node_type="text",
                article=current_article,
                clause=current_clause,
                point=current_point,
                text=line,
                parent_id=parent,
            )
        )

    return nodes

# ==============================
# CHUNK BUILDER
# ==============================

def build_chunks(nodes: List[LegalNode]):

    chunks = []

    for node in nodes:

        if node.node_type in ["clause", "point"]:

            chunk_text = node.text

            metadata = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "article": node.article,
                "clause": node.clause,
                "point": node.point,
                "action": node.action,
                "target_article": node.target_article,
                "target_clause": node.target_clause,
                "target_point": node.target_point,
            }

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata
                }
            )

    return chunks