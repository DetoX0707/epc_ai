from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pdfplumber
except Exception:  # pragma: no cover - graceful runtime fallback
    pdfplumber = None


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./_-]*")
SPEC_VALUE_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9 /()_-]{2,80}?)\s*[:=-]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kW|MW|kVA|V|VAC|minutes|min|hours|hrs|C|deg C|kg|mm|m3/h|L/s|%|dB|N\+1|N\+2)?",
    re.IGNORECASE,
)

AUDIT_PARAM_KEYS = {
    "ambient_temperature",
    "leaving_water_temperature",
    "approach_temperature",
    "motor_power",
    "noise_level",
    "backup_runtime",
    "efficiency",
    "weight",
    "fuel_autonomy",
    "step_load_acceptance",
}


@dataclass
class Chunk:
    id: str
    doc_name: str
    page: int
    section: str
    text: str
    tokens: List[str] = field(default_factory=list)
    vector: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % buckets


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec
    return vec / norm


def hashed_vector(tokens: Iterable[str], buckets: int = 384) -> np.ndarray:
    vec = np.zeros(buckets, dtype=np.float32)
    token_list = list(tokens)
    for token in token_list:
        vec[stable_hash(token, buckets)] += 1.0
        if len(token) >= 5:
            for idx in range(len(token) - 2):
                gram = token[idx : idx + 3]
                vec[stable_hash("char:" + gram, buckets)] += 0.25
    return normalize_vector(vec)


def split_sections(text: str) -> List[Tuple[str, str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections: List[Tuple[str, List[str]]] = []
    current_title = "General"
    current_lines: List[str] = []

    heading_re = re.compile(r"^(\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 /&()_-]{4,80}$")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue
        is_heading = bool(heading_re.match(stripped)) and len(stripped.split()) <= 10
        if is_heading and current_lines:
            sections.append((current_title, current_lines))
            current_title = stripped
            current_lines = []
        elif is_heading and not current_lines:
            current_title = stripped
        else:
            current_lines.append(stripped)

    if current_lines:
        sections.append((current_title, current_lines))

    output: List[Tuple[str, str]] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            output.append((title, body))
    return output or [("General", text.strip())]


def make_chunks(doc_name: str, page: int, section: str, text: str, target_words: int = 130) -> List[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks: List[Chunk] = []
    for start in range(0, len(words), target_words):
        part = " ".join(words[start : start + target_words])
        chunk_id = hashlib.sha1(f"{doc_name}|{page}|{section}|{start}|{part}".encode("utf-8")).hexdigest()[:16]
        chunks.append(Chunk(id=chunk_id, doc_name=doc_name, page=page, section=section, text=part))
    return chunks


def extract_text_by_page(path: Path) -> List[Tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not available in this Python runtime.")
        pages: List[Tuple[int, str]] = []
        with pdfplumber.open(str(path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                pages.append((idx, page.extract_text() or ""))
        return pages

    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


class HybridKnowledgeBase:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 1.0

    def clear(self) -> None:
        self.chunks.clear()
        self.doc_freq.clear()
        self.avg_doc_len = 1.0

    def ingest_file(self, path: Path) -> int:
        added = 0
        for page, page_text in extract_text_by_page(path):
            for section, body in split_sections(page_text):
                for chunk in make_chunks(path.name, page, section, body):
                    chunk.tokens = tokenize(chunk.text)
                    chunk.vector = hashed_vector(chunk.tokens)
                    self.chunks.append(chunk)
                    added += 1
        self._recompute_stats()
        return added

    def ingest_text(self, doc_name: str, text: str) -> int:
        added = 0
        for section, body in split_sections(text):
            for chunk in make_chunks(doc_name, 1, section, body):
                chunk.tokens = tokenize(chunk.text)
                chunk.vector = hashed_vector(chunk.tokens)
                self.chunks.append(chunk)
                added += 1
        self._recompute_stats()
        return added

    def _recompute_stats(self) -> None:
        self.doc_freq = Counter()
        lengths = []
        for chunk in self.chunks:
            unique = set(chunk.tokens)
            self.doc_freq.update(unique)
            lengths.append(len(chunk.tokens))
        self.avg_doc_len = float(sum(lengths) / max(1, len(lengths)))

    def bm25_score(self, query_tokens: List[str], chunk: Chunk) -> float:
        if not chunk.tokens:
            return 0.0
        token_counts = Counter(chunk.tokens)
        n_docs = max(1, len(self.chunks))
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * len(chunk.tokens) / self.avg_doc_len)
            score += idf * (freq * (k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_tokens = tokenize(query)
        query_vec = hashed_vector(query_tokens)
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for chunk in self.chunks:
            dense = float(np.dot(query_vec, chunk.vector)) if chunk.vector is not None else 0.0
            sparse = self.bm25_score(query_tokens, chunk)
            dense_scores[chunk.id] = dense
            sparse_scores[chunk.id] = sparse

        dense_ranked = sorted(self.chunks, key=lambda c: dense_scores[c.id], reverse=True)
        sparse_ranked = sorted(self.chunks, key=lambda c: sparse_scores[c.id], reverse=True)
        fused: Dict[str, float] = defaultdict(float)
        rank_constant = 60.0

        for rank, chunk in enumerate(dense_ranked, start=1):
            fused[chunk.id] += 1.0 / (rank_constant + rank)
        for rank, chunk in enumerate(sparse_ranked, start=1):
            fused[chunk.id] += 1.0 / (rank_constant + rank)

        by_id = {chunk.id: chunk for chunk in self.chunks}
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [
            SearchResult(by_id[cid], fused[cid], dense_scores[cid], sparse_scores[cid])
            for cid in ranked_ids
            if dense_scores[cid] > 0 or sparse_scores[cid] > 0
        ]

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.search(query, top_k=top_k)
        if not results:
            return {
                "answer": "I could not find matching evidence in the indexed project corpus.",
                "citations": [],
                "results": [],
            }

        bullets = []
        citations = []
        for idx, result in enumerate(results[:3], start=1):
            chunk = result.chunk
            snippet = best_sentence(query, chunk.text)
            citation = {
                "doc": chunk.doc_name,
                "page": chunk.page,
                "section": chunk.section,
                "snippet": snippet,
                "score": round(result.score, 4),
            }
            citations.append(citation)
            bullets.append(f"{idx}. {snippet} [{chunk.doc_name}, Page {chunk.page}, {chunk.section}]")

        answer_text = "Based on the retrieved project evidence:\n" + "\n".join(bullets)
        return {
            "answer": answer_text,
            "citations": citations,
            "results": [result_to_dict(result) for result in results],
        }


def best_sentence(query: str, text: str) -> str:
    query_terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:300]
    scored = sorted(
        sentences,
        key=lambda s: (
            len(query_terms.intersection(tokenize(s))),
            bool(re.search(r"\d", s)),
            len(s),
        ),
        reverse=True,
    )
    selected = []
    for sentence in scored:
        clean = sentence.strip()
        if clean and clean not in selected:
            selected.append(clean)
        if len(selected) == 2:
            break
    return " ".join(selected)[:520]


def result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return {
        "doc": result.chunk.doc_name,
        "page": result.chunk.page,
        "section": result.chunk.section,
        "text": result.chunk.text,
        "score": round(result.score, 5),
        "dense_score": round(result.dense_score, 5),
        "sparse_score": round(result.sparse_score, 5),
    }


def extract_parameters(text: str) -> Dict[str, Dict[str, Any]]:
    params: Dict[str, Dict[str, Any]] = {}
    for match in SPEC_VALUE_RE.finditer(text):
        raw_label = " ".join(match.group("label").split()).strip(" -")
        label = normalize_label(raw_label)
        value = float(match.group("value"))
        unit = (match.group("unit") or "").strip()
        params[label] = {
            "label": raw_label,
            "value": value,
            "unit": unit,
            "evidence": match.group(0).strip(),
        }
    return params


def normalize_label(label: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    synonyms = {
        "design ambient": "ambient_temperature",
        "ambient temperature": "ambient_temperature",
        "leaving water temperature": "leaving_water_temperature",
        "approach temperature": "approach_temperature",
        "motor power": "motor_power",
        "fan motor power": "motor_power",
        "noise level": "noise_level",
        "sound pressure": "noise_level",
        "backup runtime": "backup_runtime",
        "runtime at full load": "backup_runtime",
        "efficiency": "efficiency",
        "ups efficiency": "efficiency",
        "weight": "weight",
        "operating weight": "weight",
        "fuel autonomy": "fuel_autonomy",
        "step load acceptance": "step_load_acceptance",
    }
    for key, value in synonyms.items():
        if key in clean:
            return value
    return clean.replace(" ", "_")


def requirement_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    params = extract_parameters(text)
    requirements: Dict[str, Dict[str, Any]] = {}
    for key, data in params.items():
        if key not in AUDIT_PARAM_KEYS:
            continue
        sentence = sentence_containing(text, data["evidence"])
        comparator = infer_comparator(sentence)
        requirements[key] = {
            **data,
            "comparator": comparator,
            "requirement_text": sentence,
        }
    return requirements


def sentence_containing(text: str, needle: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        if needle in sentence:
            return sentence.strip()
    return needle


def infer_comparator(sentence: str) -> str:
    s = sentence.lower()
    if any(term in s for term in ["minimum", "min.", "not less than", "at least", ">="]):
        return ">="
    if any(term in s for term in ["maximum", "max.", "not exceed", "no more than", "<="]):
        return "<="
    return "=="


def compare_value(vendor_value: float, required_value: float, comparator: str, tolerance: float = 0.001) -> bool:
    if comparator == ">=":
        return vendor_value + tolerance >= required_value
    if comparator == "<=":
        return vendor_value <= required_value + tolerance
    return abs(vendor_value - required_value) <= max(tolerance, abs(required_value) * 0.03)


def audit_vendor_submittal(vendor_text: str, kb: HybridKnowledgeBase) -> Dict[str, Any]:
    vendor_params = extract_parameters(vendor_text)
    equipment_hint = infer_equipment_type(vendor_text)
    master_query = (
        f"{equipment_hint} master specification requirements runtime efficiency ambient temperature "
        "leaving water approach motor power noise level fuel autonomy step load BACnet stainless steel"
    )
    evidence_results = kb.search(master_query, top_k=5)
    equipment_terms = set(tokenize(equipment_hint))
    focused_results = [
        result
        for result in evidence_results
        if equipment_terms.intersection(tokenize(result.chunk.section + " " + result.chunk.text))
    ]
    equipment_chunks = [
        chunk
        for chunk in kb.chunks
        if equipment_terms.intersection(tokenize(chunk.section + " " + chunk.text))
    ]
    if not focused_results and evidence_results:
        focused_results = [evidence_results[0]]
    context_parts = [result.chunk.text for result in focused_results]
    seen_ids = {result.chunk.id for result in focused_results}
    for chunk in equipment_chunks:
        if chunk.id not in seen_ids:
            context_parts.append(chunk.text)
    master_context = "\n".join(context_parts)
    requirements = requirement_from_text(master_context)

    findings = []
    for key, req in requirements.items():
        if key not in vendor_params:
            findings.append(
                {
                    "parameter": readable_key(key),
                    "status": "Missing",
                    "severity": "High",
                    "vendor": "Not provided",
                    "required": requirement_display(req),
                    "evidence": req.get("requirement_text", ""),
                    "recommendation": "Ask vendor to resubmit with this parameter explicitly stated.",
                }
            )
            continue

        vendor = vendor_params[key]
        same_unit = unit_family(vendor.get("unit", "")) == unit_family(req.get("unit", ""))
        compliant = same_unit and compare_value(vendor["value"], req["value"], req.get("comparator", "=="))
        findings.append(
            {
                "parameter": readable_key(key),
                "status": "Compliant" if compliant else "Non-Compliant",
                "severity": "OK" if compliant else "Critical",
                "vendor": f"{vendor['value']:g} {vendor.get('unit', '')}".strip(),
                "required": requirement_display(req),
                "evidence": req.get("requirement_text", ""),
                "recommendation": "Accept." if compliant else remediation_for(key, req),
            }
            )

    findings.extend(textual_requirement_findings(equipment_hint, vendor_text, master_context))

    for key, vendor in vendor_params.items():
        if key not in requirements and key not in {"document_number", "revision"}:
            if key not in AUDIT_PARAM_KEYS:
                continue
            findings.append(
                {
                    "parameter": readable_key(key),
                    "status": "Informational",
                    "severity": "Info",
                    "vendor": f"{vendor['value']:g} {vendor.get('unit', '')}".strip(),
                    "required": "No matched master requirement found",
                    "evidence": vendor.get("evidence", ""),
                    "recommendation": "Route to engineering review if this parameter is contractually relevant.",
                }
            )

    critical = sum(1 for item in findings if item["status"] == "Non-Compliant")
    missing = sum(1 for item in findings if item["status"] == "Missing")
    compliant = sum(1 for item in findings if item["status"] == "Compliant")
    verdict = "Non-Compliant" if critical or missing else "Compliant"
    confidence = min(0.96, 0.55 + 0.08 * len(evidence_results) + 0.04 * len(findings))

    audit = {
        "verdict": verdict,
        "equipment": equipment_hint.title(),
        "confidence": round(confidence, 2),
        "summary": {
            "critical": critical,
            "missing": missing,
            "compliant": compliant,
            "total": len(findings),
        },
        "findings": findings,
        "citations": [result_to_dict(result) for result in (focused_results or evidence_results)[:3]],
    }
    audit["executive_report"] = build_executive_report(audit)
    return audit


def textual_requirement_findings(equipment: str, vendor_text: str, master_context: str) -> List[Dict[str, Any]]:
    vendor_lower = vendor_text.lower()
    master_lower = master_context.lower()
    checks: List[Tuple[str, str, str, str]] = []
    if equipment == "cooling tower":
        checks.extend(
            [
                (
                    "BACnet BMS Integration",
                    "bacnet",
                    "BACnet integration to the BMS is required.",
                    "Include BACnet gateway in the base package and factory-test the controls interface.",
                ),
                (
                    "Stainless Steel Basin",
                    "stainless steel",
                    "Cooling towers shall use stainless steel basin construction.",
                    "Revise construction to stainless steel basin or submit approved deviation.",
                ),
                (
                    "VFD Controlled Fans",
                    "vfd",
                    "Cooling towers shall include VFD-controlled fans.",
                    "Confirm VFD fan control and include it in the technical schedule.",
                ),
            ]
        )
    if equipment == "generator":
        checks.extend(
            [
                (
                    "EPMS Event Logs",
                    "event log",
                    "Generator control panels shall integrate with EPMS and support event logs.",
                    "Add EPMS integration and timestamped event log evidence.",
                ),
                (
                    "Continuous Power Rating",
                    "continuous power",
                    "Generator rating shall support continuous operation for critical load blocks.",
                    "Submit continuous power rating and derating calculation.",
                ),
            ]
        )

    findings = []
    for label, token, requirement, recommendation in checks:
        if token not in master_lower:
            continue
        is_optional = token in vendor_lower and "optional" in vendor_lower[max(0, vendor_lower.find(token) - 80) : vendor_lower.find(token) + 120]
        is_excluded = token in vendor_lower and "excluded" in vendor_lower[max(0, vendor_lower.find(token) - 80) : vendor_lower.find(token) + 120]
        compliant = token in vendor_lower and not is_optional and not is_excluded
        findings.append(
            {
                "parameter": label,
                "status": "Compliant" if compliant else "Non-Compliant",
                "severity": "OK" if compliant else "Critical",
                "vendor": "Included" if compliant else "Missing, optional, or excluded",
                "required": "Mandatory",
                "evidence": requirement,
                "recommendation": "Accept." if compliant else recommendation,
            }
        )
    return findings


def build_executive_report(audit: Dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    critical = summary.get("critical", 0)
    compliant = summary.get("compliant", 0)
    total = summary.get("total", 0)
    lines = [
        f"Executive QA Summary - {audit.get('equipment', 'Equipment')}",
        "",
        f"Verdict: {audit.get('verdict', 'Unknown')} with {critical} critical deviation(s), {compliant} compliant check(s), and {total} total automated checks.",
        "Business impact: The workflow converts a manual 3-4 hour engineering cross-check into a cited automated review that can be completed during procurement intake.",
        "",
        "Top issues:",
    ]
    for item in audit.get("findings", [])[:5]:
        if item.get("status") == "Non-Compliant":
            lines.append(
                f"- {item['parameter']}: vendor submitted {item['vendor']} but master specification requires {item['required']}."
            )
    lines.extend(
        [
            "",
            "Recommended next action: block procurement release, return the submittal to the vendor for revision, and attach this report to the quality audit trail.",
        ]
    )
    return "\n".join(lines)


def polish_report_with_openai(audit: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "mode": "offline",
            "report": build_executive_report(audit),
            "note": "OPENAI_API_KEY is not set, so the offline executive report was used.",
        }

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        "You are writing a concise hackathon demo executive report for a data centre EPC quality manager. "
        "Use only the audit JSON. Include verdict, business risk, top deviations, and next action. "
        "Keep it under 180 words and make it sound procurement-ready.\n\n"
        f"AUDIT_JSON:\n{json.dumps(audit, indent=2)}"
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "mode": "offline",
            "report": build_executive_report(audit),
            "note": f"OpenAI polish was unavailable, so the offline report was used. Reason: {exc}",
        }

    text_parts: List[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(content["text"])
    polished = "\n".join(text_parts).strip() or build_executive_report(audit)
    return {"mode": "openai", "report": polished, "note": f"Generated with {model}."}


def infer_equipment_type(text: str) -> str:
    lower = text.lower()
    if "cooling tower" in lower:
        return "cooling tower"
    if "ups" in lower or "uninterruptible" in lower:
        return "ups"
    if "generator" in lower or "genset" in lower:
        return "generator"
    if "switchgear" in lower:
        return "switchgear"
    return "equipment"


def readable_key(key: str) -> str:
    return key.replace("_", " ").title()


def requirement_display(req: Dict[str, Any]) -> str:
    comp = req.get("comparator", "==")
    symbol = {"==": "=", ">=": ">=", "<=": "<="}.get(comp, comp)
    return f"{symbol} {req['value']:g} {req.get('unit', '')}".strip()


def remediation_for(key: str, req: Dict[str, Any]) -> str:
    return f"Revise submittal to meet master requirement: {readable_key(key)} {requirement_display(req)}."


def unit_family(unit: str) -> str:
    u = unit.lower().strip()
    if u in {"min", "minutes"}:
        return "minutes"
    if u in {"hrs", "hours"}:
        return "hours"
    if u in {"c", "deg c"}:
        return "temperature"
    if u in {"kw", "mw", "kva"}:
        return "power"
    if u in {"db"}:
        return "noise"
    if u in {"kg"}:
        return "weight"
    if u in {"%", "n+1", "n+2"}:
        return u
    return u


def load_sample_corpus(kb: HybridKnowledgeBase, sample_dir: Path) -> Dict[str, Any]:
    kb.clear()
    loaded = []
    for path in sorted(sample_dir.glob("*.txt")):
        if path.name.startswith("vendor_"):
            continue
        chunks = kb.ingest_file(path)
        loaded.append({"file": path.name, "chunks": chunks})
    return {"documents": loaded, "chunks": len(kb.chunks)}


def serialize_kb(kb: HybridKnowledgeBase) -> Dict[str, Any]:
    docs = sorted({chunk.doc_name for chunk in kb.chunks})
    return {"documents": docs, "chunks": len(kb.chunks)}
