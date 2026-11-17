"""Spec generation: retrieved context in, validated and cited Spec out.

Two paths share one contract:
  * LLM path   - prompt -> JSON -> schema validation -> citation grounding, with a
                 repair loop that feeds validation errors back to the model.
  * Offline    - deterministic rule-based extraction, so the engine works with no model.

Either way, anything the engine cannot find in the notes becomes an open question
instead of a guess.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .gateway import LLMProvider, ProviderError
from .index import tokenize
from .schemas import Constraint, Entity, Feature, OpenQuestion, Role, Screen, SourceRef, Spec


def question_id(text: str) -> str:
    norm = " ".join(tokenize(text))
    return "Q-" + hashlib.sha1(norm.encode()).hexdigest()[:6]


@dataclass
class GenerationResult:
    spec: Spec
    repairs: int = 0
    warnings: list[str] = field(default_factory=list)


# =====================================================================
# Shared helpers
# =====================================================================

_BULLET = re.compile(r"^\s*(?:[-*•>]+|\d+[.)])\s*")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        line = _BULLET.sub("", line).strip()
        if not line or line.startswith("#"):
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", line):
            s = s.strip()
            if len(s.split()) >= 3:
                out.append(s)
    return out


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _ref(chunk: dict) -> SourceRef:
    return SourceRef(chunk_id=chunk["id"], artifact=chunk["filename"])


def _add_source(sources: list[SourceRef], chunk: dict) -> None:
    if all(s.chunk_id != chunk["id"] for s in sources):
        sources.append(_ref(chunk))


_TITLE_TRAIL = {
    "of", "who", "the", "a", "an", "for", "to", "with", "that", "and", "or", "in", "on", "at", "which", "their", "its",
    "has", "have", "is", "are", "was", "day",
}


def _title(text: str, words: int = 7) -> str:
    text = re.sub(r"^(to|be able to|able to|that|we|they|it|the system|the app|have)\s+", "", text.strip(), flags=re.I)
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.I)
    text = re.split(r",|;|:| so | \(| before | because | when | if | that ", text, maxsplit=1)[0]
    toks = re.findall(r"[A-Za-z0-9'/-]+", text)[:words]
    while toks and toks[-1].lower() in _TITLE_TRAIL:
        toks.pop()
    return " ".join(toks).capitalize() if toks else text[:40]


_UNSURE_PREFIX = re.compile(
    r"^(?:it is |it's |we are |we're )?(?:still )?(?:not sure|unclear|undecided|tbd)(?: yet)?[:,]?\s*(?:whether|if)?\s*", re.I
)


def _as_question(sentence: str) -> str:
    if sentence.endswith("?"):
        return sentence
    rest = _UNSURE_PREFIX.sub("", sentence).rstrip(".")
    if rest != sentence.rstrip("."):
        return f"Decide whether {rest}."
    return f"Decide: {rest}."


# =====================================================================
# Offline, rule-based extractor
# =====================================================================

ROLE_WORDS = [
    "front desk staff", "front desk", "receptionist", "patient", "nurse", "doctor", "physician", "clinician",
    "clinic manager", "manager", "administrator", "admin", "coordinator", "volunteer",
    "interpreter", "caregiver", "parent", "pharmacist", "social worker", "case manager", "technician",
    "analyst", "reviewer", "customer", "student", "teacher", "driver", "dispatcher",
]
_ROLE_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in ROLE_WORDS) + r")(?:e?s)?\b", re.I)
_GOAL_RE = re.compile(
    r"^\s*(?:staff\s+)?(?:needs? to|wants? to|should be able to|must be able to|can|has to|have to|would like to"
    r"|(?=wants? an? ))\s*(?:wants? )?(.+)",
    re.I,
)
_FIRST_PERSON_RE = re.compile(r"\bI (?:need|want|don't|do not|think)\b")
_REQ_RE = re.compile(
    r"\b(must|needs? to|has to|have to|required|should|wants? to|would like|be able to|could|nice to have|nice-to-have)\b",
    re.I,
)
_MUST = {"must", "need to", "needs to", "has to", "have to", "required"}
_COULD = {"could", "nice to have", "nice-to-have"}

_ENTITY_VERB_RE = re.compile(
    r"\b(?:track|store|record|capture|save|log|collect|manage)s?\s+"
    r"(?:the |a |an |all |each |every |their |our |patient |)"
    r"([a-z][a-z-]*(?: [a-z][a-z-]*){0,2}?)"
    r"(?=[,.;:)]| \(| and | for | with | in | on | so | from | by | per |$)",
    re.I,
)
_FIELDS_RE = re.compile(r"^([A-Za-z][\w ]{2,40}?)\s+(?:fields?|columns?|needs?|has|includes?|captures?)\s*[:\-]\s*(.+)$", re.I)
_SCREEN_RE = re.compile(
    r"\b((?:[A-Za-z][\w-]*\s+){1,3}?)(screen|page|dashboard|view|portal|kiosk)\b", re.I
)
_SCREEN_SKIP = {
    "a", "an", "the", "one", "simple", "single", "new", "main", "on", "in", "to", "of", "and", "our", "this", "that",
    "same", "each", "every", "their", "web", "wants", "want", "needs", "need", "see", "sees", "show", "shows", "has",
    "have", "at", "top", "with", "per", "is", "be",
}

CONSTRAINT_RULES: list[tuple[str, str]] = [
    (r"\b(hipaa|phi|gdpr|ferpa|retention|retain|consent|regulat\w*)\b", "compliance"),
    (r"\b(encrypt\w*|audit\w*|sso|log ?in|sign ?in|password|role-based|rbac|permission\w*|access control|mfa|2fa)\b", "security"),
    (r"\b(budget|deadline|launch|go-live|pilot|cost|by (?:january|february|march|april|may|june|july|august|september|october|november|december))\b", "business"),
    (r"\b(offline|air-gapped|on-prem\w*|accessib\w*|wcag|latency|uptime|internet|wifi|wi-fi|integrat\w*|sync\w*)\b", "technical"),
]
_UNCERTAIN_RE = re.compile(r"\b(tbd|not sure|unclear|unknown|undecided|to be decided|need to decide|open question|no idea|haven't decided)\b", re.I)

# Why an open question matters, by topic. The first matching rule wins, so the more
# specific topics (scope) come before broader ones (integration).
QUESTION_REASONS: list[tuple[str, str]] = [
    (r"\b(edit|change|allowed|permission|access|who can|who is allowed|approve|sign ?in|log ?in)\b",
     "Decides who gets view and edit rights, which shapes the roles, screens and audit trail."),
    (r"\b(belong|stay in|in scope|out of scope|this app or)\b",
     "Changes what this app is responsible for, and so its screens and data model."),
    (r"\b(how long|keep|retain|retention|delete|archive|years)\b",
     "Sets the data retention policy, which legal and compliance rules depend on."),
    (r"\b(ehr|integrat\w*|export|pdf|sync|api|import)\b",
     "An integration is usually the biggest scope and cost driver, so it needs a decision early."),
    (r"\b(host|aws|azure|gcp|cloud|server|on-prem\w*|deploy\w*)\b",
     "Decides where the app runs and which security controls and costs apply."),
    (r"\b(budget|cost|price|deadline|launch|pilot|timeline)\b",
     "Affects what fits in the first version's budget and timeline."),
    (r"\b(language|spanish|translat\w*|accessib\w*)\b",
     "Affects every screen and form, so it is cheaper to decide before design starts."),
]
DEFAULT_REASON = "The notes raise this but don't settle it, so the spec leaves it open."


def why_it_matters(question: str) -> str:
    for pattern, reason in QUESTION_REASONS:
        if re.search(pattern, question, re.I):
            return reason
    return DEFAULT_REASON


GAP_CHECKS = [
    (
        lambda text: not re.search(r"\b(log ?in|sign ?in|sso|password|role|permission|access)\b", text, re.I),
        "Who should be able to see and change which data, and how will people sign in?",
        "Access rules shape the data model and every screen.",
    ),
    (
        lambda text: not re.search(r"\b(cloud|on-prem\w*|server|host\w*|aws|azure|gcp|deploy\w*|kubernetes)\b", text, re.I),
        "Where will the app run: a cloud provider, on-premises servers, or offline devices?",
        "Hosting decides the deployment package and which security controls apply.",
    ),
]


def _role_name(raw: str) -> str:
    name = raw.lower()
    name = {"front desk": "front desk staff", "admin": "administrator", "physician": "doctor"}.get(name, name)
    return name.title()


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


_ENTITY_SKIP = {"it", "them", "this", "that", "time", "track", "everything", "things", "stuff", "data", "info", "information", "notes", "who", "what"}


def offline_spec(title: str, chunks: list[dict], answered: dict[str, str]) -> Spec:
    roles: dict[str, Role] = {}
    features: list[Feature] = []
    entities: dict[str, Entity] = {}
    screens: dict[str, Screen] = {}
    constraints: list[Constraint] = []
    questions: dict[str, OpenQuestion] = {}
    summary = ""
    all_text = "\n".join(c["text"] for c in chunks)

    for chunk in chunks:
        is_answer = chunk["id"].startswith(("answers#", "feedback#"))
        for s in sentences(chunk["text"]):
            low = s.lower()

            if not summary and re.search(r"\b(goal|purpose|objective|we want to build|we need an?|build an? )\b", low):
                summary = s

            # Open questions from the notes themselves
            if not is_answer and (s.endswith("?") or _UNCERTAIN_RE.search(s)):
                q = _as_question(s)
                qid = question_id(q)
                if qid not in answered and qid not in questions:
                    questions[qid] = OpenQuestion(
                        id=qid, question=q, why_it_matters=why_it_matters(q), sources=[_ref(chunk)]
                    )
                continue

            if s == summary or _FIRST_PERSON_RE.search(s):
                continue

            # Roles and their goals
            for m in _ROLE_RE.finditer(s):
                if s[max(0, m.start() - 10) : m.start()].lower().endswith(("insurance ", "for ")):
                    continue
                name = _role_name(m.group(1))
                role = roles.setdefault(name, Role(name=name))
                _add_source(role.sources, chunk)
                g = _GOAL_RE.match(s[m.end():])
                if g and len(role.goals) < 5:
                    goal = re.split(r", so | so that ", g.group(1))[0].rstrip(".!")
                    if re.match(r"(a|an)\s", goal, re.I):
                        goal = "Get " + goal
                    if not any(_similar(goal, x) for x in role.goals):
                        role.goals.append(goal[0].upper() + goal[1:])

            constraint_kind = next((k for pat, k in CONSTRAINT_RULES if re.search(pat, low)), None)
            fm = _FIELDS_RE.match(s)

            # Requirements -> features (pure compliance/security rules become constraints only)
            req = _REQ_RE.search(s)
            if req and constraint_kind not in {"compliance", "security"} and not fm:
                modal = req.group(1).lower()
                priority = "must" if modal in _MUST else "could" if modal in _COULD else "should"
                existing = next((f for f in features if _similar(f.description, s, 0.5)), None)
                if existing:
                    _add_source(existing.sources, chunk)
                else:
                    tail = s[req.end():].strip() or s
                    features.append(
                        Feature(
                            id=f"F{len(features) + 1}",
                            name=_title(tail),
                            description=s,
                            priority=priority,
                            sources=[_ref(chunk)],
                        )
                    )

            # Entities with explicit fields ("Intake form fields: name, DOB, insurance")
            if fm:
                name = " ".join(_singular(w) for w in fm.group(1).lower().split()[-2:]).title()
                fields_ = [f.strip(" .").lower() for f in re.split(r",|;|\band\b", fm.group(2)) if f.strip(" .")]
                ent = entities.setdefault(name, Entity(name=name))
                for f in fields_:
                    if f not in ent.fields and len(f) < 40:
                        ent.fields.append(f)
                _add_source(ent.sources, chunk)
            for em in _ENTITY_VERB_RE.finditer(s):
                words = [w for w in em.group(1).lower().split() if w not in _ENTITY_SKIP]
                if not words:
                    continue
                name = " ".join(_singular(w) for w in words[-2:]).title()
                if name.lower() in _ENTITY_SKIP or len(name) < 3:
                    continue
                ent = entities.setdefault(name, Entity(name=name))
                _add_source(ent.sources, chunk)
                paren = re.match(r"\s*\(([^)]+)\)", s[em.end():])
                if paren:  # "record vitals (blood pressure, temperature, weight)"
                    for f in re.split(r",|\band\b", paren.group(1)):
                        f = f.strip().lower()
                        if f and f not in ent.fields:
                            ent.fields.append(f)

            # Screens
            for sm in _SCREEN_RE.finditer(s):
                words = [w for w in sm.group(1).split() if w.lower() not in _SCREEN_SKIP]
                if not words:
                    continue
                name = f"{' '.join(words)} {sm.group(2)}".title()
                key = set(name.lower().split())
                same = next((n for n in screens if key <= set(n.lower().split()) or key >= set(n.lower().split())), None)
                if same is None:
                    screens[name] = Screen(name=name, purpose=s, sources=[_ref(chunk)])
                    continue
                screen = screens[same]
                _add_source(screen.sources, chunk)
                if len(s) > len(screen.purpose):
                    screen.purpose = s
                if len(name) > len(same):  # keep the more specific name
                    screen.name = name
                    screens[name] = screens.pop(same)

            # Constraints
            if constraint_kind and not fm and not any(_similar(c.text, s, 0.5) for c in constraints):
                constraints.append(Constraint(text=s, kind=constraint_kind, sources=[_ref(chunk)]))

    # Gaps the notes never mention become questions, not assumptions.
    for check, q, why in GAP_CHECKS:
        qid = question_id(q)
        if check(all_text) and qid not in answered:
            questions.setdefault(qid, OpenQuestion(id=qid, question=q, why_it_matters=why))

    # Drop entities that are really roles or screens
    role_names = {r.lower() for r in roles}
    entities = {
        k: v for k, v in entities.items() if k.lower() not in role_names and not k.lower().endswith(("screen", "page"))
    }

    return Spec(
        title=title,
        summary=summary
        or f"Draft spec built from {len({c['filename'] for c in chunks})} source documents.",
        users=sorted(
            (r for r in roles.values() if r.goals or len(r.sources) > 1), key=lambda r: (-len(r.goals), -len(r.sources))
        )[:8],
        features=features[:25],
        entities=sorted(entities.values(), key=lambda e: (-len(e.fields), -len(e.sources)))[:12],
        screens=list(screens.values())[:10],
        constraints=constraints[:15],
        open_questions=list(questions.values())[:10],
    )


# =====================================================================
# LLM path
# =====================================================================

SYSTEM_PROMPT = """You are a senior product engineer. You turn messy project notes into a precise app specification.

Rules:
- Use ONLY facts found in the numbered context passages. Never invent requirements.
- Every role, feature, entity, screen and constraint must list the passage ids it came from in "sources", e.g. [{"chunk_id": "art_123#0"}].
- If something important is missing, ambiguous or contradictory, add it to "open_questions" instead of guessing.
- Do not ask again about questions that already have answers.
- Feature priority is "must", "should" or "could", based on the wording in the notes.
- Constraint kind is "security", "compliance", "technical" or "business".

Return one JSON object with exactly these keys:
{"title": str, "summary": str,
 "users": [{"name": str, "goals": [str], "sources": [ref]}],
 "features": [{"id": "F1", "name": str, "description": str, "priority": str, "sources": [ref]}],
 "entities": [{"name": str, "fields": [str], "sources": [ref]}],
 "screens": [{"name": str, "purpose": str, "sources": [ref]}],
 "constraints": [{"text": str, "kind": str, "sources": [ref]}],
 "open_questions": [{"id": "Q1", "question": str, "why_it_matters": str, "sources": [ref]}]}"""


def build_user_prompt(title: str, chunks: list[dict], answered: dict[str, str], questions: dict[str, str]) -> str:
    parts = [f"Project: {title}", "", "Context passages:"]
    for c in chunks:
        parts.append(f"[{c['id']}] ({c['filename']})\n{c['text']}\n")
    if answered:
        parts.append("Questions already answered (do not ask again):")
        for qid, ans in answered.items():
            parts.append(f"- {questions.get(qid, qid)} -> {ans}")
    parts.append("\nWrite the spec JSON now.")
    return "\n".join(parts)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(raw[start : end + 1])


def ground(spec: Spec, chunks: list[dict], answered: dict[str, str]) -> tuple[Spec, list[str]]:
    """Keep only citations that point at real context passages; drop uncited claims."""
    by_id = {c["id"]: c for c in chunks}
    warnings: list[str] = []

    def fix(sources: list[SourceRef]) -> list[SourceRef]:
        return [SourceRef(chunk_id=s.chunk_id, artifact=by_id[s.chunk_id]["filename"]) for s in sources if s.chunk_id in by_id]

    def keep(items: list, label: str, attr: str) -> list:
        kept = []
        for item in items:
            item.sources = fix(item.sources)
            if item.sources:
                kept.append(item)
            else:
                warnings.append(f"Dropped uncited {label}: {getattr(item, attr)[:80]}")
        return kept

    spec.users = keep(spec.users, "role", "name")
    spec.features = keep(spec.features, "feature", "name")
    spec.entities = keep(spec.entities, "entity", "name")
    spec.screens = keep(spec.screens, "screen", "name")
    spec.constraints = keep(spec.constraints, "constraint", "text")
    for i, f in enumerate(spec.features, 1):
        f.id = f"F{i}"
    questions = []
    for q in spec.open_questions:
        q.id = question_id(q.question)
        q.sources = fix(q.sources)
        if q.id not in answered:
            questions.append(q)
    spec.open_questions = questions
    return spec, warnings


def llm_spec(
    provider: LLMProvider,
    title: str,
    chunks: list[dict],
    answered: dict[str, str],
    question_text: dict[str, str],
    max_repairs: int = 2,
) -> GenerationResult:
    user = build_user_prompt(title, chunks, answered, question_text)
    last_error = ""
    raw = ""
    for attempt in range(max_repairs + 1):
        prompt = user
        if last_error:
            prompt += (
                f"\n\nYour previous answer was invalid:\n{last_error}\n"
                f"Previous answer:\n{raw[:4000]}\nReturn corrected JSON only."
            )
        raw = provider.complete_json(SYSTEM_PROMPT, prompt)
        try:
            spec = Spec.model_validate(_extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:1500]
            continue
        spec, warnings = ground(spec, chunks, answered)
        return GenerationResult(spec=spec, repairs=attempt, warnings=warnings)
    raise ProviderError(f"Model output failed validation after {max_repairs + 1} attempts: {last_error[:300]}")
