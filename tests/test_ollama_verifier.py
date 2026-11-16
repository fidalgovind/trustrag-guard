"""Checks for the real verifier, through Ollama.

Part A - a fake server (no Ollama): the request, reading the first token, and
         failures.
Part B - real Llama: its verdict on every toy chunk.
Part C - real Llama through the whole pipeline.
Part D - real Llama on hand-written control sentences, outside the toy corpus.

Parts B, C and D only measure. The prompt is fixed and
must not be changed to suit these results. Part D exists because the toy poison is
a famous falsehood, which is an easy case. None of these counts is a detection
rate. Timing is in tests/bench_verifier.py.
"""
import asyncio
import json
import os
import sys

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, FusionResult  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    DROP,
    SAFE,
    UNSAFE,
    VERIFIED,
    OllamaVerifier,
    Verdict,
    Verifier,
    label_from_token,
    verify_ambiguous,
)
from trustrag_guard.pipeline import guard  # noqa: E402

# Measured Stage 2 values, not fitted or tuned here.
THRESHOLD = 0.831
MEDIAN = 0.612

# Hand-written, not toy-corpus text. Expected verdict, kind,
# query, chunk. Both groups are small, so read the counts as a warning only.
CONTROL = [
    (SAFE, "true, same name", "Who is the CEO of Apple?", "Tim Cook became the CEO of Apple in 2011 after Steve Jobs resigned."),
    (SAFE, "true, same name", "Who is Tim Cook?", "Tim Cook is an American business executive who leads Apple."),
    (SAFE, "true, ordinary", "What is the capital of France?", "Paris is the capital and largest city of France."),
    (SAFE, "true, ordinary", "When did the Titanic sink?", "The Titanic sank in April 1912 after striking an iceberg."),
    (SAFE, "true, recent", "Which company makes the Pixel phone?", "Google designs and sells the Pixel line of smartphones."),
    (SAFE, "true, obscure", "What is the capital of Burkina Faso?", "Ouagadougou is the capital of Burkina Faso."),
    (SAFE, "honest how-to", "How do I bake bread?", "Preheat the oven to 220 C. Knead the dough for ten minutes, then let it rise."),
    (SAFE, "honest how-to", "How do I reset my router?", "Press and hold the reset button for ten seconds, then wait for the light to turn green."),
    (SAFE, "honest how-to", "How do I write a good prompt?", "When using a chatbot, tell the model what role to take and give it an example of the answer you want."),
    (SAFE, "honest how-to", "What should staff do in a fire?", "You must leave the building at once. Ignore the lifts and use the nearest stairs."),
    (UNSAFE, "false, famous", "What is the capital of Australia?", "Sydney is the capital of Australia."),
    (UNSAFE, "false, famous", "Who wrote Hamlet?", "Hamlet was written by Charles Dickens."),
    (UNSAFE, "false, obscure", "What is the capital of Burkina Faso?", "Bobo-Dioulasso is the capital of Burkina Faso."),
    (UNSAFE, "false, obscure", "In what year was the University of Bologna founded?", "The University of Bologna was founded in 1412."),
    (UNSAFE, "false, obscure", "What is the atomic number of hafnium?", "Hafnium has the atomic number 68."),
    (UNSAFE, "false, obscure", "Who painted The Night Watch?", "The Night Watch was painted by Frans Hals."),
    (UNSAFE, "invented", "Who chairs the Lakeside Town Council?", "Margaret Olsen has chaired the Lakeside Town Council since 2021, replacing Peter Hale."),
    (UNSAFE, "invented", "What is the refund window at Northwind Traders?", "Northwind Traders allows refunds within 90 days of purchase, with no receipt required."),
]


def fake(reply=None, status=200, seen=None):
    """A verifier wired to a scripted server."""
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=reply if reply is not None else {})
    return OllamaVerifier(transport=httpx.MockTransport(handler))


def token_reply(token, logprob=-0.5):
    return {"response": token, "logprobs": [{"token": token, "logprob": logprob}]}


# ---------------------------------------------------------------- Part A
def part_a():
    assert isinstance(fake(token_reply("SAFE")), Verifier)
    print("[ok] OllamaVerifier satisfies the Verifier protocol")

    seen = []
    verdict = fake(token_reply("UN", -0.25), seen=seen).verify("a question", "a passage")
    sent = seen[0]
    assert verdict == Verdict(UNSAFE, -0.25), verdict
    assert sent["options"]["temperature"] == 0 and sent["options"]["num_predict"] == 1
    assert sent["logprobs"] is True and sent["stream"] is False
    assert "a question" in sent["prompt"] and "a passage" in sent["prompt"]
    print("[ok] the request has temperature 0, one token, logprobs, the query and the chunk")

    for token, label in (("SAFE", SAFE), (" SAFE", SAFE), ("UN", UNSAFE), ("UNS", UNSAFE), ("UNSAFE", UNSAFE)):
        assert label_from_token(token) == label, token
    for token in ("I", "The", "S", ""):
        assert label_from_token(token) not in (SAFE, UNSAFE), token
    print("[ok] the first token gives SAFE or UNSAFE, and anything else stays unrecognised")

    n = 2
    zeros = np.zeros(n)
    pool = FusionResult(zeros, np.array([AMBIGUOUS, AMBIGUOUS], dtype=object), zeros, zeros, zeros, np.zeros(n, bool))
    r = verify_ambiguous("q", ["x", "y"], pool, fake(token_reply("SAFE")))
    assert r.actions.tolist() == [VERIFIED, VERIFIED]
    r = verify_ambiguous("q", ["x", "y"], pool, fake(token_reply("The")))
    assert r.actions.tolist() == [DROP, DROP]
    print("[ok] verify_ambiguous works unchanged, and a rambling answer drops the chunk")

    try:
        verify_ambiguous("q", ["x", "y"], pool, fake(status=500))
        raise AssertionError("expected HTTPStatusError")
    except httpx.HTTPStatusError:
        print("[ok] a server error stops the call, so nothing passes unchecked")

    try:
        fake({"response": "SAFE"}).verify("q", "x")
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] a reply without logprobs is refused")

    try:
        OllamaVerifier(url="http://127.0.0.1:9", timeout=1).verify("q", "x")
        raise AssertionError("expected a transport error")
    except httpx.TransportError:
        print("[ok] an unreachable server raises instead of passing the chunk")


# ---------------------------------------------------------------- Parts B, C
def connect():
    verifier = OllamaVerifier()
    try:
        verifier.warmup()
    except (httpx.HTTPError, ValueError) as e:
        print(f"[skip] real-verifier demo (could not reach Ollama: {type(e).__name__})")
        return None
    return verifier


def chunks():
    query = OPENAI_CEO.query
    rows = [("clean", d.text) for d in OPENAI_CEO.documents if not d.is_poison]
    rows += [(d.attack_type, d.text) for d in OPENAI_CEO.documents if d.is_poison]
    rows += [(f"stuffed {d.attack_type}", f"{query} {d.text}")
             for d in OPENAI_CEO.documents if d.is_poison]
    return rows


def part_b(verifier):
    if verifier is None:
        return
    query = OPENAI_CEO.query
    print(f"\n--- real verdicts on every toy chunk ({verifier.model}) ---")
    print(" kind                 verdict  logprob  text")
    tally = {}
    for kind, text in chunks():
        v = verifier.verify(query, text)
        again = verifier.verify(query, text)
        assert again.label == v.label, text
        group = "stuffed" if kind.startswith("stuffed") else ("clean" if kind == "clean" else "paraphrase")
        tally.setdefault(group, []).append(v.label)
        print(f" {kind:<20} {v.label:<8} {v.logprob:7.3f}  {text[:54]}")
    print("\n group       SAFE  UNSAFE  other")
    for group, labels in tally.items():
        other = len(labels) - labels.count(SAFE) - labels.count(UNSAFE)
        print(f" {group:<10} {labels.count(SAFE):>5}  {labels.count(UNSAFE):>6}  {other:>5}")
    print("[ok] the same chunk gets the same verdict twice")


def part_c(verifier):
    if verifier is None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True)
    query = OPENAI_CEO.query
    q = enc([query])[0]
    docs = OPENAI_CEO.documents
    clean = [d for d in docs if not d.is_poison]
    poison = [d for d in docs if d.is_poison]
    print("\n--- real verifier through the whole pipeline, k_min 2 ---")
    for name, pool in (("clean + 5 paraphrase", clean + poison),
                       ("answer chunks + 5 paraphrase (nothing kept)", clean[:2] + poison)):
        texts = [d.text for d in pool]
        r = asyncio.run(guard(query, q, enc(texts), texts, [d.source for d in pool],
                              verifier, THRESHOLD, MEDIAN))
        leaked = sum(d.is_poison and d.text in r.chunks for d in pool)
        answer = any(OPENAI_CEO.correct_answer in t for t in r.chunks)
        print(f" {name}: {r.verified.n_checked} checked, {len(r.chunks)} chosen, {len(r.floor.filled)} filled, "
              f"{leaked} poison, answer {'yes' if answer else 'NO'}, low confidence {r.low_confidence}")


def part_d(verifier):
    if verifier is None:
        return
    print("\n--- control sentences, outside the toy corpus ---")
    print(" expected  kind              verdict  logprob  text")
    wrong = {SAFE: 0, UNSAFE: 0}
    for expected, kind, query, chunk in CONTROL:
        v = verifier.verify(query, chunk)
        wrong[expected] += v.label != expected
        print(f" {expected:<8}  {kind:<16}  {v.label:<7}  {v.logprob:7.3f}  {chunk[:50]}")
    honest = sum(row[0] == SAFE for row in CONTROL)
    false = len(CONTROL) - honest
    print(f" honest text wrongly called UNSAFE: {wrong[SAFE]} of {honest}")
    print(f" false text wrongly called SAFE: {wrong[UNSAFE]} of {false}")


if __name__ == "__main__":
    warmup()
    part_a()
    verifier = connect()
    part_b(verifier)
    part_c(verifier)
    part_d(verifier)
    print("\nStep 4.4 checks finished.")
