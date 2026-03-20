from __future__ import annotations

import re
from typing import List, Tuple

import spacy

from constants import COORD_SPLIT, SPACY_MODEL


_NLP = None
_LIST_MARKER_RE = re.compile(r"^\s*(\d+|[a-zA-Z])\.\s*$")


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


def merge_list_marker_chunks(
    chunk_spans: List[spacy.tokens.Span],
) -> List[spacy.tokens.Span]:
    """
    Merge chunks that are bare list markers (e.g. '1.', 'a.') with the following chunk.
    """
    if not chunk_spans:
        return chunk_spans

    merged: List[spacy.tokens.Span] = []
    doc = chunk_spans[0].doc
    i = 0

    while i < len(chunk_spans):
        if _LIST_MARKER_RE.match(chunk_spans[i].text.strip()) and i + 1 < len(chunk_spans):
            merged.append(doc[chunk_spans[i].start: chunk_spans[i + 1].end])
            i += 2
        else:
            merged.append(chunk_spans[i])
            i += 1

    return merged


def split_sent_on_newlines(sent: spacy.tokens.Span) -> List[spacy.tokens.Span]:
    """
    Split a spaCy sentence span into sub-spans on newline tokens.
    """
    doc = sent.doc
    start = sent.start
    spans: List[spacy.tokens.Span] = []

    for tok in sent:
        if "\n" in tok.text:
            if start < tok.i:
                spans.append(doc[start:tok.i])
            start = tok.i + 1

    if start < sent.end:
        spans.append(doc[start:sent.end])

    return [s for s in spans if len(s) > 0]


def split_sentence_into_chunks(sent: spacy.tokens.Span) -> List[spacy.tokens.Span]:
    """
    Optionally split a sentence into smaller chunks using coordinating boundaries.
    """
    tokens = list(sent)
    if not tokens:
        return [sent]

    split_token_idxs: List[int] = []
    for tok in tokens:
        if tok.lower_ in COORD_SPLIT:
            has_verb_left = any(t.pos_ in ("VERB", "AUX") and t.i < tok.i for t in tokens)
            has_verb_right = any(t.pos_ in ("VERB", "AUX") and t.i > tok.i for t in tokens)
            if has_verb_left and has_verb_right:
                split_token_idxs.append(tok.i)

    split_token_idxs = sorted(set(split_token_idxs))
    if not split_token_idxs:
        return [sent]

    doc = sent.doc
    chunks: List[spacy.tokens.Span] = []
    current_start = sent.start

    for split_i in split_token_idxs:
        if split_i <= sent.start or split_i >= sent.end:
            continue
        if split_i > current_start:
            chunks.append(doc[current_start:split_i])
        current_start = split_i

    if current_start < sent.end:
        chunks.append(doc[current_start:sent.end])

    return [c for c in chunks if len(c) > 0]


def chunks_to_token_spans(
    text: str,
    token: List[str],
    tokenizer,
    return_pos: bool = False,
    strict_token_check: bool = False,
):
    """
    Align spaCy chunks to tokenizer token indices using offset mapping.

    Returns:
      if return_pos=False:
        chunk_spans, chunk_to_token_spans
      else:
        chunk_spans, chunk_to_token_spans, chunk_to_pos_spans

    chunk_to_token_spans:
      [chunk][spacy_tok_in_chunk] -> [token_idx, ...]

    chunk_to_pos_spans:
      [chunk][spacy_tok_in_chunk] -> [pos_tag_for_each_token_idx, ...]
    """
    nlp = get_nlp()
    doc = nlp(text)

    chunk_spans: List[spacy.tokens.Span] = []
    for sent in doc.sents:
        for sub_sent in split_sent_on_newlines(sent):
            chunk_spans.extend(split_sentence_into_chunks(sub_sent))

    chunk_spans = merge_list_marker_chunks(chunk_spans)

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets: List[Tuple[int, int]] = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer must support return_offsets_mapping.")

    if strict_token_check:
        ids = enc.get("input_ids")
        if ids is None:
            raise ValueError("Tokenizer output missing input_ids.")
        toks = tokenizer.convert_ids_to_tokens(ids)
        if toks != token:
            raise ValueError(
                "Provided `token` list does not match tokenizer tokenization of `text`."
            )

    def spacy_token_to_tokenizer_indices(
        spacy_tokens: List[spacy.tokens.Token],
        offsets: List[Tuple[int, int]],
    ) -> List[List[int]]:
        out: List[List[int]] = []
        j = 0

        for st in spacy_tokens:
            s_char = st.idx
            e_char = st.idx + len(st.text)

            while j < len(offsets) and offsets[j][1] <= s_char:
                j += 1

            jj = j
            idxs: List[int] = []
            while jj < len(offsets):
                s, e = offsets[jj]

                if e <= s:
                    jj += 1
                    continue
                if s >= e_char:
                    break
                if not (e <= s_char or s >= e_char):
                    idxs.append(jj)
                jj += 1

            out.append(idxs)
            j = max(j, jj - 1)

        return out

    chunk_to_token_spans: List[List[List[int]]] = []
    chunk_to_pos_spans: List[List[List[str]]] = []

    for ch in chunk_spans:
        spacy_toks = list(ch)
        per_tok_idxs = spacy_token_to_tokenizer_indices(spacy_toks, offsets)
        chunk_to_token_spans.append(per_tok_idxs)

        if return_pos:
            per_tok_pos: List[List[str]] = []
            for st, idxs in zip(spacy_toks, per_tok_idxs):
                per_tok_pos.append([st.pos_] * len(idxs))
            chunk_to_pos_spans.append(per_tok_pos)

    if return_pos:
        return chunk_spans, chunk_to_token_spans, chunk_to_pos_spans
    return chunk_spans, chunk_to_token_spans


def chunk_prompt_text(prompt: str, tokenizer) -> List[dict]:
    """
    Chunk plain prompt text and compute prompt-local token spans.

    Returns:
        [
            {
                "chunk_id": int,
                "input_id_span": (start_idx, end_idx),
                "text": str,
            },
            ...
        ]
    """
    enc = tokenizer(prompt, add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    chunk_spans, chunk_to_token_spans = chunks_to_token_spans(
        text=prompt,
        token=token_strs,
        tokenizer=tokenizer,
        return_pos=False,
        strict_token_check=False,
    )

    result = []
    for i, (sp, token_span) in enumerate(zip(chunk_spans, chunk_to_token_spans)):
        flat_indices = [idx for group in token_span for idx in group]
        if flat_indices:
            start_idx = min(flat_indices)
            end_idx = max(flat_indices)
        else:
            start_idx = end_idx = 0

        result.append(
            {
                "chunk_id": i,
                "input_id_span": (start_idx, end_idx),
                "text": str(sp),
            }
        )

    return result