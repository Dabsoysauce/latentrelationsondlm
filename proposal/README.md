# DiffuLM @ NeurIPS 2026 — project proposal

`proposal.tex` targets the DiffuLM workshop (Sydney, 12 Dec 2026).

Submission facts, checked against the CFP on 2026-08-06:
- deadline **29 Aug 2026 AoE**; notification 29 Sep; camera-ready Oct
- **non-archival**, double-blind, >=3 reviews, Best Paper / Best Student Paper
- two tracks: extended abstract <=4 pages, short paper <=8 pages (main text)
- NeurIPS 2026 style template required

Eligibility: DiffuLM bars work "already published at NeurIPS 2026 or another
archival venue". The COLM 2026 NonAR-LM workshop is explicitly non-archival
("submitting does not preclude publishing elsewhere"), so the prior paper does
not block this submission -- but it must be cited in the third person and the
submission must be a substantive extension, not a reprint.

## To build

`neurips_2026.sty` is not in this directory; download the style bundle from
neurips.cc and drop it alongside `proposal.tex`. The preamble already selects
`[dblblindworkshop]` and sets `\workshoptitle`, which the workshop template
requires in addition to `\title`.
