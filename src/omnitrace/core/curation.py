from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from omnitrace.constants import POS_W, SourceCurationConfig, DEFAULT_CURATE_CFG


def curate_sources_with_conf(
    source_ids: List[int],
    pos: List[str],
    conf: List[float],
    cfg: SourceCurationConfig = DEFAULT_CURATE_CFG,
) -> List[int]:
    """
    Curate a compact subset of source ids from token-level source assignments.

    Each token vote is weighted by:
      - POS importance
      - confidence^gamma

    Then each source is scored using a mixture of:
      - total normalized attribution mass
      - strongest contiguous run mass
    """
    T = len(source_ids)
    if T == 0:
        return []
    if not (T == len(pos) == len(conf)):
        raise ValueError("source_ids, pos, conf must have the same length.")

    vote: List[float] = []
    for p, c in zip(pos, conf):
        pw = POS_W.get(p, 0.3)
        cw = max(c, 0.0) ** cfg.gamma
        vote.append(pw * cw)

    total = float(sum(vote))
    if total <= 0:
        return []

    mass = defaultdict(float)
    for s, v in zip(source_ids, vote):
        mass[s] += v
    p_mass: Dict[int, float] = {s: m / total for s, m in mass.items()}

    run_max = defaultdict(float)
    cur_s = source_ids[0]
    cur_run = vote[0]

    for i in range(1, T):
        if source_ids[i] == cur_s:
            cur_run += vote[i]
        else:
            run_max[cur_s] = max(run_max[cur_s], cur_run)
            cur_s = source_ids[i]
            cur_run = vote[i]
    run_max[cur_s] = max(run_max[cur_s], cur_run)

    run_frac: Dict[int, float] = {s: run_max[s] / total for s in mass.keys()}

    def score(s: int) -> float:
        return cfg.alpha * p_mass[s] + (1.0 - cfg.alpha) * run_frac[s]

    ranked = sorted(p_mass.keys(), key=score, reverse=True)

    selected: List[int] = []
    cum = 0.0
    for s in ranked:
        strong_run = run_frac[s] >= cfg.run_min
        if p_mass[s] < cfg.p_min and not strong_run:
            continue
        selected.append(s)
        cum += p_mass[s]
        if cum >= cfg.coverage:
            break

    return selected