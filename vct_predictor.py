#!/usr/bin/env python3
"""
VCT Champions match predictor
=============================

Setup
-----
    pip install numpy
    mkdir data            # put your 16 team JSONs in ./data  (Paper_Rex.json, NRG.json, ...)

Usage
-----
    python vct_predictor.py train                    # learn from the JSONs, print CV accuracy
    python vct_predictor.py predict "PRX" "NRG"      # predict, then answer right / wrong
    python vct_predictor.py chat                     # keep predicting matchups in a loop
    python vct_predictor.py learn "PRX" "NRG" --winner NRG   # log a result you already know
    python vct_predictor.py history                  # your running accuracy on corrections
    python vct_predictor.py teams                    # list teams / accepted names

How it works
------------
1. Every match in every JSON contains BOTH teams' player stats, so each unique match id
   becomes one training example (win/loss label) and opponents outside your 16 files
   still build up a profile from the matches they appear in.
2. A team "profile" = recency-weighted, shrunk-toward-average mean of match results
   (match / map / round win %) and player stats (rating, ACS, ADR, KAST, K/D, first-kill
   diff, attack & defend rating). For training, the match being predicted is left out of
   both teams' profiles, so the model never sees the answer.
3. Model = L2-regularised logistic regression on (profile A - profile B). No intercept,
   so P(A beats B) = 1 - P(B beats A).
4. Every "right / wrong" you give is stored in feedback.jsonl and the model is refit on
   base data + your corrections (corrections count FEEDBACK_WEIGHT times as much).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
MODEL_PATH = HERE / "model.json"
FEEDBACK_PATH = HERE / "feedback.jsonl"

# ----------------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------------
SHRINK_K = 2.0            # pseudo-matches of "average team" mixed into every profile
RECENCY_HALF_LIFE = 10.0  # a match this many games older counts half as much
MIN_OTHER_MATCHES = 2     # a team needs this many *other* matches to be used in training
FEEDBACK_WEIGHT = 3.0     # each of your corrections counts as this many normal samples
LAMBDA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# Manual name merges if the same team is spelled differently across files, e.g.
# {"KRU Esports": "KRÜ Esports"}
ALIASES: dict[str, str] = {}

RESULT_FEATURES = ["match_win", "map_win", "round_pct"]
PERF_FEATURES = ["rating", "acs", "adr", "kast", "kd", "fk_pr", "atk_rating", "def_rating"]
FEATURES = RESULT_FEATURES + PERF_FEATURES
LABELS = {
    "match_win": "Match win %", "map_win": "Map win %", "round_pct": "Round win %",
    "rating": "Rating", "acs": "ACS", "adr": "ADR", "kast": "KAST", "kd": "K/D",
    "fk_pr": "First-kill diff/round", "atk_rating": "Attack rating", "def_rating": "Defense rating",
}


def norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", str(s).casefold())


_ALIASES_N = {norm(k): norm(v) for k, v in ALIASES.items()}


def canon(s: str) -> str:
    n = norm(s)
    return _ALIASES_N.get(n, n)


# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
@dataclass
class TeamMatch:
    match_id: int
    key: str
    name: str
    opp_key: str
    won: bool
    feats: np.ndarray


@dataclass
class Sample:
    match_id: int
    team_a: str
    team_b: str
    x: np.ndarray   # profile(A) - profile(B), raw units
    y: float        # 1.0 if A won


def _series_players(maps: dict, side_key: str):
    """Player stat blocks are series totals repeated on every map, so read each player once."""
    out, mismatches = {}, 0
    for mm in maps.values():
        for tname, players in mm.items():
            if tname == "score" or canon(tname) != side_key:
                continue
            for pname, blk in players.items():
                if pname not in out:
                    out[pname] = blk
                elif out[pname].get("overall") != blk.get("overall"):
                    mismatches += 1
    return out, mismatches


def _wmean(players: dict, block: str, field: str) -> float:
    num = den = 0.0
    for p in players.values():
        v = (p.get(block) or {}).get(field)
        if v is None:
            continue
        o = p.get("overall") or {}
        w = max(1.0, (o.get("kills") or 0) + (o.get("deaths") or 0) + (o.get("assists") or 0))
        num += w * float(v)
        den += w
    return num / den if den else float("nan")


def _side_features(players, won, maps_won, maps_played, rounds_won, rounds_total) -> np.ndarray:
    kills = sum((p.get("overall") or {}).get("kills") or 0 for p in players.values())
    deaths = sum((p.get("overall") or {}).get("deaths") or 0 for p in players.values())
    fkd = sum((p.get("overall") or {}).get("firstKDdiff") or 0 for p in players.values())
    f = {
        "match_win": float(won),
        "map_win": maps_won / maps_played,
        "round_pct": rounds_won / rounds_total if rounds_total else float("nan"),
        "rating": _wmean(players, "overall", "rating"),
        "acs": _wmean(players, "overall", "acs"),
        "adr": _wmean(players, "overall", "adr"),
        "kast": _wmean(players, "overall", "kast"),
        "kd": kills / max(1, deaths) if players else float("nan"),
        "fk_pr": fkd / rounds_total if rounds_total and players else float("nan"),
        "atk_rating": _wmean(players, "attack", "rating"),
        "def_rating": _wmean(players, "defend", "rating"),
    }
    return np.array([f[k] for k in FEATURES], dtype=float)


class TeamDB:
    def __init__(self):
        self.rows: dict[str, dict[int, TeamMatch]] = defaultdict(dict)
        self.names: dict[str, str] = {}
        self.dataset: dict[str, dict] = {}   # teams that have their own JSON
        self.alias: dict[str, str] = {}
        self.global_mean = np.zeros(len(FEATURES))
        self.stat_mismatches = 0
        self.skipped_matches = 0
        self._fake_id = 0

    # ---- ingestion -------------------------------------------------------------
    def add_file(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        matches = [v for k, v in raw.items() if re.fullmatch(r"Match \d+", k) and isinstance(v, dict)]
        own_name = None
        for m in matches:
            names = [k for k in (m.get("Score") or {}) if k != "opp"]
            if names:
                own_name = names[0]
                break
        if own_name is None:
            own_name = path.stem.replace("_", " ")
        own_key = canon(own_name)
        self.names.setdefault(own_key, own_name)
        self.dataset[own_key] = {
            "tag": raw.get("tag", ""), "name": own_name, "file": path.name,
            "country": raw.get("country", ""), "roster": raw.get("currRoster", []),
        }
        for m in matches:
            if not self._add_match(m, own_key):
                self.skipped_matches += 1

    def _add_match(self, m: dict, own_key: str) -> bool:
        maps = m.get("Each Map") or {}
        teams = {}
        for mm in maps.values():
            for t in mm:
                if t != "score":
                    teams.setdefault(canon(t), t)
        if own_key not in teams or len(teams) != 2:
            return False
        opp_key = next(k for k in teams if k != own_key)
        rounds = {own_key: 0, opp_key: 0}
        mwon = {own_key: 0, opp_key: 0}
        played = 0
        for mm in maps.values():
            sc = {canon(n): r for n, r in mm.get("score", [])}
            if own_key not in sc or opp_key not in sc:
                continue
            rounds[own_key] += sc[own_key]
            rounds[opp_key] += sc[opp_key]
            played += 1
            if sc[own_key] > sc[opp_key]:
                mwon[own_key] += 1
            elif sc[opp_key] > sc[own_key]:
                mwon[opp_key] += 1
        if played == 0:
            return False
        total_rounds = rounds[own_key] + rounds[opp_key]
        if isinstance(m.get("win"), bool):
            own_won = m["win"]
        else:
            own_won = mwon[own_key] > mwon[opp_key]
        if m.get("id") is not None:
            mid = int(m["id"])
        else:
            self._fake_id -= 1
            mid = self._fake_id

        for key, opp, won in ((own_key, opp_key, own_won), (opp_key, own_key, not own_won)):
            players, mism = _series_players(maps, key)
            self.stat_mismatches += mism
            feats = _side_features(players, won, mwon[key], played, rounds[key], total_rounds)
            self.names.setdefault(key, teams[key])
            self.rows[key].setdefault(mid, TeamMatch(mid, key, teams[key], opp, won, feats))
        return True

    def finalize(self):
        allf = np.vstack([r.feats for d in self.rows.values() for r in d.values()])
        gm = np.nanmean(allf, axis=0)
        self.global_mean = np.where(np.isnan(gm), 0.0, gm)
        self.alias = {k: k for k in self.names}
        for key, info in self.dataset.items():
            for a in (info["tag"], info["name"], Path(info["file"]).stem):
                n = canon(a)
                if n:
                    self.alias.setdefault(n, key)

    # ---- team lookup -----------------------------------------------------------
    def resolve(self, query: str) -> str:
        q = canon(query)
        if q in self.alias:
            return self.alias[q]
        cands = {self.alias[a] for a in self.alias if q and (a.startswith(q) or q in a)}
        if len(cands) == 1:
            return cands.pop()
        if len(cands) > 1:
            raise KeyError(f"'{query}' is ambiguous: " + ", ".join(sorted(self.names[c] for c in cands)))
        close = difflib.get_close_matches(q, list(self.alias), n=3, cutoff=0.65)
        hint = ", ".join(sorted({self.names[self.alias[c]] for c in close}))
        raise KeyError(f"Unknown team '{query}'." + (f" Did you mean: {hint}?" if hint else ""))

    # ---- profiles --------------------------------------------------------------
    def profile(self, key: str, exclude: int | None = None):
        d = self.rows.get(key)
        if not d:
            return None, 0
        rows = [d[m] for m in sorted(d, reverse=True) if m != exclude]   # newest first
        n = len(rows)
        if n == 0:
            return None, 0
        X = np.vstack([r.feats for r in rows])
        w = 0.5 ** (np.arange(n) / RECENCY_HALF_LIFE)
        mask = ~np.isnan(X)
        num = (w[:, None] * np.where(mask, X, 0.0)).sum(0)
        den = (w[:, None] * mask).sum(0)
        return (num + SHRINK_K * self.global_mean) / (den + SHRINK_K), n

    def build_samples(self, min_other: int) -> tuple[list[Sample], int]:
        by_match = defaultdict(list)
        for d in self.rows.values():
            for mid, r in d.items():
                by_match[mid].append(r)
        samples, skipped = [], 0
        for mid, rs in sorted(by_match.items()):
            if len(rs) != 2:
                skipped += 1
                continue
            a, b = sorted(rs, key=lambda r: r.key)
            pa, na = self.profile(a.key, exclude=mid)
            pb, nb = self.profile(b.key, exclude=mid)
            if na < min_other or nb < min_other:
                skipped += 1
                continue
            samples.append(Sample(mid, a.name, b.name, pa - pb, 1.0 if a.won else 0.0))
        return samples, skipped


def load_db(data_dir: Path) -> TeamDB:
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        sys.exit(f"No .json files found in {data_dir}. Put your team JSONs there (or pass --data).")
    db = TeamDB()
    for f in files:
        try:
            db.add_file(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not read {f.name}: {e}")
    if not db.rows:
        sys.exit("No usable matches found in the JSON files.")
    db.finalize()
    if db.stat_mismatches:
        print(f"  ! {db.stat_mismatches} player stat blocks differed between maps of one series; "
              "the first map's values were used (stats are assumed to be series totals).")
    return db


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, sw, lam, iters=60):
    """Weighted L2 logistic regression (no intercept) via Newton / IRLS."""
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (sw * (p - y)) + lam * w
        H = (X.T * (sw * p * (1 - p))) @ X + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def cross_validate(X, y, sw, lam, k=5, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    correct = ll = 0.0
    for _ in range(repeats):
        idx = rng.permutation(n)
        for fold in np.array_split(idx, k):
            tr = np.setdiff1d(idx, fold)
            w = fit_logreg(X[tr], y[tr], sw[tr], lam)
            p = sigmoid(X[fold] @ w)
            correct += np.sum((p > 0.5) == (y[fold] > 0.5))
            ll += -np.sum(np.log(np.where(y[fold] > 0.5, p, 1 - p) + 1e-12))
    return correct / (repeats * n), ll / (repeats * n)


def baseline_acc(X, y, feature: str) -> float:
    s = np.sign(X[:, FEATURES.index(feature)])
    return float(np.mean(np.where(s == 0, 0.5, (s > 0) == (y > 0.5))))


def load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    out = []
    for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            if len(rec.get("x", [])) == len(FEATURES):
                out.append(rec)
    return out


def refit(model: dict, base: list[Sample]) -> dict:
    fb = load_feedback()
    X = np.array([s.x for s in base] + [np.array(f["x"]) for f in fb])
    y = np.array([s.y for s in base] + [f["y"] for f in fb])
    sw = np.array([1.0] * len(base) + [FEEDBACK_WEIGHT] * len(fb))
    w = fit_logreg(X / np.array(model["scale"]), y, sw, model["lam"])
    model["w"] = w.tolist()
    model["n_feedback"] = len(fb)
    model["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_model() -> dict:
    if not MODEL_PATH.exists():
        sys.exit("No model yet. Run:  python vct_predictor.py train")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------
# Prediction + feedback
# ----------------------------------------------------------------------------------
@dataclass
class Prediction:
    key_a: str
    key_b: str
    name_a: str
    name_b: str
    x: np.ndarray
    p_a: float
    contrib: np.ndarray
    prof_a: np.ndarray
    prof_b: np.ndarray
    n_a: int
    n_b: int

    @property
    def pick_key(self):
        return self.key_a if self.p_a >= 0.5 else self.key_b

    @property
    def pick_name(self):
        return self.name_a if self.p_a >= 0.5 else self.name_b


def predict_pair(db: TeamDB, model: dict, ka: str, kb: str) -> Prediction:
    if ka == kb:
        raise KeyError("Pick two different teams.")
    pa, na = db.profile(ka)
    pb, nb = db.profile(kb)
    if pa is None or pb is None:
        raise KeyError("No match data for one of those teams.")
    x = pa - pb
    xs = x / np.array(model["scale"])
    w = np.array(model["w"])
    return Prediction(ka, kb, db.names[ka], db.names[kb], x, float(sigmoid(xs @ w)),
                      xs * w, pa, pb, na, nb)


def _fmt(feature: str, v: float) -> str:
    return f"{v * 100:.0f}%" if feature in ("match_win", "map_win", "round_pct", "kast") else (
        f"{v:.3f}" if feature in ("fk_pr",) else f"{v:.2f}" if v < 10 else f"{v:.0f}")


def show_prediction(p: Prediction):
    win_p = p.p_a if p.p_a >= 0.5 else 1 - p.p_a
    print(f"\n  {p.name_a}  vs  {p.name_b}")
    if abs(p.p_a - 0.5) < 0.05:
        print(f"  Prediction: toss-up, slight lean {p.pick_name} ({win_p:.0%})")
    else:
        print(f"  Prediction: {p.pick_name} wins  ({p.name_a} {p.p_a:.0%} / {p.name_b} {1 - p.p_a:.0%})")
    print(f"  (profiles built from {p.n_a} matches for {p.name_a}, {p.n_b} for {p.name_b})")
    if min(p.n_a, p.n_b) < 4:
        print("  ! thin data for one team - treat this one with extra suspicion")
    print("  Biggest factors:")
    for j in np.argsort(-np.abs(p.contrib))[:4]:
        f = FEATURES[j]
        favours = p.name_a if p.contrib[j] > 0 else p.name_b
        print(f"    {LABELS[f]:<22} {_fmt(f, p.prof_a[j]):>6} vs {_fmt(f, p.prof_b[j]):<6} -> favours {favours}")


def accuracy_line(fb: list[dict]) -> str:
    if not fb:
        return "no corrections logged yet"
    ok = sum(1 for r in fb if r["correct"])
    return f"{ok}/{len(fb)} predictions correct ({ok / len(fb):.0%})"


def record_feedback(p: Prediction, winner_key: str) -> dict:
    a_won = winner_key == p.key_a
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "team_a": p.name_a, "team_b": p.name_b,
        "x": [float(v) for v in p.x],
        "prob_a": round(p.p_a, 4),
        "predicted": p.pick_name,
        "winner": p.name_a if a_won else p.name_b,
        "correct": p.pick_key == winner_key,
        "y": 1.0 if a_won else 0.0,
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def apply_feedback(db, model, base, pred: Prediction, winner_key: str):
    before = pred.p_a
    rec = record_feedback(pred, winner_key)
    refit(model, base)
    after = predict_pair(db, model, pred.key_a, pred.key_b).p_a
    print(f"\n  Logged: {rec['winner']} won ({'model was right' if rec['correct'] else 'model was wrong'}).")
    print(f"  Refit on {len(base)} base samples + {model['n_feedback']} corrections. "
          f"P({pred.name_a}) for this matchup: {before:.0%} -> {after:.0%}")
    print(f"  Running record: {accuracy_line(load_feedback())}")


def ask_verdict(db, model, base, pred: Prediction):
    try:
        ans = input("\n  Was that right?  [y] right  [n] wrong  [s] skip : ").strip().lower()
    except EOFError:
        return
    if ans in ("y", "yes", "r", "right"):
        apply_feedback(db, model, base, pred, pred.pick_key)
    elif ans in ("n", "no", "w", "wrong"):
        other = pred.key_b if pred.pick_key == pred.key_a else pred.key_a
        apply_feedback(db, model, base, pred, other)
    else:
        print("  Skipped - nothing learned.")


# ----------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------
def _context(args, need_samples=True):
    model = load_model()
    data_dir = Path(args.data) if getattr(args, "data", None) else Path(model.get("data_dir", DEFAULT_DATA_DIR))
    db = load_db(data_dir)
    base = db.build_samples(model.get("min_matches", MIN_OTHER_MATCHES))[0] if need_samples else []
    return db, model, base


def cmd_train(args):
    data_dir = Path(args.data) if args.data else DEFAULT_DATA_DIR
    db = load_db(data_dir)
    samples, skipped = db.build_samples(args.min_matches)
    n_matches = len({m for d in db.rows.values() for m in d})
    print(f"Loaded {len(db.dataset)} team files -> {len(db.names)} teams (incl. opponents), {n_matches} unique matches.")
    print(f"Training samples: {len(samples)}  (skipped {skipped}: a team had < {args.min_matches} other matches)")
    if len(samples) < 12:
        sys.exit("Too few usable samples to train. Add more team JSONs, or lower --min-matches to 1.")

    dataset_keys = list(db.dataset)
    for key in db.names:
        if key not in db.dataset:
            near = difflib.get_close_matches(key, dataset_keys, n=1, cutoff=0.85)
            if near:
                print(f"  ! '{db.names[key]}' looks like '{db.names[near[0]]}' - if same team, add to ALIASES")

    X = np.array([s.x for s in samples])
    y = np.array([s.y for s in samples])
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    scale[scale == 0] = 1.0
    Xs = X / scale
    ones = np.ones(len(y))

    print("\nCross-validating (5-fold x 5 repeats) to pick regularisation ...")
    results = {lam: cross_validate(Xs, y, ones, lam) for lam in LAMBDA_GRID}
    lam = min(results, key=lambda l: results[l][1])
    acc, ll = results[lam]
    print(f"  best lambda = {lam}:  accuracy {acc:.1%}   log-loss {ll:.3f}   (coin flip = 50.0% / 0.693)")
    print(f"  baseline 'higher match win %':  {baseline_acc(X, y, 'match_win'):.1%}")
    print(f"  baseline 'higher round win %':  {baseline_acc(X, y, 'round_pct'):.1%}")
    print(f"  With ~{len(samples)} samples the accuracy estimate is only good to about +/-{100 * 1 / np.sqrt(len(samples)):.0f} points.")

    model = {
        "features": FEATURES, "scale": scale.tolist(), "w": [0.0] * len(FEATURES), "lam": lam,
        "n_base": len(samples), "n_feedback": 0, "min_matches": args.min_matches,
        "data_dir": str(data_dir.resolve()), "cv_accuracy": acc, "cv_logloss": ll,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    refit(model, samples)
    fb_n = model["n_feedback"]
    if fb_n:
        print(f"  (also re-applied {fb_n} logged corrections from feedback.jsonl)")

    print("\nFeature weights (standardised; + = having more of it helps you win):")
    w = np.array(model["w"])
    for j in np.argsort(-np.abs(w)):
        print(f"  {LABELS[FEATURES[j]]:<22} {w[j]:+.3f}")
    print(f"\nSaved {MODEL_PATH.name}. Next:  python vct_predictor.py predict \"Team A\" \"Team B\"")


def cmd_predict(args):
    db, model, base = _context(args)
    pred = predict_pair(db, model, db.resolve(args.team_a), db.resolve(args.team_b))
    show_prediction(pred)
    if sys.stdin.isatty() and not args.no_ask:
        ask_verdict(db, model, base, pred)


def _parse_matchup(line: str):
    parts = re.split(r"\s+(?:vs\.?|versus|v)\s+|\s*,\s*|\s+-\s+", line.strip(), flags=re.I)
    return parts if len(parts) == 2 and all(parts) else None


def cmd_chat(args):
    db, model, base = _context(args)
    print(f"Loaded model ({model['n_base']} base samples, {model['n_feedback']} corrections). "
          f"Record so far: {accuracy_line(load_feedback())}")
    print("Type a matchup like  PRX vs NRG   (or 'teams', 'q' to quit).")
    while True:
        try:
            line = input("\nMatchup> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            return
        if line.lower() == "teams":
            cmd_teams(args, db)
            continue
        pair = _parse_matchup(line)
        if not pair:
            print("  Use the form  Team A vs Team B")
            continue
        try:
            pred = predict_pair(db, model, db.resolve(pair[0]), db.resolve(pair[1]))
        except KeyError as e:
            print(f"  {e.args[0]}")
            continue
        show_prediction(pred)
        ask_verdict(db, model, base, pred)


def cmd_learn(args):
    db, model, base = _context(args)
    ka, kb, kw = db.resolve(args.team_a), db.resolve(args.team_b), db.resolve(args.winner)
    if kw not in (ka, kb):
        sys.exit("--winner must be one of the two teams.")
    apply_feedback(db, model, base, predict_pair(db, model, ka, kb), kw)


def cmd_history(args):
    fb = load_feedback()
    if not fb:
        print("No corrections logged yet.")
        return
    for r in fb:
        mark = "OK " if r["correct"] else "MISS"
        print(f"  {r['ts']}  [{mark}] {r['team_a']} vs {r['team_b']}: "
              f"picked {r['predicted']} ({max(r['prob_a'], 1 - r['prob_a']):.0%}), winner {r['winner']}")
    print(f"\n  {accuracy_line(fb)}")


def cmd_teams(args, db: TeamDB | None = None):
    if db is None:
        data_dir = Path(args.data) if args.data else (
            Path(json.loads(MODEL_PATH.read_text())["data_dir"]) if MODEL_PATH.exists() else DEFAULT_DATA_DIR)
        db = load_db(data_dir)
    print(f"\n  {'Tag':<8}{'Team':<28}{'Country':<16}Matches")
    for key, info in sorted(db.dataset.items(), key=lambda kv: kv[1]["name"].lower()):
        print(f"  {info['tag']:<8}{info['name']:<28}{info['country']:<16}{len(db.rows[key])}")


def main():
    ap = argparse.ArgumentParser(description="VCT Champions match predictor")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--data", help="folder with the team JSONs (default: ./data next to the script)")

    p = sub.add_parser("train", help="build the model from the JSONs")
    common(p)
    p.add_argument("--min-matches", type=int, default=MIN_OTHER_MATCHES,
                   help="min other matches a team needs to be used as a training sample")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="predict one matchup, then give right/wrong")
    common(p)
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--no-ask", action="store_true", help="don't prompt for feedback")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("chat", help="interactive predict + feedback loop")
    common(p)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("learn", help="log a result you already know")
    common(p)
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--winner", required=True)
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("history", help="show logged corrections and accuracy")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("teams", help="list teams")
    common(p)
    p.set_defaults(fn=cmd_teams)

    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["chat"])
    args.fn(args)


if __name__ == "__main__":
    main()
