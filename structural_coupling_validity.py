# ==========================================================
# Structural Coupling Validity (SCV) Metric
# ==========================================================
# "Regression doesn't explain — it recites numbers."
#
# Traditional causal benchmarks ask: "does your graph match
# the ground truth?" This penalises methods that detect
# genuine structural couplings not in the causal graph
# (e.g., confounded co-movement).
#
# SCV asks a different question: "is each detected edge
# backed by independent numerical evidence?"
#
# This is the same distinction as:
#   Gate & Channel: "is the molecular mechanism disrupted?"
#   ClinVar:        "does the patient get sick?"
#
# Both are correct. They measure different things.
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# ==========================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


# ===================================================================
#  EVIDENCE DIMENSIONS
# ===================================================================
#
# For each detected edge, we check independent lines of evidence:
#
#  1. Effect Size      — is β large relative to noise floor?
#  2. CI Exclusion     — does the bootstrap CI exclude zero?
#  3. Decomposability  — can we explain WHY (pos/neg/abs/stress)?
#  4. Sync Evidence    — does event synchronization corroborate?
#  5. Asymmetry        — is the direction backed by β_fwd >> β_bwd?
#
# VAR can answer (1) and (2). That's it.
# Λ³ can answer all five.
# ===================================================================


def compute_scv_lambda3(
    adjacency_bin: np.ndarray,
    scores: np.ndarray,
    meta: Dict[str, Any],
    pair_details: Optional[Dict[str, Dict[str, Any]]] = None,
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Structural Coupling Validity for Λ³ outputs.

    For each detected edge, evaluates how many independent lines
    of numerical evidence support the coupling.

    Parameters
    ----------
    adjacency_bin : (n, n) array — binary adjacency.
    scores : (n, n) array — β_total or similar score.
    meta : dict — adapter metadata (edge_threshold, etc.)
    pair_details : dict, optional — per-pair decomposed β values.
        Keys: "A_B" → {"beta_pos": .., "beta_neg": .., "beta_abs": ..,
                        "beta_stress": .., "sync_rate": .., "ci_excludes_zero": bool}
    names : list of str — series names.

    Returns
    -------
    dict with:
        scv_score : float [0, 1] — average validity across detected edges.
        n_edges_detected : int
        n_edges_valid : int — edges with SCV ≥ 0.5
        per_edge_detail : list of dicts
        evidence_depth_mean : float — average number of evidence lines per edge
    """
    n = adjacency_bin.shape[0]
    if names is None:
        names = [f"X{i}" for i in range(n)]

    detected_edges = []
    for i in range(n):
        for j in range(n):
            if i != j and adjacency_bin[i, j] == 1:
                detected_edges.append((i, j))

    if not detected_edges:
        return {
            "scv_score": 1.0,  # No edges = no invalid claims
            "n_edges_detected": 0,
            "n_edges_valid": 0,
            "per_edge_detail": [],
            "evidence_depth_mean": 0.0,
        }

    edge_details = []
    total_scv = 0.0

    for (i, j) in detected_edges:
        edge_name = f"{names[i]}_to_{names[j]}"
        evidence = {}
        n_evidence = 0
        n_supported = 0

        # --- (1) Effect size: is β above noise floor? ---
        beta_total = scores[i, j] if scores is not None else 0.0
        threshold = meta.get("edge_threshold", 0.0)

        evidence["effect_size"] = {
            "value": float(beta_total),
            "threshold": float(threshold),
            "supported": beta_total > threshold * 0.5,  # At least half of threshold
        }
        n_evidence += 1
        if evidence["effect_size"]["supported"]:
            n_supported += 1

        # --- (2-5) Require pair_details ---
        if pair_details and edge_name in pair_details:
            pd_ = pair_details[edge_name]

            # --- (2) CI exclusion: does bootstrap CI exclude zero? ---
            ci_excl = pd_.get("ci_excludes_zero", False)
            evidence["ci_exclusion"] = {
                "supported": ci_excl,
                "method": "bootstrap_hdi",
            }
            n_evidence += 1
            if ci_excl:
                n_supported += 1

            # --- (3) Decomposability: can we explain WHY? ---
            beta_pos = abs(pd_.get("beta_pos", 0))
            beta_neg = abs(pd_.get("beta_neg", 0))
            beta_abs = abs(pd_.get("beta_abs", 0))
            beta_stress = abs(pd_.get("beta_stress", 0))

            active_channels = sum([
                beta_pos > 0.1,
                beta_neg > 0.1,
                beta_abs > 0.1,
                beta_stress > 0.1,
            ])

            evidence["decomposability"] = {
                "beta_pos": float(beta_pos),
                "beta_neg": float(beta_neg),
                "beta_abs": float(beta_abs),
                "beta_stress": float(beta_stress),
                "active_channels": active_channels,
                "supported": active_channels >= 1,
                "explanation": _generate_explanation(
                    beta_pos, beta_neg, beta_abs, beta_stress),
            }
            n_evidence += 1
            if evidence["decomposability"]["supported"]:
                n_supported += 1

            # --- (4) Sync corroboration ---
            sync = pd_.get("sync_rate", None)
            if sync is not None:
                evidence["sync_corroboration"] = {
                    "value": float(sync),
                    "supported": sync > 0.01,  # Any non-trivial sync
                    "note": "low sync + high β_abs = hidden causality (valid!)",
                }
                n_evidence += 1
                # Both high-sync AND low-sync-with-high-β_abs are valid
                if sync > 0.01 or beta_abs > 0.3:
                    n_supported += 1

            # --- (5) Asymmetry evidence ---
            reverse_key = f"{names[j]}_to_{names[i]}"
            if reverse_key in pair_details:
                rev = pair_details[reverse_key]
                rev_total = (abs(rev.get("beta_pos", 0))
                             + abs(rev.get("beta_neg", 0))
                             + abs(rev.get("beta_abs", 0))
                             + abs(rev.get("beta_stress", 0)))
                fwd_total = beta_pos + beta_neg + beta_abs + beta_stress
                ratio = fwd_total / (rev_total + 1e-8)

                evidence["asymmetry"] = {
                    "forward_total": float(fwd_total),
                    "reverse_total": float(rev_total),
                    "ratio": float(ratio),
                    "supported": ratio > 1.5,
                    "note": "ratio > 1.5 = clear directional evidence",
                }
                n_evidence += 1
                if ratio > 1.5:
                    n_supported += 1

        # --- Edge SCV ---
        edge_scv = n_supported / n_evidence if n_evidence > 0 else 0.0

        edge_details.append({
            "edge": edge_name,
            "scv": float(edge_scv),
            "n_evidence": n_evidence,
            "n_supported": n_supported,
            "evidence": evidence,
        })
        total_scv += edge_scv

    n_detected = len(detected_edges)
    mean_scv = total_scv / n_detected if n_detected > 0 else 0.0
    n_valid = sum(1 for e in edge_details if e["scv"] >= 0.5)
    mean_depth = np.mean([e["n_evidence"] for e in edge_details])

    return {
        "scv_score": float(mean_scv),
        "n_edges_detected": n_detected,
        "n_edges_valid": n_valid,
        "evidence_depth_mean": float(mean_depth),
        "per_edge_detail": edge_details,
    }


def compute_scv_baseline(
    adjacency_bin: np.ndarray,
    method_name: str,
    scores: Optional[np.ndarray] = None,
    pvalues: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute SCV for baseline methods (VAR, TE, PCMCI, GL).

    Baselines can only provide effect_size and CI_exclusion
    (via p-value). They cannot decompose, corroborate via sync,
    or explain WHY a coupling exists.

    Parameters
    ----------
    adjacency_bin : (n, n) — binary adjacency.
    method_name : str
    scores : (n, n) — effect sizes or -log10(p).
    pvalues : (n, n), optional — p-values if available.
    names : list of str

    Returns
    -------
    dict — same structure as compute_scv_lambda3.
    """
    n = adjacency_bin.shape[0]
    if names is None:
        names = [f"X{i}" for i in range(n)]

    detected_edges = []
    for i in range(n):
        for j in range(n):
            if i != j and adjacency_bin[i, j] == 1:
                detected_edges.append((i, j))

    if not detected_edges:
        return {
            "scv_score": 1.0,
            "n_edges_detected": 0,
            "n_edges_valid": 0,
            "evidence_depth_mean": 0.0,
            "per_edge_detail": [],
        }

    # Determine max possible evidence lines for this method
    method_capabilities = {
        "VAR_Granger":     {"effect_size": True,  "ci": True,  "decompose": False, "sync": False, "asymmetry": True},
        "TransferEntropy":  {"effect_size": True,  "ci": True,  "decompose": False, "sync": False, "asymmetry": True},
        "PCMCIPlus":        {"effect_size": True,  "ci": True,  "decompose": False, "sync": False, "asymmetry": True},
        "GraphicalLasso":   {"effect_size": True,  "ci": False, "decompose": False, "sync": False, "asymmetry": False},
        "EventXCorr":       {"effect_size": True,  "ci": True,  "decompose": False, "sync": False, "asymmetry": True},
    }

    caps = method_capabilities.get(method_name, {"effect_size": True, "ci": False, "decompose": False, "sync": False, "asymmetry": False})

    edge_details = []
    total_scv = 0.0

    for (i, j) in detected_edges:
        edge_name = f"{names[i]}_to_{names[j]}"
        n_evidence = 0
        n_supported = 0
        evidence = {}

        # (1) Effect size
        if caps["effect_size"] and scores is not None:
            val = scores[i, j]
            evidence["effect_size"] = {
                "value": float(val),
                "supported": val > 0,
            }
            n_evidence += 1
            if val > 0:
                n_supported += 1

        # (2) CI / p-value
        if caps["ci"] and pvalues is not None:
            pval = pvalues[i, j]
            evidence["ci_exclusion"] = {
                "pvalue": float(pval),
                "supported": pval < 0.05,
                "method": "frequentist_p",
            }
            n_evidence += 1
            if pval < 0.05:
                n_supported += 1

        # (3) Decomposability — CANNOT
        evidence["decomposability"] = {
            "supported": False,
            "explanation": f"{method_name} reports a single coefficient. "
                           f"Cannot decompose into structural components "
                           f"(jump type, tension, absolute coupling).",
            "active_channels": 0,
        }
        n_evidence += 1
        # n_supported stays the same — decomposability always fails for baselines

        # (4) Sync — CANNOT
        evidence["sync_corroboration"] = {
            "supported": False,
            "note": f"{method_name} does not compute event synchronization.",
        }
        n_evidence += 1

        # (5) Asymmetry
        if caps["asymmetry"] and scores is not None:
            fwd = scores[i, j]
            bwd = scores[j, i] if j < scores.shape[0] and i < scores.shape[1] else 0.0
            ratio = fwd / (bwd + 1e-8)
            evidence["asymmetry"] = {
                "forward": float(fwd),
                "reverse": float(bwd),
                "ratio": float(ratio),
                "supported": ratio > 1.5,
            }
            n_evidence += 1
            if ratio > 1.5:
                n_supported += 1

        edge_scv = n_supported / n_evidence if n_evidence > 0 else 0.0

        edge_details.append({
            "edge": edge_name,
            "scv": float(edge_scv),
            "n_evidence": n_evidence,
            "n_supported": n_supported,
            "evidence": evidence,
        })
        total_scv += edge_scv

    n_detected = len(detected_edges)
    mean_scv = total_scv / n_detected if n_detected > 0 else 0.0
    n_valid = sum(1 for e in edge_details if e["scv"] >= 0.5)
    mean_depth = np.mean([e["n_evidence"] for e in edge_details])

    return {
        "scv_score": float(mean_scv),
        "n_edges_detected": n_detected,
        "n_edges_valid": n_valid,
        "evidence_depth_mean": float(mean_depth),
        "per_edge_detail": edge_details,
    }


# ===================================================================
#  EXPLANATION GENERATOR
# ===================================================================

def _generate_explanation(
    beta_pos: float,
    beta_neg: float,
    beta_abs: float,
    beta_stress: float,
    threshold: float = 0.1,
) -> str:
    """Generate human-readable explanation of WHY a coupling exists.

    This is what regression cannot do.
    Regression says: "β = 1.35"
    Λ³ says: "positive jumps propagate with tension co-resonance"
    """
    parts = []

    if beta_pos > threshold and beta_neg > threshold:
        parts.append("bidirectional jump propagation (both positive and negative ΔΛC)")
    elif beta_pos > threshold:
        parts.append("positive structural jumps propagate")
    elif beta_neg > threshold:
        parts.append("negative structural jumps propagate")

    if beta_abs > threshold:
        if beta_pos < threshold and beta_neg < threshold:
            parts.append("ΛF-invariant coupling (phase-modulated hidden causality)")
        else:
            parts.append("absolute event intensity coupling")

    if beta_stress > threshold:
        parts.append("tension (ρT) co-resonance")

    if not parts:
        return "weak coupling — no dominant mechanism"

    return " + ".join(parts)


# ===================================================================
#  COMPARISON REPORT
# ===================================================================

def print_scv_comparison(
    results: Dict[str, Dict[str, Any]],
    scenario_name: str = "",
) -> None:
    """Print side-by-side SCV comparison across methods.

    Highlights the evidence depth gap between Λ³ and baselines.
    """
    print(f"\n{'=' * 70}")
    print(f"Structural Coupling Validity — {scenario_name}")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'SCV':>6} {'Edges':>6} {'Valid':>6} "
          f"{'Depth':>6} {'Decompose':>10}")
    print("-" * 70)

    for method, scv in sorted(results.items(),
                               key=lambda kv: kv[1]["scv_score"],
                               reverse=True):
        decomp = "N/A"
        for ed in scv.get("per_edge_detail", []):
            d = ed.get("evidence", {}).get("decomposability", {})
            if d.get("supported"):
                decomp = f"{d['active_channels']} channels"
                break
            elif d.get("explanation", ""):
                decomp = "cannot"

        print(f"{method:<20} {scv['scv_score']:>6.3f} "
              f"{scv['n_edges_detected']:>6} "
              f"{scv['n_edges_valid']:>6} "
              f"{scv['evidence_depth_mean']:>6.1f} "
              f"{decomp:>10}")

    # The punchline
    lambda3_scv = results.get("Lambda3", {})
    if lambda3_scv and lambda3_scv.get("per_edge_detail"):
        print(f"\nΛ³ Edge Explanations:")
        for ed in lambda3_scv["per_edge_detail"]:
            decomp = ed["evidence"].get("decomposability", {})
            expl = decomp.get("explanation", "N/A")
            print(f"  {ed['edge']}: {expl}")

        # Find a baseline for contrast
        for method, scv in results.items():
            if method != "Lambda3" and scv.get("per_edge_detail"):
                print(f"\n{method} Edge Explanations:")
                for ed in scv["per_edge_detail"]:
                    decomp = ed["evidence"].get("decomposability", {})
                    expl = decomp.get("explanation", "N/A")
                    print(f"  {ed['edge']}: {expl}")
                break  # Just one baseline for contrast


def aggregate_scv_across_scenarios(
    all_scv: Dict[str, Dict[str, Dict[str, Any]]],
) -> pd.DataFrame:
    """Aggregate SCV scores across all scenarios.

    Parameters
    ----------
    all_scv : {scenario_name: {method_name: scv_result}}

    Returns
    -------
    pd.DataFrame with mean SCV and evidence depth per method.
    """
    rows = []
    for scenario, methods in all_scv.items():
        for method, scv in methods.items():
            rows.append({
                "scenario": scenario,
                "method": method,
                "scv_score": scv["scv_score"],
                "n_edges": scv["n_edges_detected"],
                "n_valid": scv["n_edges_valid"],
                "evidence_depth": scv["evidence_depth_mean"],
            })

    df = pd.DataFrame(rows)

    # Summary per method
    summary = df.groupby("method").agg({
        "scv_score": "mean",
        "n_edges": "sum",
        "n_valid": "sum",
        "evidence_depth": "mean",
    }).sort_values("scv_score", ascending=False)

    return summary


# ===================================================================
#  INTEGRATION HOOK
# ===================================================================

def evaluate_scv_for_output(output, method_name: str, meta: Dict = None) -> Dict[str, Any]:
    """Convenience function to compute SCV from a MethodOutput.

    Call this inside the benchmark's evaluate_shared_tasks() or
    after it, using the same MethodOutput object.
    """
    adj = output.adjacency_bin
    if adj is None:
        return {"scv_score": float("nan"), "n_edges_detected": 0,
                "n_edges_valid": 0, "evidence_depth_mean": 0.0,
                "per_edge_detail": []}

    if method_name == "Lambda3":
        return compute_scv_lambda3(
            adjacency_bin=adj,
            scores=output.adjacency_scores,
            meta=output.meta or {},
            pair_details=output.meta.get("pair_details") if output.meta else None,
            names=output.names,
        )
    else:
        pvals = None
        if output.meta and "pvalues" in output.meta:
            pvals = output.meta["pvalues"]
        return compute_scv_baseline(
            adjacency_bin=adj,
            method_name=method_name,
            scores=output.adjacency_scores,
            pvalues=pvals,
            names=output.names,
        )
