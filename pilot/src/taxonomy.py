"""Hand-written 3-level ESG taxonomy and centroid-based topic assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SEED
from util import percentile


# root → E/S/G → 14 climate-relevant leaves.
TAXONOMY_LEAVES: list[dict] = [
    {
        "id": "e_physical_climate_risk",
        "pillar": "E",
        "name": "Physical climate risk",
        "description": (
            "Physical climate hazards and their effects on assets, operations, and supply chains, "
            "including extreme weather, flooding, heat, drought, and chronic climate impacts."
        ),
    },
    {
        "id": "e_transition_risk",
        "pillar": "E",
        "name": "Transition risk",
        "description": (
            "Risks from the shift to a low-carbon economy, including carbon pricing, regulation, "
            "stranded assets, changing demand, and technology substitution."
        ),
    },
    {
        "id": "e_climate_adaptation",
        "pillar": "E",
        "name": "Climate adaptation",
        "description": (
            "Actions, plans, and investments that build resilience to climate change impacts, "
            "including adaptation targets, scenario-informed adjustments, and resilience measures."
        ),
    },
    {
        "id": "e_ghg_emissions",
        "pillar": "E",
        "name": "GHG emissions",
        "description": (
            "Greenhouse gas inventories and reduction targets across Scope 1, 2, and 3, including "
            "net-zero, carbon removal, and emissions intensity metrics."
        ),
    },
    {
        "id": "e_energy",
        "pillar": "E",
        "name": "Energy",
        "description": (
            "Energy use, renewable electricity, carbon-free power procurement, energy efficiency, "
            "and the transition of operations and products away from fossil fuels."
        ),
    },
    {
        "id": "e_water",
        "pillar": "E",
        "name": "Water",
        "description": (
            "Water consumption, scarcity, pollution, replenishment, sanitation, and engagement "
            "with partners on water stewardship."
        ),
    },
    {
        "id": "e_biodiversity",
        "pillar": "E",
        "name": "Biodiversity",
        "description": (
            "Impacts and dependencies on ecosystems, land, habitats, and biodiversity, including "
            "conservation, land footprint, and nature-related risk."
        ),
    },
    {
        "id": "e_waste",
        "pillar": "E",
        "name": "Waste and circularity",
        "description": (
            "Waste generation, zero-waste goals, recycling, reuse, circular product design, and "
            "short-term actions to reduce waste."
        ),
    },
    {
        "id": "s_workforce",
        "pillar": "S",
        "name": "Workforce",
        "description": (
            "Employees, green skills, just transition, labor practices, training, and how the "
            "workforce is affected by climate and sustainability strategy."
        ),
    },
    {
        "id": "s_health_safety",
        "pillar": "S",
        "name": "Health and safety",
        "description": (
            "Occupational health and safety, wellbeing, and harm-prevention policies for workers "
            "and people affected by company operations."
        ),
    },
    {
        "id": "s_community",
        "pillar": "S",
        "name": "Community",
        "description": (
            "Community impacts, social license, downstream partners, value-chain engagement, and "
            "effects of business activities on people and society."
        ),
    },
    {
        "id": "g_board_oversight",
        "pillar": "G",
        "name": "Board oversight",
        "description": (
            "Board and executive oversight of climate and sustainability, including governance "
            "structures, accountability, and management ownership of climate topics."
        ),
    },
    {
        "id": "g_risk_management",
        "pillar": "G",
        "name": "Risk management",
        "description": (
            "Processes to identify, assess, and manage climate-related risks and opportunities, "
            "including scenario analysis, third-party scenarios, and enterprise risk integration."
        ),
    },
    {
        "id": "g_ethics",
        "pillar": "G",
        "name": "Ethics",
        "description": (
            "Ethics, conduct, do-no-significant-harm principles, disclosure integrity, and "
            "policies that guide responsible business behavior."
        ),
    },
]

PILLARS = {
    "E": {
        "id": "E",
        "name": "Environmental",
        "description": (
            "Environmental topics: climate risk and adaptation, greenhouse gas emissions, energy, "
            "water, biodiversity, waste, and related ecological impacts of the business."
        ),
    },
    "S": {
        "id": "S",
        "name": "Social",
        "description": (
            "Social topics: workforce, health and safety, communities, and how people in the "
            "value chain are affected by climate and sustainability activities."
        ),
    },
    "G": {
        "id": "G",
        "name": "Governance",
        "description": (
            "Governance topics: board oversight, climate risk management processes, ethics, and "
            "accountability for sustainability targets and disclosures."
        ),
    },
}

ROOT = {
    "id": "root",
    "name": "ESG",
    "description": (
        "Corporate environmental, social, and governance disclosures, with an emphasis on climate "
        "risk, adaptation, emissions, and the people and oversight processes around them."
    ),
}


@dataclass(frozen=True)
class Taxonomy:
    leaf_ids: list[str]
    pillar_ids: list[str]
    leaf_to_pillar: dict[str, str]
    node_ids: list[str]
    node_level: dict[str, str]
    node_texts: list[str]
    node_names: dict[str, str]


def build_taxonomy() -> Taxonomy:
    leaf_ids = [x["id"] for x in TAXONOMY_LEAVES]
    pillar_ids = ["E", "S", "G"]
    leaf_to_pillar = {x["id"]: x["pillar"] for x in TAXONOMY_LEAVES}
    node_ids = ["root"] + pillar_ids + leaf_ids
    node_level = {"root": "root", "E": "pillar", "S": "pillar", "G": "pillar"}
    node_level.update({lid: "leaf" for lid in leaf_ids})
    texts = (
        [ROOT["description"]]
        + [PILLARS[p]["description"] for p in pillar_ids]
        + [x["description"] for x in TAXONOMY_LEAVES]
    )
    names = {ROOT["id"]: ROOT["name"]}
    names.update({p: PILLARS[p]["name"] for p in pillar_ids})
    names.update({x["id"]: x["name"] for x in TAXONOMY_LEAVES})
    return Taxonomy(
        leaf_ids=leaf_ids,
        pillar_ids=pillar_ids,
        leaf_to_pillar=leaf_to_pillar,
        node_ids=node_ids,
        node_level=node_level,
        node_texts=texts,
        node_names=names,
    )


def embed_nodes(model, query_prefix: str | None, tax: Taxonomy) -> np.ndarray:
    texts = list(tax.node_texts)
    if query_prefix:
        # Node centroids are documents, not queries.
        pass
    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return emb


def node_matrices(tax: Taxonomy, node_emb: np.ndarray) -> dict[str, np.ndarray]:
    idx = {nid: i for i, nid in enumerate(tax.node_ids)}
    leaf_idx = np.array([idx[i] for i in tax.leaf_ids], dtype=int)
    pillar_idx = np.array([idx[i] for i in tax.pillar_ids], dtype=int)
    return {
        "all": node_emb,
        "leaf": node_emb[leaf_idx],
        "pillar": node_emb[pillar_idx],
        "leaf_idx": leaf_idx,
        "pillar_idx": pillar_idx,
    }


def choose_taus(chunk_leaf_sim: np.ndarray) -> tuple[list[float], dict]:
    """Pick 3 τ values from the per-chunk second-highest leaf cosine distribution."""
    sims = np.asarray(chunk_leaf_sim, dtype=np.float64)
    top2 = np.partition(sims, -2, axis=1)[:, -2:]
    second = np.min(top2, axis=1)
    first = np.max(sims, axis=1)
    all_flat = sims.ravel()
    taus = [
        float(np.quantile(second, 0.25)),
        float(np.quantile(second, 0.50)),
        float(np.quantile(second, 0.75)),
    ]
    # Keep them strictly increasing; if ties, nudge using first-score quantiles.
    for i in range(1, 3):
        if taus[i] <= taus[i - 1]:
            taus[i] = min(0.999, taus[i - 1] + 0.01)
    dist = {
        "rule": (
            "τ is taken from the per-chunk second-highest leaf cosine "
            "(p25, p50, p75), so the threshold changes multi-leaf assignment. "
            "Every chunk still receives at least its top-1 leaf."
        ),
        "n_chunk_leaf_scores": int(all_flat.size),
        "all_scores": {
            "min": float(all_flat.min()),
            "p25": percentile(all_flat, 25),
            "p50": percentile(all_flat, 50),
            "p75": percentile(all_flat, 75),
            "max": float(all_flat.max()),
        },
        "top1_scores": {
            "min": float(first.min()),
            "p25": percentile(first, 25),
            "p50": percentile(first, 50),
            "p75": percentile(first, 75),
            "max": float(first.max()),
        },
        "second_scores": {
            "min": float(second.min()),
            "p25": percentile(second, 25),
            "p50": percentile(second, 50),
            "p75": percentile(second, 75),
            "max": float(second.max()),
        },
        "tau_values": taus,
        "seed": SEED,
    }
    return taus, dist


def assign_leaves(chunk_leaf_sim: np.ndarray, tau: float) -> np.ndarray:
    """Boolean (n_chunks, n_leaves): cosine >= τ, plus forced top-1."""
    sim = np.asarray(chunk_leaf_sim)
    assigned = sim >= tau
    top1 = np.argmax(sim, axis=1)
    assigned[np.arange(sim.shape[0]), top1] = True
    return assigned


def leaf_to_pillar_membership(leaf_mem: np.ndarray, tax: Taxonomy) -> np.ndarray:
    pillar_mem = np.zeros((leaf_mem.shape[0], len(tax.pillar_ids)), dtype=bool)
    for j, leaf_id in enumerate(tax.leaf_ids):
        p = tax.pillar_ids.index(tax.leaf_to_pillar[leaf_id])
        pillar_mem[:, p] |= leaf_mem[:, j]
    return pillar_mem


def membership_for_level(leaf_mem: np.ndarray, tax: Taxonomy, level: str) -> np.ndarray:
    if level == "leaf":
        return leaf_mem
    if level == "pillar":
        return leaf_to_pillar_membership(leaf_mem, tax)
    raise ValueError(level)


def route_top_lambda(query_node_sim: np.ndarray, lam: int) -> np.ndarray:
    """Return indices of the top-λ nodes (highest cosine)."""
    sim = np.asarray(query_node_sim)
    k = min(int(lam), sim.size)
    if k <= 0:
        return np.array([], dtype=int)
    if k == sim.size:
        return np.argsort(-sim)
    idx = np.argpartition(-sim, kth=k - 1)[:k]
    return idx[np.argsort(-sim[idx])]


def chunk_assignment_stats(leaf_mem: np.ndarray, tax: Taxonomy) -> dict:
    per_leaf = leaf_mem.sum(axis=0)
    leaves_per_chunk = leaf_mem.sum(axis=1)
    return {
        "chunks_per_leaf": {
            tax.leaf_ids[i]: int(per_leaf[i]) for i in range(len(tax.leaf_ids))
        },
        "mean_leaves_per_chunk": float(leaves_per_chunk.mean()) if len(leaves_per_chunk) else None,
        "min_leaves_per_chunk": int(leaves_per_chunk.min()) if len(leaves_per_chunk) else None,
        "max_leaves_per_chunk": int(leaves_per_chunk.max()) if len(leaves_per_chunk) else None,
        "n_chunks": int(leaf_mem.shape[0]),
    }
