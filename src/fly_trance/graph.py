"""Load and verify the original LiquidAion CX512 export."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def project_root():
    return Path(__file__).resolve().parents[2]


def data_root():
    return Path(__file__).resolve().parent / "data"


def load_graph(directory=None):
    directory = Path(directory or data_root() / "cx512")
    manifest = json.loads((directory / "provenance.json").read_text())
    for name, key in (("nodes.csv", "node_sha256"), ("edges.csv", "edge_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"Connectome hash mismatch: {name}")
    with (directory / "nodes.csv").open() as f:
        nodes = list(csv.DictReader(f))
    ids = [row["root_id"] for row in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate neuron IDs")
    index = {root: i for i, root in enumerate(ids)}
    with (directory / "edges.csv").open() as f:
        edges = list(csv.DictReader(f))
    graph = sparse.csr_matrix((
        [float(row["syn_count"]) for row in edges],
        ([index[row["post_root_id"]] for row in edges],
         [index[row["pre_root_id"]] for row in edges]),
    ), shape=(len(ids), len(ids)))
    if graph.nnz != manifest["edges"] or not np.isfinite(graph.data).all():
        raise ValueError("Invalid connectome edges")
    return graph, nodes, manifest
