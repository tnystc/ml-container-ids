"""
Label naming and grouping presets for multi-class attack classification.

`ORIG_LABEL_NAMES` maps original dataset label IDs to human-readable names.
`LABEL_CONFIGS` holds the grouping/dropping presets used in the
label-granularity ablation (see PROJECT_REPORT.md).

`apply_label_config` applies a preset *after* the per-original-class few-shot
train/test split, so each original class still gets `--train-n` samples and
supergroups inherit the union of their constituents' samples.
"""

import numpy as np

ORIG_LABEL_NAMES = {
    0: "Normal",
    1: "Grafana SSRF",
    2: "Node-RED Recon",
    3: "Node-RED RCE",
    4: "Node-RED Escape",
    6: "InfluxDB JWT",
    7: "runc race",
    8: "kubelet symlink",
    11: "Nuclei scanner",
}

# Each preset maps original-label IDs to a "drop" set and a
# {orig_label: supergroup_idx} grouping, plus the supergroup display names.
LABEL_CONFIGS = {
    "none": None,

    # Drop the three failing rare classes (RCE, Escape, runc race).
    "drop3": {
        "drop": [3, 4, 7],
        "groups": {0: 0, 1: 1, 2: 2, 6: 3, 8: 4, 11: 5},
        "names": ["Normal", "Grafana SSRF", "Node-RED Recon",
                  "InfluxDB JWT", "kubelet symlink", "Nuclei scanner"],
    },

    # Kill-chain stage grouping (4 classes).
    "killchain": {
        "drop": [],
        "groups": {
            0: 0,                          # Normal
            2: 1, 11: 1,                   # Reconnaissance
            1: 2, 6: 2, 3: 2,              # Exploitation
            4: 3, 7: 3, 8: 3,              # Container Escape
        },
        "names": ["Normal", "Reconnaissance", "Exploitation", "Container Escape"],
    },

    # Group by attacked service (6 classes).
    "by-service": {
        "drop": [],
        "groups": {
            0: 0,                          # Normal
            2: 1, 3: 1, 4: 1,              # Node-RED activity
            1: 2,                          # Grafana
            6: 3,                          # InfluxDB
            7: 4, 8: 4,                    # Container Runtime
            11: 5,                         # Nuclei scanning
        },
        "names": ["Normal", "Node-RED", "Grafana", "InfluxDB",
                  "Container Runtime", "Nuclei scanner"],
    },

    # Merge only the failing rare classes into one supergroup (7 classes).
    "group-failing": {
        "drop": [],
        "groups": {
            0: 0,                          # Normal
            1: 1,                          # Grafana SSRF
            2: 2,                          # Node-RED Recon
            3: 3, 4: 3, 7: 3,              # Container Compromise (RCE+Escape+runc)
            6: 4,                          # InfluxDB JWT
            8: 5,                          # kubelet symlink
            11: 6,                         # Nuclei scanner
        },
        "names": ["Normal", "Grafana SSRF", "Node-RED Recon",
                  "Container Compromise", "InfluxDB JWT",
                  "kubelet symlink", "Nuclei scanner"],
    },

    # Drop the two rarest only (Escape + runc race), keep RCE.
    "drop-2rarest": {
        "drop": [4, 7],
        "groups": {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 8: 5, 11: 6},
        "names": ["Normal", "Grafana SSRF", "Node-RED Recon", "Node-RED RCE",
                  "InfluxDB JWT", "kubelet symlink", "Nuclei scanner"],
    },
}


def apply_label_config(X_train, y_train, X_test, y_test, label_map, config_name):
    """Apply a label-grouping preset (after few-shot sampling).

    Translates the preset (defined in original-label space) to remapped-label
    space using `label_map`, drops the configured classes, and remaps the
    remaining rows to their supergroup index.

    Returns: X_train, y_train, X_test, y_test, target_names
    """
    if config_name == "none" or LABEL_CONFIGS[config_name] is None:
        inv = {v: k for k, v in label_map.items()}
        names = [ORIG_LABEL_NAMES.get(inv[i], f"class_{i}")
                 for i in range(len(label_map))]
        return X_train, y_train, X_test, y_test, names

    cfg = LABEL_CONFIGS[config_name]
    drop_remapped = {label_map[o] for o in cfg["drop"] if o in label_map}
    groups_remapped = {label_map[o]: g for o, g in cfg["groups"].items()
                       if o in label_map}

    def _transform(X, y):
        keep = ~np.isin(y, list(drop_remapped))
        X_kept = X[keep]
        y_new = np.array([groups_remapped[int(yi)] for yi in y[keep]],
                         dtype=np.int64)
        return X_kept, y_new

    X_tr, y_tr = _transform(X_train, y_train)
    X_te, y_te = _transform(X_test, y_test)
    return X_tr, y_tr, X_te, y_te, cfg["names"]
