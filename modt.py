import heapq
import time

import numpy as np
import pandas as pd
import sklearn.datasets as skd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logger
from read_data import read_dataset


class ObliqueNode:
    def __init__(self, left=None, right=None, svm=None, label=None, node_id=None):
        self.left = left
        self.right = right
        self.svm = svm
        self.label = label
        self.node_id = node_id


class MergeObliqueTree:
    def __init__(self, smart=True):
        self.root = None
        self._node_counter = 0
        self.smart = smart

    def _calc_distance(self, node_a, node_b):
        return np.linalg.norm(node_a["centroid"] - node_b["centroid"])

    def fit(self, X, y):
        unique_labels = np.unique(y)
        logger.info(f"Starting training with {len(unique_labels)} unique classes.")

        # 1. Initialize nodes for each class
        active_nodes = {}
        heap = []

        for label in unique_labels:
            indices = np.where(y == label)[0]

            if (not self.smart) or (len(indices) <= 5):
                node_id = self._next_id()
                node_info = {
                    "node": ObliqueNode(label=label, node_id=node_id),
                    "centroid": np.mean(X[indices], axis=0),
                    "data": X[indices],
                    "id": node_id,
                    "active": True,
                }
                active_nodes[node_id] = node_info
                if self.smart:
                    logger.info(
                        f"Class {label}: Only {len(indices)} samples, creating single node."
                    )
                continue

            else:
                hdb = HDBSCAN(min_cluster_size=5, allow_single_cluster=True, copy=True)
                clusters = hdb.fit_predict(X[indices])
                unique_labels = np.unique(clusters)

                logger.info(
                    f"Class {label}: Found {len(unique_labels)} clusters using HDBSCAN."
                )

                for cluster_label in unique_labels:
                    cluster_mask = clusters == cluster_label
                    cluster_indices = indices[cluster_mask]
                    if np.sum(cluster_indices) == 0:
                        continue
                    node_id = self._next_id()
                    node_info = {
                        "node": ObliqueNode(label=label, node_id=node_id),
                        "centroid": np.mean(X[cluster_indices], axis=0),
                        "data": X[cluster_indices],
                        "id": node_id,
                        "active": True,
                    }
                    active_nodes[node_id] = node_info

        # 2. Build initial heap (all-pairs distance)
        node_ids = list(active_nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id_a, id_b = node_ids[i], node_ids[j]
                # skip same class merges
                if active_nodes[id_a]["node"].label == active_nodes[id_b]["node"].label:
                    continue
                dist = self._calc_distance(active_nodes[id_a], active_nodes[id_b])
                heapq.heappush(heap, (dist, id_a, id_b))

        logger.info(f"Initialized heap with {len(heap)} potential edges.")

        # 3. Agglomerative Loop
        while len(active_nodes) > 1:
            if not heap:
                break

            dist, id_a, id_b = heapq.heappop(heap)
            if id_a not in active_nodes or id_b not in active_nodes:
                continue

            node_a = active_nodes.pop(id_a)
            node_b = active_nodes.pop(id_b)
            if (node_a["node"].label is not None) and (
                node_a["node"].label == node_b["node"].label
            ):
                continue

            logger.info(f"Merging Node {id_a} and Node {id_b} (Distance: {dist:.4f})")

            # Train SVM for this split
            merged_X = np.vstack([node_a["data"], node_b["data"]])
            merged_y = np.array([0] * len(node_a["data"]) + [1] * len(node_b["data"]))

            clf = LinearSVC(dual="auto")
            clf.fit(merged_X, merged_y)

            # Create Parent
            new_id = self._next_id()
            parent_node = ObliqueNode(
                left=node_a["node"], right=node_b["node"], svm=clf, node_id=new_id
            )

            # New node metadata
            new_node_info = {
                "node": parent_node,
                "centroid": np.mean(merged_X, axis=0),
                "data": merged_X,
                "id": new_id,
            }

            # Update heap with distances from the new node to all other active nodes
            for other_id, other_node in active_nodes.items():
                new_dist = self._calc_distance(new_node_info, other_node)
                heapq.heappush(heap, (new_dist, new_id, other_id))

            active_nodes[new_id] = new_node_info

        self.root = list(active_nodes.values())[0]["node"]

    def _next_id(self):
        self._node_counter += 1
        return self._node_counter

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.label is not None:
            return node.label

        decision = node.svm.predict(x.reshape(1, -1))[0]
        if decision == 0:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


def test_dataset(name="Dataset", cmp_dt=True, cmp_xgb=True):
    logger.info(f"--- Benchmarking on {name} ---")
    X, y = read_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    results = []

    # Proposed ODT
    proposed_tree = MergeObliqueTree()
    start = time.time()
    proposed_tree.fit(X_train, y_train)
    train_time = time.time() - start
    logger.success("mODT (smart) construction complete.")
    proposed_preds = proposed_tree.predict(X_test)
    results.append(
        {
            "Model": "Proposed (Smart)",
            "Accuracy": accuracy_score(y_test, proposed_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # Proposed ODT not smart
    proposed_tree = MergeObliqueTree(smart=False)
    start = time.time()
    proposed_tree.fit(X_train, y_train)
    train_time = time.time() - start
    logger.success("mODT construction complete.")
    proposed_preds = proposed_tree.predict(X_test)
    results.append(
        {
            "Model": "Proposed (Vanilla)",
            "Accuracy": accuracy_score(y_test, proposed_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # Traditional Decision Tree (Scikit-Learn)
    if cmp_dt:
        dt = DecisionTreeClassifier(max_depth=16)
        start = time.time()
        dt.fit(X_train, y_train)
        train_time = time.time() - start
        logger.success("Decision Tree training complete.")
        dt_preds = dt.predict(X_test)
        results.append(
            {
                "Model": "CART",
                "Accuracy": accuracy_score(y_test, dt_preds),
                "Train Time (s)": round(train_time, 4),
            }
        )

    # XGBoost
    if cmp_xgb:
        xgb = XGBClassifier(eval_metric="mlogloss", n_estimators=100)
        start = time.time()
        xgb.fit(X_train, y_train)
        train_time = time.time() - start
        logger.success("XGBoost training complete.")
        xgb_preds = xgb.predict(X_test)
        results.append(
            {
                "Model": "XGBoost",
                "Accuracy": accuracy_score(y_test, xgb_preds),
                "Train Time (s)": round(train_time, 4),
            }
        )

    # Display results
    df = pd.DataFrame(results)
    logger.info(f"\nResults for {name}:")
    logger.info(f"\n{df.to_string(index=False)}")
    return df


# --- Test Run ---
if __name__ == "__main__":
    for ds in ["heart", "poker", "sensorless"]:
        test_dataset(name=ds)
