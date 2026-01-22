import heapq
import time

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logger
from read_data import read_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ObliqueNode:
    def __init__(self, left=None, right=None, svm=None, label=None, node_id=None):
        self.left = left
        self.right = right
        self.svm = svm
        self.label = label
        self.node_id = node_id


class MergeObliqueTree:
    def __init__(self, dist_weight=0.9, cluster_strategy="adaptive", verbose=False):
        self.root = None
        self._node_counter = 0
        self.dist_weight = dist_weight
        self.cluster_strategy = cluster_strategy
        self.verbose = verbose

    def _log(self, message):
        if self.verbose:
            self._log(message)

    def _calc_distance(self, node_a, node_b):
        w1 = sigmoid(np.linalg.norm(node_a["centroid"] - node_b["centroid"]))
        w2 = abs(len(node_a["data"]) - len(node_b["data"])) / (
            len(node_a["data"]) + len(node_b["data"])
        )
        return self.dist_weight * w1 + (1 - self.dist_weight) * w2

    def _should_cluster(self, X_class, label, total_samples):
        """Decide whether to cluster based on adaptive criteria"""
        if self.cluster_strategy == "none":
            return False

        n_samples = len(X_class)

        if n_samples < 10:
            return False

        if self.cluster_strategy == "fixed":
            return True

        n_features = X_class.shape[1]

        # Adaptive strategy
        # 1. Don't cluster very small classes (not enough data)
        # 2. For very large classes relative to features, clustering helps
        samples_per_feature = n_samples / n_features
        if samples_per_feature > 50:
            return True

        # 3. Check if class is actually multi-modal using silhouette heuristic
        # Sample for efficiency on large datasets
        sample_size = min(n_samples, 500)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X_class[indices]
        else:
            X_sample = X_class

        # Try quick clustering to assess modality
        try:
            if len(X_sample) < 10:
                return False

            kmeans = KMeans(n_clusters=min(2, len(X_sample) // 2), n_init=3)
            labels = kmeans.fit_predict(X_sample)

            # If only one cluster found, don't cluster
            if len(np.unique(labels)) < 2:
                return False

            sil_score = silhouette_score(X_sample, labels)
            return sil_score > 0.3

        except Exception as e:
            logger.warning(f"Silhouette check failed: {e}, defaulting to no clustering")
            return False

    def get_label_clusters(self, X, y):
        unique_labels = np.unique(y)
        active_nodes = {}

        self._log(f"Starting training with {len(unique_labels)} unique classes.")
        for label in unique_labels:
            indices = np.where(y == label)[0]
            X_class = X[indices]
            should_cluster = self._should_cluster(X_class, label, len(X))
            if not should_cluster:
                node_id = self._next_id()
                node_info = {
                    "node": ObliqueNode(label=label, node_id=node_id),
                    "centroid": np.mean(X_class, axis=0),
                    "data": set(indices),
                    "id": node_id,
                    "active": True,
                }
                active_nodes[node_id] = node_info
                self._log(
                    f"Class {label}: {len(indices)} samples, creating single node (no clustering)"
                )
                continue

            # Adaptive clustering parameters
            min_cluster_size = max(3, int(len(indices) * 0.02))  # At least 2% of class
            min_cluster_size = min(min_cluster_size, 20)  # Cap at 20
            self._log(
                f"Class {label}: Attempting HDBSCAN with min_cluster_size={min_cluster_size}"
            )
            hdb = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=max(1, min_cluster_size // 2),
                allow_single_cluster=True,
                cluster_selection_epsilon=0.0,
                copy=True,
            )
            clusters = hdb.fit_predict(X_class)
            unique_clusters = np.unique(clusters)

            # If HDBSCAN produces too many tiny clusters or only noise, fall back
            valid_clusters = unique_clusters[unique_clusters != -1]
            if len(valid_clusters) == 0 or (len(valid_clusters) > len(indices) / 5):
                logger.warning(
                    f"Class {label}: HDBSCAN produced {len(valid_clusters)} clusters "
                    f"({np.sum(clusters == -1)} noise points). Using single node instead."
                )

                node_id = self._next_id()
                node_info = {
                    "node": ObliqueNode(label=label, node_id=node_id),
                    "centroid": np.mean(X_class, axis=0),
                    "data": set(indices),
                    "id": node_id,
                    "active": True,
                }
                active_nodes[node_id] = node_info
                continue
            self._log(
                f"Class {label}: Found {len(valid_clusters)} valid clusters using HDBSCAN "
                f"({np.sum(clusters == -1)} noise points assigned to nearest cluster)"
            )

            # Handle noise points by assigning to nearest cluster
            noise_mask = clusters == -1

            if np.any(noise_mask):
                noise_indices = np.where(noise_mask)[0]
                cluster_centroids = []
                for c in valid_clusters:
                    cluster_centroids.append(np.mean(X_class[clusters == c], axis=0))
                cluster_centroids = np.array(cluster_centroids)
                for noise_idx in noise_indices:
                    dists = np.linalg.norm(
                        cluster_centroids - X_class[noise_idx], axis=1
                    )
                    clusters[noise_idx] = valid_clusters[np.argmin(dists)]

            for cluster_label in unique_clusters:
                if cluster_label == -1:  # Skip noise if any remain
                    continue
                cluster_mask = clusters == cluster_label
                cluster_indices = indices[cluster_mask]

                if len(cluster_indices) == 0:
                    continue

                node_id = self._next_id()
                node_info = {
                    "node": ObliqueNode(label=label, node_id=node_id),
                    "centroid": np.mean(X[cluster_indices], axis=0),
                    "data": set(cluster_indices),
                    "id": node_id,
                    "active": True,
                }
                active_nodes[node_id] = node_info

        # Build initial heap
        heap = []
        node_ids = list(active_nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id_a, id_b = node_ids[i], node_ids[j]
                if active_nodes[id_a]["node"].label == active_nodes[id_b]["node"].label:
                    continue
                dist = self._calc_distance(active_nodes[id_a], active_nodes[id_b])
                heapq.heappush(heap, (dist, id_a, id_b))

        self._log(f"Initialized heap with {len(heap)} potential edges.")
        return active_nodes, heap

    def fit(self, X, y):
        self._log(f"Starting mODT training on {len(y)} samples.")

        active_nodes, heap = self.get_label_clusters(X, y)
        total_samples = len(X)

        # Agglomerative Loop
        while heap:
            dist, id_a, id_b = heapq.heappop(heap)
            if id_a not in active_nodes or id_b not in active_nodes:
                continue

            node_a = active_nodes[id_a]
            node_b = active_nodes[id_b]

            # Skip if same class
            if (
                (node_a["node"].label is not None)
                and (node_b["node"].label is not None)
                and (node_a["node"].label == node_b["node"].label)
            ):
                continue

            # CRITICAL FIX: Use union instead of concatenation
            merged_data = node_a["data"] | node_b["data"]  # Set union

            # Skip if another node already covers this merge
            skip_merge = False

            for other_id, other_node in active_nodes.items():
                if other_id in (id_a, id_b):
                    continue

                if merged_data.issubset(other_node["data"]):
                    skip_merge = True
                    break

            if skip_merge:
                continue

            # Calculate training dataset

            inter_data = node_a["data"] & node_b["data"]  # Set intersection

            if len(inter_data) > 0:
                a_excl = node_a["data"] - inter_data
                b_excl = node_b["data"] - inter_data

                if len(a_excl) == 0 or len(b_excl) == 0:
                    continue

                train_a = X[list(a_excl)]
                train_b = X[list(b_excl)]

            else:
                train_a = X[list(node_a["data"])]
                train_b = X[list(node_b["data"])]

            self._log(
                f"Merging Node {id_a} ({len(node_a['data'])}) and Node {id_b} ({len(node_b['data'])}): "
                f"Total {len(merged_data)} samples (Distance: {dist:.4f})"
            )

            self._log(f" - Intersection: {len(inter_data)} samples")

            # Train SVM
            train_x = np.concatenate([train_a, train_b])
            train_y = np.concatenate([[0] * len(train_a), [1] * len(train_b)])

            clf = LinearSVC(dual="auto", max_iter=2000)
            clf.fit(train_x, train_y)

            # Create Parent
            new_id = self._next_id()
            parent_node = ObliqueNode(
                left=node_a["node"], right=node_b["node"], svm=clf, node_id=new_id
            )

            # Get indices as list for centroid calculation
            merged_indices = list(merged_data)

            # New node metadata
            new_node_info = {
                "node": parent_node,
                "centroid": np.mean(X[merged_indices], axis=0),
                "data": merged_data,  # Keep as set
                "id": new_id,
            }

            active_nodes[new_id] = new_node_info

            # Remove merged nodes
            del active_nodes[id_a]
            del active_nodes[id_b]

            # Check if root is found

            if len(merged_data) == total_samples:
                self._log("All samples merged into root node.")
                self.root = parent_node
                break

            # Update heap
            for other_id, other_node in active_nodes.items():
                if other_id == new_id:
                    continue

                # Check if one is subset of the other
                if new_node_info["data"].issubset(other_node["data"]) or other_node[
                    "data"
                ].issubset(new_node_info["data"]):
                    continue

                new_dist = self._calc_distance(new_node_info, other_node)
                heapq.heappush(heap, (new_dist, new_id, other_id))

        if self.root is None:
            logger.warning("Did not reach a full merge. Using largest subtree as root.")

            # Find the node with the most data
            if active_nodes:
                largest_node = max(active_nodes.values(), key=lambda n: len(n["data"]))
                self.root = largest_node["node"]
                self._log(
                    f"Root set to node {largest_node['id']} with {len(largest_node['data'])} samples"
                )

    def _next_id(self):
        self._node_counter += 1

        return self._node_counter

    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree has not been fitted yet")

        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.label is not None:
            return node.label

        decision = node.svm.predict(x.reshape(1, -1))[0]

        if decision == 0:
            return self._predict_single(x, node.left)

        else:
            return self._predict_single(x, node.right)


def test_dataset(name="Dataset", cmp_dt=True, cmp_xgb=True, verbose=False):
    logger.info(f"--- Benchmarking on {name} ---")

    X, y = read_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = []

    # Proposed ODT (Adaptive - NEW!)
    proposed_tree = MergeObliqueTree(cluster_strategy="adaptive", verbose=verbose)
    start = time.time()
    proposed_tree.fit(X_train, y_train)
    train_time = time.time() - start
    logger.success("mODT (adaptive) construction complete.")
    proposed_preds = proposed_tree.predict(X_test)
    results.append(
        {
            "Model": "Proposed (Adaptive)",
            "Accuracy": accuracy_score(y_test, proposed_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # Proposed ODT (Vanilla/No Clustering)
    proposed_tree = MergeObliqueTree(cluster_strategy="none", verbose=verbose)
    start = time.time()
    proposed_tree.fit(X_train, y_train)
    train_time = time.time() - start
    logger.success("mODT (vanilla) construction complete.")
    proposed_preds = proposed_tree.predict(X_test)
    results.append(
        {
            "Model": "Proposed (Vanilla)",
            "Accuracy": accuracy_score(y_test, proposed_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # Traditional Decision Tree
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


if __name__ == "__main__":
    ds1 = ["iris", "wine", "digits", "breast_cancer"]
    ds2 = ["mushrooms", "heart", "poker", "sensorless"]
    for ds in ds2:
        test_dataset(name=ds, cmp_dt=True, cmp_xgb=False, verbose=False)
