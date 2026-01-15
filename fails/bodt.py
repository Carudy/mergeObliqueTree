import heapq

import numpy as np
from sklearn.svm import LinearSVC
import time

import pandas as pd
import sklearn.datasets as skd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logger



class ObliqueNode:
    def __init__(self, left=None, right=None, svm=None, label=None, node_id=None):
        self.left = left
        self.right = right
        self.svm = svm
        self.label = label
        self.node_id = node_id


class BottomUpObliqueTree:
    def __init__(self):
        self.root = None
        self._node_counter = 0

    def fit(self, X, y):
        unique_labels = np.unique(y)
        logger.info(f"Starting training with {len(unique_labels)} unique classes.")

        # 1. Initialize nodes for each class
        active_nodes = {}
        heap = []

        for label in unique_labels:
            indices = y == label
            node_id = self._next_id()
            node_info = {
                "node": ObliqueNode(label=label, node_id=node_id),
                "centroid": np.mean(X[indices], axis=0),
                "data": X[indices],
                "id": node_id,
                "active": True,
            }
            active_nodes[node_id] = node_info
            logger.debug(f"Created leaf node {node_id} for class {label}.")

        # 2. Build initial heap (all-pairs distance)
        node_ids = list(active_nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                id_a, id_b = node_ids[i], node_ids[j]
                dist = np.linalg.norm(
                    active_nodes[id_a]["centroid"] - active_nodes[id_b]["centroid"]
                )
                heapq.heappush(heap, (dist, id_a, id_b))

        logger.info(f"Initialized heap with {len(heap)} potential edges.")

        # 3. Agglomerative Loop
        while len(active_nodes) > 1:
            # Pop the closest pair from the heap
            if not heap:
                break

            dist, id_a, id_b = heapq.heappop(heap)

            # Skip if one of the nodes was already merged (Lazy Deletion)
            if id_a not in active_nodes or id_b not in active_nodes:
                continue

            logger.info(f"Merging Node {id_a} and Node {id_b} (Distance: {dist:.4f})")

            node_a = active_nodes.pop(id_a)
            node_b = active_nodes.pop(id_b)

            # Train SVM for this split
            merged_X = np.vstack([node_a["data"], node_b["data"]])
            merged_y = np.array([0] * len(node_a["data"]) + [1] * len(node_b["data"]))

            clf = LinearSVC(dual=False, max_iter=10000, tol=1e-3)
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
                new_dist = np.linalg.norm(
                    new_node_info["centroid"] - other_node["centroid"]
                )
                heapq.heappush(heap, (new_dist, new_id, other_id))

            active_nodes[new_id] = new_node_info
            logger.success(
                f"New Parent Node {new_id} created. {len(active_nodes)} clusters remaining."
            )

        self.root = list(active_nodes.values())[0]["node"]
        logger.info("Tree construction complete.")

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


def test_dataset(dataset_loader, name="Dataset"):
    logger.info(f"--- Benchmarking on {name} ---")
    data = dataset_loader()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results = []

    # 1. Proposed Bottom-Up Oblique Tree
    proposed_tree = BottomUpObliqueTree()
    start = time.time()
    proposed_tree.fit(X_train, y_train)
    train_time = time.time() - start
    proposed_preds = proposed_tree.predict(X_test)
    results.append(
        {
            "Model": "Proposed Oblique (Bottom-Up)",
            "Accuracy": accuracy_score(y_test, proposed_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # 2. Traditional Decision Tree (Scikit-Learn)
    dt = DecisionTreeClassifier()
    start = time.time()
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    dt_preds = dt.predict(X_test)
    results.append(
        {
            "Model": "Standard Decision Tree",
            "Accuracy": accuracy_score(y_test, dt_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # 3. XGBoost
    xgb = XGBClassifier(eval_metric="mlogloss")
    start = time.time()
    xgb.fit(X_train, y_train)
    train_time = time.time() - start
    xgb_preds = xgb.predict(X_test)
    results.append(
        {
            "Model": "XGBoost (Ensemble)",
            "Accuracy": accuracy_score(y_test, xgb_preds),
            "Train Time (s)": round(train_time, 4),
        }
    )

    # Display results
    df = pd.DataFrame(results)
    print(f"\nResults for {name}:")
    print(df.to_string(index=False))
    return df


# --- Test Run ---
if __name__ == "__main__":
    test_dataset(skd.load_wine, "Wine Recognition")
    test_dataset(skd.load_digits, "MNIST Digits (8x8)")
    test_dataset(skd.load_iris, "Iris Dataset")
    test_dataset(skd.load_breast_cancer, "Breast Cancer Dataset")
