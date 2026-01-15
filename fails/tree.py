import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import heapq
from itertools import combinations
from dataclasses import dataclass
from typing import Set, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SVMNode:
    """Node in the oblique decision tree"""
    pos_labels: Set[int]  # Labels on positive side of hyperplane
    neg_labels: Set[int]  # Labels on negative side of hyperplane
    svm: SVC  # Trained SVM classifier
    left_child: Optional['SVMNode'] = None  # Positive side
    right_child: Optional['SVMNode'] = None  # Negative side
    node_id: int = 0
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (single label)"""
        return len(self.pos_labels) == 1 or len(self.neg_labels) == 1
    
    def covers_all_labels(self, all_labels: Set[int]) -> bool:
        """Check if node splits all labels"""
        return self.pos_labels.union(self.neg_labels) == all_labels
    
    def __hash__(self):
        return self.node_id
    
    def __eq__(self, other):
        return self.node_id == other.node_id


class ObliqueDecisionTree:
    """Oblique Decision Tree built via hierarchical SVM merging"""
    
    def __init__(self, C=1.0, kernel='linear', distance_weights=(0.7, 0.2, 0.1)):
        """
        Args:
            C: SVM regularization parameter
            kernel: SVM kernel type
            distance_weights: (angle_weight, offset_weight, margin_weight)
        """
        self.C = C
        self.kernel = kernel
        self.distance_weights = distance_weights
        self.root = None
        self.node_counter = 0
        
    def fit(self, X, y):
        """Build oblique decision tree"""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        print(f"Training oblique decision tree for {self.n_classes_} classes...")
        
        # Phase 1: Train all pairwise SVMs
        print("Phase 1: Training pairwise SVMs...")
        initial_nodes = self._train_pairwise_svms(X, y)
        print(f"Created {len(initial_nodes)} initial nodes")
        
        # Phase 2: Hierarchical merging
        print("Phase 2: Hierarchical merging...")
        self.root = self._hierarchical_merge(initial_nodes, X, y)
        print("Tree construction complete!")
        
        return self
    
    def _train_pairwise_svms(self, X, y) -> List[SVMNode]:
        """Train SVM for each pair of classes"""
        nodes = []
        
        for label_i, label_j in combinations(self.classes_, 2):
            # Get samples for this pair
            mask = (y == label_i) | (y == label_j)
            X_pair = X[mask]
            y_pair = y[mask]
            
            # Convert to binary: label_i=+1, label_j=-1
            y_binary = np.where(y_pair == label_i, 1, -1)
            
            # Train SVM
            svm = SVC(C=self.C, kernel=self.kernel, probability=True)
            svm.fit(X_pair, y_binary)
            
            # Create node
            node = SVMNode(
                pos_labels={label_i},
                neg_labels={label_j},
                svm=svm,
                node_id=self.node_counter
            )
            self.node_counter += 1
            nodes.append(node)
        
        return nodes
    
    def _compute_distance(self, node1: SVMNode, node2: SVMNode) -> float:
        """
        Compute geometric distance between two SVM hyperplanes
        Uses weighted combination of:
        1. Angular distance (direction similarity)
        2. Offset distance (parallel shift)
        3. Margin quality (confidence similarity)
        """
        if self.kernel != 'linear':
            # For non-linear kernels, use support vector overlap
            return self._compute_nonlinear_distance(node1, node2)
        
        # Extract hyperplane parameters for linear SVMs
        w1 = node1.svm.coef_[0]
        b1 = node1.svm.intercept_[0]
        w2 = node2.svm.coef_[0]
        b2 = node2.svm.intercept_[0]
        
        # 1. Angular distance (0=same direction, 1=perpendicular, 2=opposite)
        cos_sim = np.abs(np.dot(w1, w2)) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        angular_dist = 1 - cos_sim
        
        # 2. Offset distance (normalized)
        # Distance between parallel hyperplanes
        offset_dist = np.abs(b1 / np.linalg.norm(w1) - b2 / np.linalg.norm(w2))
        offset_dist = min(offset_dist / 10.0, 1.0)  # Normalize
        
        # 3. Margin quality difference
        margin1 = 1.0 / np.linalg.norm(w1)
        margin2 = 1.0 / np.linalg.norm(w2)
        margin_dist = np.abs(margin1 - margin2) / max(margin1, margin2)
        
        # Weighted combination
        w_angle, w_offset, w_margin = self.distance_weights
        distance = w_angle * angular_dist + w_offset * offset_dist + w_margin * margin_dist
        
        return distance
    
    def _compute_nonlinear_distance(self, node1: SVMNode, node2: SVMNode) -> float:
        """Fallback distance for non-linear kernels"""
        # Use support vector overlap as proxy
        sv1 = set(tuple(sv) for sv in node1.svm.support_vectors_)
        sv2 = set(tuple(sv) for sv in node2.svm.support_vectors_)
        
        if len(sv1) == 0 or len(sv2) == 0:
            return 1.0
        
        overlap = len(sv1 & sv2)
        union = len(sv1 | sv2)
        
        return 1.0 - (overlap / union)
    
    def _find_merge_configuration(self, node1: SVMNode, node2: SVMNode) -> Optional[Tuple[Set, Set]]:
        """
        Determine which sides to merge based on label overlap
        Returns: (new_pos_labels, new_neg_labels) or None if no valid merge
        """
        # Check if negative sides have overlap → merge positive sides
        neg_overlap = node1.neg_labels & node2.neg_labels
        if neg_overlap:
            new_pos = node1.pos_labels | node2.pos_labels
            new_neg = neg_overlap
            return (new_pos, new_neg)
        
        # Check if positive sides have overlap → merge negative sides
        pos_overlap = node1.pos_labels & node2.pos_labels
        if pos_overlap:
            new_pos = pos_overlap
            new_neg = node1.neg_labels | node2.neg_labels
            return (new_pos, new_neg)
        
        # No common side, cannot merge
        return None
    
    def _train_merged_svm(self, X, y, pos_labels: Set[int], neg_labels: Set[int]) -> SVC:
        """Train SVM on merged label sets"""
        all_labels = pos_labels | neg_labels  # Set union
        mask = np.isin(y, list(all_labels))
        X_subset = X[mask]
        y_subset = y[mask]
        
        # Convert to binary: pos_labels=+1, neg_labels=-1
        print(f"Training merged SVM: {pos_labels} vs {neg_labels} on {len(X_subset)} samples")
        y_binary = np.where(np.isin(y_subset, list(pos_labels)), 1, -1)
        
        # Train SVM with class weighting for balance
        svm = SVC(C=self.C, kernel=self.kernel, probability=True, class_weight='balanced')
        svm.fit(X_subset, y_binary)
        
        return svm
    
    def _hierarchical_merge(self, nodes: List[SVMNode], X, y) -> SVMNode:
        """Perform hierarchical merging to build tree"""
        all_labels = set(self.classes_)
        active_nodes = set(nodes)
        
        # Initialize heap with all pairwise distances
        heap = []
        distances = {}
        
        for node1, node2 in combinations(nodes, 2):
            dist = self._compute_distance(node1, node2)
            distances[(node1.node_id, node2.node_id)] = dist
            heapq.heappush(heap, (dist, node1.node_id, node2.node_id, node1, node2))
        
        iteration = 0
        while len(active_nodes) > 1:
            iteration += 1
            
            # Find next best merge
            merged = False
            while heap and not merged:
                dist, id1, id2, node1, node2 = heapq.heappop(heap)
                
                # Check if nodes are still active
                if node1 not in active_nodes or node2 not in active_nodes:
                    continue
                
                # Check if merge is valid
                merge_config = self._find_merge_configuration(node1, node2)
                if merge_config is None:
                    continue
                
                new_pos, new_neg = merge_config
                
                # Train new SVM
                new_svm = self._train_merged_svm(X, y, new_pos, new_neg)
                
                # Create parent node
                parent = SVMNode(
                    pos_labels=new_pos,
                    neg_labels=new_neg,
                    svm=new_svm,
                    left_child=node1 if node1.pos_labels.issubset(new_pos) else node2,
                    right_child=node2 if node2.neg_labels.issubset(new_neg) else node1,
                    node_id=self.node_counter
                )
                self.node_counter += 1
                
                print(f"Iter {iteration}: Merged {new_pos} vs {new_neg} (dist={dist:.4f})")
                
                # Update active nodes
                active_nodes.remove(node1)
                active_nodes.remove(node2)
                active_nodes.add(parent)
                
                # Check if we have root
                if parent.covers_all_labels(all_labels):
                    return parent
                
                # Update heap with new distances
                for other_node in active_nodes:
                    if other_node != parent:
                        new_dist = self._compute_distance(parent, other_node)
                        heapq.heappush(heap, (new_dist, parent.node_id, other_node.node_id, parent, other_node))
                
                merged = True
            
            if not merged:
                # Cannot merge further, return arbitrary root
                print("Warning: Could not complete full merge, returning partial tree")
                return list(active_nodes)[0]
        
        return list(active_nodes)[0]
    
    def predict(self, X):
        """Predict class labels"""
        return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x):
        """Predict single sample by traversing tree"""
        node = self.root
        
        while not node.is_leaf():
            prediction = node.svm.predict([x])[0]
            
            if prediction == 1:  # Positive side
                if node.left_child is not None:
                    node = node.left_child
                else:
                    break
            else:  # Negative side
                if node.right_child is not None:
                    node = node.right_child
                else:
                    break
        
        # Return the single label from leaf
        if len(node.pos_labels) == 1:
            return list(node.pos_labels)[0]
        elif len(node.neg_labels) == 1:
            return list(node.neg_labels)[0]
        else:
            # Fallback: return most frequent label
            return list(node.pos_labels | node.neg_labels)[0]



if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits, load_wine
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 70)
    print("Oblique Decision Tree via Hierarchical SVM Merging")
    print("=" * 70)
    
    # Test on multiple real datasets
    datasets = {
        'Iris': load_iris(),
        # 'Wine': load_wine(),
        # 'Digits (subset)': load_digits()
    }
    
    for dataset_name, dataset in datasets.items():
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")
        
        X, y = dataset.data, dataset.target
        
        # For digits, use subset to speed up
        if dataset_name == 'Digits (subset)':
            # Use only first 5 digits (0-4) and subsample
            mask = y < 5
            X, y = X[mask], y[mask]
            np.random.seed(42)
            indices = np.random.choice(len(X), size=min(400, len(X)), replace=False)
            X, y = X[indices], y[indices]
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Samples: {len(X_train)} train, {len(X_test)} test")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {len(np.unique(y))} ({list(np.unique(y))})")
        
        # Train oblique decision tree
        print("\nTraining Oblique Decision Tree...")
        odt = ObliqueDecisionTree(C=1.0, kernel='linear')
        odt.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = odt.predict(X_train)
        y_pred_test = odt.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n{'-' * 70}")
        print(f"Oblique Decision Tree Results:")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:     {test_acc:.4f}")
        
        # Compare with standard SVM
        from sklearn.svm import SVC as StandardSVC
        svm_baseline = StandardSVC(C=1.0, kernel='linear')
        svm_baseline.fit(X_train, y_train)
        svm_pred_train = svm_baseline.predict(X_train)
        svm_pred_test = svm_baseline.predict(X_test)
        
        print(f"\nStandard Linear SVM (one-vs-rest):")
        print(f"  Training accuracy: {accuracy_score(y_train, svm_pred_train):.4f}")
        print(f"  Test accuracy:     {accuracy_score(y_test, svm_pred_test):.4f}")
        
        # Compare with decision tree
        from sklearn.tree import DecisionTreeClassifier
        dt_baseline = DecisionTreeClassifier(random_state=42)
        dt_baseline.fit(X_train, y_train)
        dt_pred_test = dt_baseline.predict(X_test)
        
        print(f"\nAxis-aligned Decision Tree:")
        print(f"  Test accuracy:     {accuracy_score(y_test, dt_pred_test):.4f}")
        print(f"{'-' * 70}")
    
    print(f"\n{'=' * 70}")
    print("All experiments complete!")
    print(f"{'=' * 70}")