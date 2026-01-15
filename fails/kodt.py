import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler



class HybridNode:
    def __init__(self, depth):
        self.depth = depth
        self.vector, self.bias = None, None  # Oblique
        self.feature, self.threshold = None, None  # Axis
        self.left, self.right = None, None
        self.label = None
        self.is_oblique = False
        self.is_leaf = False

class OptimizedObliqueTree:
    def __init__(self, max_depth=10, min_samples=2, min_gain=1e-4):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_gain = min_gain
        self.root = None
        self.n_oblique_splits = 0

    def _gini(self, y):
        if len(y) == 0: return 0
        probs = np.bincount(y) / len(y)
        return 1.0 - np.sum(probs**2)

    def _get_best_axis_split(self, X, y):
        m, n = X.shape
        best_gini = self._gini(y)
        best_feat, best_thresh = None, None
        
        for feat in range(n):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_y, right_y = y[X[:, feat] <= thresh], y[X[:, feat] > thresh]
                if len(left_y) < 1 or len(right_y) < 1: continue
                
                gini = (len(left_y)*self._gini(left_y) + len(right_y)*self._gini(right_y)) / m
                if gini < best_gini:
                    best_gini, best_feat, best_thresh = gini, feat, thresh
        return best_gini, best_feat, best_thresh

    def _get_oblique_split(self, X, y):
        m = len(X)
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42).fit_predict(X)
        svm = LinearSVC(dual=False, penalty='l1', C=0.5, max_iter=2000)
        svm.fit(X, kmeans)
        
        vector, bias = svm.coef_[0], svm.intercept_[0]
        decision = np.dot(X, vector) + bias
        left_y, right_y = y[decision <= 0], y[decision > 0]
        
        if len(left_y) < 1 or len(right_y) < 1: return 1.0, None, None
        
        gini = (len(left_y)*self._gini(left_y) + len(right_y)*self._gini(right_y)) / m
        return gini, vector, bias

    def fit(self, X, y, X_val=None, y_val=None):
        self.root = self._build_tree(X, y, 0)
        if X_val is not None:
            self.prune(self.root, X_val, y_val)
        print(f"Number of Oblique Splits Used: {self.n_oblique_splits}")

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        current_gini = self._gini(y)
        leaf_label = np.argmax(np.bincount(y)) if num_samples > 0 else None

        if depth >= self.max_depth or current_gini == 0 or num_samples < self.min_samples:
            node = HybridNode(depth)
            node.label, node.is_leaf = leaf_label, True
            return node

        # Auto-Detect: Compare Oblique vs Axis-Aligned
        gini_oblique, vec, bias = self._get_oblique_split(X, y)
        gini_axis, feat, thresh = self._get_best_axis_split(X, y)

        node = HybridNode(depth)
        node.label = leaf_label # Keep for pruning
        
        # Heuristic: Prefer Oblique only if it provides significantly better gain
        if gini_oblique < (gini_axis * 0.95):
            self.n_oblique_splits += 1
            node.is_oblique = True
            node.vector, node.bias = vec, bias
            mask = (np.dot(X, vec) + bias) <= 0
        else:
            if feat is None:
                node.is_leaf = True
                return node
            node.is_oblique = False
            node.feature, node.threshold = feat, thresh
            mask = X[:, feat] <= thresh

        node.left = self._build_tree(X[mask], y[mask], depth + 1)
        node.right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return node

    def prune(self, node, X_val, y_val):
        """Reduced Error Pruning: Post-order traversal."""
        if node.is_leaf or len(y_val) == 0:
            return

        # Split validation data to recurse
        if node.is_oblique:
            mask = (np.dot(X_val, node.vector) + node.bias) <= 0
        else:
            mask = X_val[:, node.feature] <= node.threshold

        self.prune(node.left, X_val[mask], y_val[mask])
        self.prune(node.right, X_val[~mask], y_val[~mask])

        # If children are leaves, evaluate if merging them is better
        if node.left.is_leaf and node.right.is_leaf:
            # Accuracy before pruning this node
            preds = np.where(mask, node.left.label, node.right.label)
            error_before = 1 - accuracy_score(y_val, preds)
            
            # Accuracy if we prune (turn this node into a leaf)
            error_after = 1 - accuracy_score(y_val, [node.label]*len(y_val))
            
            if error_after <= error_before:
                node.is_leaf = True
                node.left = node.right = None

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf: return node.label
        go_left = (np.dot(x, node.vector) + node.bias <= 0) if node.is_oblique else (x[node.feature] <= node.threshold)
        return self._traverse(x, node.left if go_left else node.right)


def main(dataset='iris', max_depth=5, min_samples=2):
    # 1. Load Data
    if dataset == 'iris':
        data = load_iris()
    elif dataset == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset == 'wine':
        data = load_wine()
    elif dataset == 'digits':
        data = load_digits()
    else:
        raise ValueError("Unsupported dataset. Choose from 'iris', 'breast_cancer', 'wine', 'digits'.")

    X, y = data.data, data.target
    
    # 2. Preprocessing
    # IMPORTANT: SVM and K-Means are distance-based. 
    # Always scale features for Oblique Trees!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3
    )

    # 3. Initialize and Train our Oblique Tree
    print("Training K-SVM Oblique Tree...")
    oblique_tree = OptimizedObliqueTree(max_depth=max_depth, min_samples=min_samples)
    oblique_tree.fit(X_train, y_train)
    
    # 4. Compare with Standard Axis-Aligned Decision Tree
    standard_tree = DecisionTreeClassifier(max_depth=max_depth)
    standard_tree.fit(X_train, y_train)

    # 5. Evaluation
    y_pred_oblique = oblique_tree.predict(X_test)
    y_pred_standard = standard_tree.predict(X_test)

    acc_oblique = accuracy_score(y_test, y_pred_oblique)
    acc_standard = accuracy_score(y_test, y_pred_standard)

    print("-" * 30)
    print(f"Oblique Tree Accuracy:  {acc_oblique:.4f}")
    print(f"Standard Tree Accuracy: {acc_standard:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main('iris', max_depth=6, min_samples=2)
    main('wine', max_depth=8, min_samples=2)
    main('breast_cancer', max_depth=10, min_samples=2)
    main('digits', max_depth=16, min_samples=2)
