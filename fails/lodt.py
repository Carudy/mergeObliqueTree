import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import sklearn.datasets as skd

import warnings
warnings.filterwarnings('ignore')


class AdaptiveObliqueDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Oblique Decision Tree that compares multiple split methods at each node
    and selects the best one based on information gain.
    
    Methods compared:
    1. LDA (Linear Discriminant Analysis)
    2. Clustering + SVM (HDBSCAN/KMeans + Linear SVM)
    3. Axis-aligned CART
    
    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_info_gain : float, default=0.01
        Minimum information gain required for a split
    """
    
    def __init__(self, max_depth=10, min_info_gain=0.01):
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        
        # Statistics tracking
        self.split_method_counts_ = {
            'lda': 0,
            'clustering_svm': 0,
            'axis_aligned': 0,
            'leaf': 0
        }
        
    def fit(self, X, y):
        """
        Build the adaptive oblique decision tree
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Reset statistics
        self.split_method_counts_ = {
            'lda': 0,
            'clustering_svm': 0,
            'axis_aligned': 0,
            'leaf': 0
        }
        
        # Build tree
        self.tree_ = self._build_tree(X, y, depth=0)
        
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_classes = len(np.unique(y))
        
        # Create leaf node if termination conditions met
        if (depth >= self.max_depth or n_classes == 1):
            self.split_method_counts_['leaf'] += 1
            return self._create_leaf(y)
        
        # Try all split methods and compare
        best_split = None
        best_gain = -np.inf
        best_method = None
        
        # Method 1: LDA
        lda_split = self._try_lda_split(X, y)
        if lda_split is not None and lda_split['gain'] > best_gain:
            best_gain = lda_split['gain']
            best_split = lda_split
            best_method = 'lda'
        
        # Method 2: Clustering + SVM
        cluster_split = self._try_clustering_split(X, y)
        if cluster_split is not None and cluster_split['gain'] > best_gain:
            best_gain = cluster_split['gain']
            best_split = cluster_split
            best_method = 'clustering_svm'
        
        # Method 3: Axis-aligned CART
        axis_split = self._try_axis_split(X, y)
        if axis_split is not None and axis_split['gain'] > best_gain:
            best_gain = axis_split['gain']
            best_split = axis_split
            best_method = 'axis_aligned'
        
        # If no good split found, create leaf
        if best_split is None or best_gain < self.min_info_gain:
            self.split_method_counts_['leaf'] += 1
            return self._create_leaf(y)
        
        # Record which method was used
        self.split_method_counts_[best_method] += 1
        
        # Apply split
        left_mask, right_mask = self._apply_split(X, best_split)
  
        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'type': 'split',
            'method': best_method,
            'split': best_split,
            'left': left_child,
            'right': right_child
        }
    
    def _try_lda_split(self, X, y):
        """Try Linear Discriminant Analysis split"""
        try:
            # Use regularized LDA for stability
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X, y)
            
            # For binary classification
            if self.n_classes_ == 2:
                w = lda.coef_[0]
                b = lda.intercept_[0]
            else:
                # Multi-class: use first discriminant direction
                X_proj = lda.transform(X)[:, 0]
                threshold = np.median(X_proj)
                
                # Get direction vector
                w = lda.scalings_[:, 0]
                b = -threshold * np.linalg.norm(w)
            
            # Calculate predictions and information gain
            projections = X @ w + b
            left_mask = projections <= 0
            
            gain = self._information_gain(y, left_mask)
            
            return {
                'method': 'lda',
                'w': w,
                'b': b,
                'gain': gain
            }
            
        except Exception as e:
            return None
    
    def _try_clustering_split(self, X, y):
        """Try Clustering + SVM split"""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Try HDBSCAN first
            min_cluster_size = max(5, len(X) // 20)
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                               min_samples=3,
                               allow_single_cluster=False)
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # If we get exactly 2 clusters, great!
            if n_clusters == 2:
                # Map clusters to binary labels for SVM
                cluster_binary = (cluster_labels == np.unique(cluster_labels[cluster_labels >= 0])[0]).astype(int)
                cluster_binary = cluster_binary[cluster_labels >= 0]
                X_clustered = X_scaled[cluster_labels >= 0]
                
            # If we get multiple clusters, use K-means to group into 2
            elif n_clusters > 2:
                valid_mask = cluster_labels >= 0
                X_valid = X_scaled[valid_mask]
                
                if len(X_valid) < 10:
                    return None
                
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                cluster_binary = kmeans.fit_predict(X_valid)
                X_clustered = X_valid
                
            else:
                # Clustering failed, fall back to K-means directly
                if len(X) < 10:
                    return None
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                cluster_binary = kmeans.fit_predict(X_scaled)
                X_clustered = X_scaled
            
            # Train linear SVM on clusters
            if len(np.unique(cluster_binary)) < 2:
                return None
                
            svm = LinearSVC(C=1.0, max_iter=1000, dual='auto')
            svm.fit(X_clustered, cluster_binary)
            
            # Get hyperplane parameters (in scaled space)
            w_scaled = svm.coef_[0]
            b_scaled = svm.intercept_[0]
            
            # Transform back to original space
            w = w_scaled / scaler.scale_
            b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)
            
            # Calculate predictions and information gain
            projections = X @ w + b
            left_mask = projections <= 0
            
            gain = self._information_gain(y, left_mask)
            
            return {
                'method': 'clustering_svm',
                'w': w,
                'b': b,
                'gain': gain,
                'scaler': scaler
            }
            
        except Exception as e:
            return None
    
    def _try_axis_split(self, X, y):
        """Try axis-aligned CART split"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try different thresholds
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                
                gain = self._information_gain(y, left_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return None
        
        return {
            'method': 'axis_aligned',
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain
        }
    
    def _information_gain(self, y, left_mask):
        """Calculate information gain using Gini impurity"""
        def gini_impurity(labels):
            if len(labels) == 0:
                return 0
            counts = np.bincount(labels)
            probs = counts / len(labels)
            return 1 - np.sum(probs ** 2)
        
        # Parent impurity
        parent_impurity = gini_impurity(y)
        
        # Children impurities
        left_y = y[left_mask]
        right_y = y[~left_mask]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        n = len(y)
        left_impurity = gini_impurity(left_y)
        right_impurity = gini_impurity(right_y)
        
        # Weighted average of children impurities
        child_impurity = (len(left_y) / n) * left_impurity + \
                        (len(right_y) / n) * right_impurity
        
        # Information gain
        return parent_impurity - child_impurity
    
    def _apply_split(self, X, split):
        """Apply a split to data"""
        if split['method'] in ['lda', 'clustering_svm']:
            projections = X @ split['w'] + split['b']
            left_mask = projections <= 0
        else:  # axis_aligned
            left_mask = X[:, split['feature']] <= split['threshold']
        
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _create_leaf(self, y):
        """Create a leaf node"""
        counts = np.bincount(y, minlength=self.n_classes_)
        return {
            'type': 'leaf',
            'class': np.argmax(counts),
            'proba': counts / len(y)
        }
    
    def predict(self, X):
        """Predict class labels"""
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = np.array([self._predict_sample(x, self.tree_) for x in X])
        return predictions
    
    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node['type'] == 'leaf':
            return node['class']
        
        # Apply split
        split = node['split']
        if split['method'] in ['lda', 'clustering_svm']:
            go_left = (x @ split['w'] + split['b']) <= 0
        else:  # axis_aligned
            go_left = x[split['feature']] <= split['threshold']
        
        if go_left:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = check_array(X)
        
        probas = np.array([self._predict_proba_sample(x, self.tree_) for x in X])
        return probas
    
    def _predict_proba_sample(self, x, node):
        """Predict probabilities for single sample"""
        if node['type'] == 'leaf':
            return node['proba']
        
        # Apply split
        split = node['split']
        if split['method'] in ['lda', 'clustering_svm']:
            go_left = (x @ split['w'] + split['b']) <= 0
        else:  # axis_aligned
            go_left = x[split['feature']] <= split['threshold']
        
        if go_left:
            return self._predict_proba_sample(x, node['left'])
        else:
            return self._predict_proba_sample(x, node['right'])
    
    def get_method_statistics(self):
        """
        Get statistics about which split methods were used
        
        Returns
        -------
        dict : Dictionary with counts and percentages for each method
        """
        total_splits = sum([v for k, v in self.split_method_counts_.items() 
                           if k != 'leaf'])
        total_nodes = sum(self.split_method_counts_.values())
        
        stats = {
            'counts': self.split_method_counts_.copy(),
            'percentages': {},
            'total_splits': total_splits,
            'total_nodes': total_nodes
        }
        
        for method, count in self.split_method_counts_.items():
            if method == 'leaf':
                stats['percentages'][method] = (count / total_nodes * 100) if total_nodes > 0 else 0
            else:
                stats['percentages'][method] = (count / total_splits * 100) if total_splits > 0 else 0
        
        return stats
    
    def print_method_statistics(self):
        """Print detailed statistics about method usage"""
        stats = self.get_method_statistics()
        
        print("=" * 60)
        print("Split Method Statistics")
        print("=" * 60)
        print(f"\nTotal nodes in tree: {stats['total_nodes']}")
        print(f"Total split nodes: {stats['total_splits']}")
        print(f"Total leaf nodes: {stats['counts']['leaf']}")
        print("\n" + "-" * 60)
        print("Split Method Distribution:")
        print("-" * 60)
        
        for method in ['lda', 'clustering_svm', 'axis_aligned']:
            count = stats['counts'][method]
            pct = stats['percentages'][method]
            print(f"{method:20s}: {count:4d} splits ({pct:5.1f}%)")
        
        print("-" * 60)
        print(f"{'Leaf nodes':20s}: {stats['counts']['leaf']:4d} nodes "
              f"({stats['percentages']['leaf']:5.1f}%)")
        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("Testing Adaptive Oblique Decision Tree\n")
    
    # Test on different datasets
    datasets = {
        'iris': skd.load_iris(return_X_y=True),
        'wine': skd.load_wine(return_X_y=True),
        'breast_cancer': skd.load_breast_cancer(return_X_y=True),
        'minist': skd.load_digits(return_X_y=True),
    }
    
    for name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train tree
        tree = AdaptiveObliqueDecisionTree(
            max_depth=10,
            min_info_gain=0.01
        )
        
        tree.fit(X_train, y_train)
        
        # Predictions
        y_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\n" + classification_report(y_test, y_pred))
        
        # Print method statistics
        tree.print_method_statistics()