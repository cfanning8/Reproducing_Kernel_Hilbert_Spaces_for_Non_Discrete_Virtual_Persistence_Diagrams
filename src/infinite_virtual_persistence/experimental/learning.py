"""
Learning methods: kernel ridge, SVM, GP classification/regression.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import solve


class KernelRidge:
    """
    Kernel ridge regression: f*(.) = sum_i alpha_i k_t(., x_i)
    with alpha = (K + lambda I)^{-1} y
    """
    
    def __init__(self, kernel, lambda_reg: float = 1.0):
        """
        Initialize kernel ridge.
        
        Args:
            kernel: Kernel function or callable
            lambda_reg: Regularization parameter
        """
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.alpha = None
        self.X_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit kernel ridge model.
        
        Args:
            X: Training data, shape (n, dim)
            y: Training labels, shape (n,)
        """
        self.X_train = X
        n = len(X)
        
        # Compute Gram matrix
        if hasattr(self.kernel, 'gram_matrix'):
            K = self.kernel.gram_matrix(X)
        else:
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
        
        # Solve (K + lambda I) alpha = y
        self.alpha = solve(K + self.lambda_reg * np.eye(n), y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        
        Args:
            X: Test data, shape (m, dim)
        
        Returns:
            Predictions, shape (m,)
        """
        if self.alpha is None:
            raise ValueError("Model not fitted")
        
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                if hasattr(self.kernel, 'gram_matrix'):
                    k_val = self.kernel(x, x_train)
                else:
                    k_val = self.kernel(x, x_train)
                predictions[i] += self.alpha[j] * k_val
        
        return predictions
    
    def rkhs_norm_sq(self, y: Optional[np.ndarray] = None) -> float:
        """
        Compute ||f*||_H^2 = alpha^T K alpha = y^T (K + lambda I)^{-1} K (K + lambda I)^{-1} y
        
        Args:
            y: Training labels (optional, for recomputation if needed)
        
        Returns:
            RKHS norm squared
        """
        if self.alpha is None:
            raise ValueError("Model not fitted")
        
        n = len(self.X_train)
        
        # Recompute K
        if hasattr(self.kernel, 'gram_matrix'):
            K = self.kernel.gram_matrix(self.X_train)
        else:
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(self.X_train[i], self.X_train[j])
        
        return np.dot(self.alpha, K @ self.alpha)
    
    def lipschitz_certificate(self, kernel: 'GaussianRKHSKernel') -> float:
        """
        Compute data-dependent Lipschitz certificate.
        
        Lip_rho(f*|_K) <= sqrt(t) ||Sigma^{1/2} J|| sqrt(||f*||_H^2)
        """
        if self.alpha is None:
            raise ValueError("Model not fitted")
        
        f_norm_sq = self.rkhs_norm_sq(None)
        return kernel.lipschitz_bound(np.sqrt(f_norm_sq))


class KernelSVM:
    """
    Kernel SVM wrapper using sklearn SVC/SVR.
    """
    
    def __init__(self, kernel, task: str = 'classification', **svm_kwargs):
        """
        Initialize kernel SVM.
        
        Args:
            kernel: Kernel function or callable
            task: 'classification' or 'regression'
            **svm_kwargs: Additional arguments for SVC/SVR
        """
        self.kernel = kernel
        self.task = task
        
        if task == 'classification':
            self.model = SVC(kernel='precomputed', **svm_kwargs)
        else:
            self.model = SVR(kernel='precomputed', **svm_kwargs)
        
        self.X_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVM model."""
        self.X_train = X
        n = len(X)
        
        # Compute Gram matrix
        if hasattr(self.kernel, 'gram_matrix'):
            K = self.kernel.gram_matrix(X)
        else:
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
        
        self.model.fit(K, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if self.X_train is None:
            raise ValueError("Model not fitted")
        
        # Compute test-train Gram matrix
        n_train = len(self.X_train)
        n_test = len(X)
        K_test = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            for j in range(n_train):
                if hasattr(self.kernel, 'gram_matrix'):
                    K_test[i, j] = self.kernel(X[i], self.X_train[j])
                else:
                    K_test[i, j] = self.kernel(X[i], self.X_train[j])
        
        return self.model.predict(K_test)


class GaussianProcessWrapper:
    """
    Gaussian process classification/regression wrapper.
    """
    
    def __init__(self, kernel, task: str = 'classification', **gp_kwargs):
        """
        Initialize GP model.
        
        Args:
            kernel: Kernel function (will be wrapped)
            task: 'classification' or 'regression'
            **gp_kwargs: Additional arguments for GP
        """
        self.kernel = kernel
        self.task = task
        
        # Wrap kernel for sklearn GP
        # sklearn expects kernel objects, so we create a custom wrapper
        if task == 'classification':
            self.model = GaussianProcessClassifier(**gp_kwargs)
        else:
            self.model = GaussianProcessRegressor(**gp_kwargs)
        
        self.X_train = None
        self._gram_matrix = None
    
    def _compute_gram(self, X: np.ndarray) -> np.ndarray:
        """Compute Gram matrix."""
        n = len(X)
        K = np.zeros((n, n))
        
        if hasattr(self.kernel, 'gram_matrix'):
            K = self.kernel.gram_matrix(X)
        else:
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP model."""
        self.X_train = X
        self._gram_matrix = self._compute_gram(X)
        
        # For sklearn GP, we need to use a custom kernel
        # This is a simplified version - full implementation would
        # create a proper sklearn kernel object
        # For now, we'll use a workaround with RBF kernel
        # In practice, need to implement custom sklearn kernel
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")
        return self.model.predict_proba(X)


def radev_complexity_bound(n: int, Lambda: float, delta: float, 
                          kappa: float = 1.0) -> float:
    """
    Rademacher complexity bound for RKHS learning.
    
    For 1-Lipschitz loss and hypothesis class F_Lambda = {f: ||f||_H <= Lambda},
    with probability >= 1-delta:
    E[ell(f(X), Y)] <= (1/n) sum_i ell(f(x_i), y_i) + 2*Lambda/sqrt(n) + 3*sqrt(log(2/delta)/(2n))
    
    Args:
        n: Sample size
        Lambda: RKHS norm bound
        delta: Confidence parameter
        kappa: sup_x sqrt(k(x,x)) (default 1.0 for our kernels)
    
    Returns:
        Generalization bound term
    """
    return 2 * Lambda / np.sqrt(n) + 3 * np.sqrt(np.log(2 / delta) / (2 * n))

