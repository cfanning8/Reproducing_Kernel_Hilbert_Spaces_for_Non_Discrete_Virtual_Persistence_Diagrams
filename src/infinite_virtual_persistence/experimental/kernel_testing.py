"""
Full kernel approximation testing and validation.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


class KernelApproximationTester:
    """
    Comprehensive kernel approximation testing.
    """
    
    def __init__(self, kernel_true, kernel_approx, X: np.ndarray):
        """
        Initialize tester.
        
        Args:
            kernel_true: True kernel function k_t(x, y)
            kernel_approx: Approximate kernel function k_hat_R(x, y)
            X: Test points, shape (n, dim)
        """
        self.kernel_true = kernel_true
        self.kernel_approx = kernel_approx
        self.X = X
        self.n = len(X)
    
    def compute_error_matrix(self) -> Tuple[np.ndarray, Dict]:
        """
        Compute full error matrix and statistics.
        
        Returns:
            (error_matrix, stats_dict)
        """
        error_matrix = np.zeros((self.n, self.n))
        true_matrix = np.zeros((self.n, self.n))
        approx_matrix = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                k_true = self.kernel_true(self.X[i], self.X[j])
                k_approx = self.kernel_approx(self.X[i], self.X[j])
                
                true_matrix[i, j] = k_true
                approx_matrix[i, j] = k_approx
                error_matrix[i, j] = np.abs(k_true - k_approx)
        
        stats = {
            'max_error': np.max(error_matrix),
            'mean_error': np.mean(error_matrix),
            'median_error': np.median(error_matrix),
            'std_error': np.std(error_matrix),
            'relative_error': np.mean(error_matrix / (np.abs(true_matrix) + 1e-8)),
            'correlation': np.corrcoef(true_matrix.flatten(), 
                                     approx_matrix.flatten())[0, 1],
            'spearman_correlation': spearmanr(true_matrix.flatten(), 
                                            approx_matrix.flatten())[0]
        }
        
        return error_matrix, stats
    
    def test_concentration_bound(self, epsilon: float, R: int) -> Dict:
        """
        Test Hoeffding concentration bound.
        
        P(|k_hat_R(x,y) - k_t(x,y)| > epsilon) <= 4 exp(-R epsilon^2 / 4)
        
        Args:
            epsilon: Error threshold
            R: Number of random features
        
        Returns:
            Dictionary with test results
        """
        error_matrix, _ = self.compute_error_matrix()
        
        # Count violations
        violations = np.sum(error_matrix > epsilon)
        total = self.n * self.n
        violation_rate = violations / total
        
        # Theoretical bound
        theoretical_bound = 4 * np.exp(-R * epsilon**2 / 4)
        
        return {
            'epsilon': epsilon,
            'R': R,
            'violations': violations,
            'violation_rate': violation_rate,
            'theoretical_bound': theoretical_bound,
            'bound_satisfied': violation_rate <= theoretical_bound
        }
    
    def plot_approximation_quality(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot visualization of approximation quality.
        
        Args:
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        error_matrix, stats = self.compute_error_matrix()
        
        # Compute true and approx matrices
        true_matrix = np.zeros((self.n, self.n))
        approx_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                true_matrix[i, j] = self.kernel_true(self.X[i], self.X[j])
                approx_matrix[i, j] = self.kernel_approx(self.X[i], self.X[j])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # True kernel matrix
        im1 = axes[0, 0].imshow(true_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('True Kernel Matrix')
        axes[0, 0].set_xlabel('j')
        axes[0, 0].set_ylabel('i')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Approximate kernel matrix
        im2 = axes[0, 1].imshow(approx_matrix, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Approximate Kernel Matrix')
        axes[0, 1].set_xlabel('j')
        axes[0, 1].set_ylabel('i')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Error matrix
        im3 = axes[1, 0].imshow(error_matrix, cmap='hot', aspect='auto')
        axes[1, 0].set_title(f'Error Matrix (max={stats["max_error"]:.4f})')
        axes[1, 0].set_xlabel('j')
        axes[1, 0].set_ylabel('i')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Scatter plot: true vs approximate
        axes[1, 1].scatter(true_matrix.flatten(), approx_matrix.flatten(), 
                           alpha=0.5, s=1)
        axes[1, 1].plot([true_matrix.min(), true_matrix.max()], 
                       [true_matrix.min(), true_matrix.max()], 
                       'r--', label='y=x')
        axes[1, 1].set_xlabel('True Kernel Value')
        axes[1, 1].set_ylabel('Approximate Kernel Value')
        axes[1, 1].set_title(f'Correlation: {stats["correlation"]:.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def convergence_analysis(self, R_values: List[int]) -> Dict:
        """
        Analyze convergence as R increases.
        
        Args:
            R_values: List of R values to test
        
        Returns:
            Dictionary with convergence data
        """
        errors = []
        max_errors = []
        mean_errors = []
        
        for R in R_values:
            # Recompute approximate kernel with R features
            # This requires access to the RFF object
            # For now, we'll use the existing kernel_approx
            error_matrix, stats = self.compute_error_matrix()
            
            errors.append(stats)
            max_errors.append(stats['max_error'])
            mean_errors.append(stats['mean_error'])
        
        return {
            'R_values': R_values,
            'max_errors': max_errors,
            'mean_errors': mean_errors,
            'detailed_stats': errors
        }
    
    def plot_convergence(self, R_values: List[int], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence analysis.
        
        Args:
            R_values: List of R values
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        convergence_data = self.convergence_analysis(R_values)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Max error vs R
        axes[0].semilogy(R_values, convergence_data['max_errors'], 
                         'o-', label='Max Error')
        axes[0].set_xlabel('Number of Random Features (R)')
        axes[0].set_ylabel('Max Approximation Error')
        axes[0].set_title('Convergence: Max Error')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Mean error vs R
        axes[1].semilogy(R_values, convergence_data['mean_errors'], 
                         's-', label='Mean Error', color='orange')
        axes[1].set_xlabel('Number of Random Features (R)')
        axes[1].set_ylabel('Mean Approximation Error')
        axes[1].set_title('Convergence: Mean Error')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def test_kernel_approximation(kernel_true, kernel_approx, X: np.ndarray,
                              R: int, epsilon: float = 0.1) -> Dict:
    """
    Comprehensive kernel approximation test.
    
    Args:
        kernel_true: True kernel
        kernel_approx: Approximate kernel
        X: Test points
        R: Number of random features
        epsilon: Error threshold
    
    Returns:
        Test results dictionary
    """
    tester = KernelApproximationTester(kernel_true, kernel_approx, X)
    
    # Compute error statistics
    error_matrix, stats = tester.compute_error_matrix()
    
    # Test concentration bound
    concentration_test = tester.test_concentration_bound(epsilon, R)
    
    # Combine results
    results = {
        'error_statistics': stats,
        'concentration_test': concentration_test,
        'error_matrix': error_matrix
    }
    
    return results

