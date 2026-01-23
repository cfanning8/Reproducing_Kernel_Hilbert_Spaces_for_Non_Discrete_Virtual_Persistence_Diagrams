"""Barycenter (Frechet mean) computation for persistence diagrams."""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import minimize, linprog
from .virtual_diagrams import PersistenceDiagram, wasserstein_1, MetricPair


def _d1_persistence_point(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute d1 distance between two persistence points."""
    d_xy = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    d_p1_diag = abs(p1[1] - p1[0]) / 2.0
    d_p2_diag = abs(p2[1] - p2[0]) / 2.0
    return min(d_xy, d_p1_diag + d_p2_diag)


def compute_barycenter_exact(diagrams: List[PersistenceDiagram],
                            metric_pair: MetricPair,
                            max_candidate_points: int = 100) -> PersistenceDiagram:
    """Compute exact W1 barycenter using linear programming."""
    if len(diagrams) == 0:
        return PersistenceDiagram(np.empty((0, 2)))
    
    if len(diagrams) == 1:
        return diagrams[0]
    
    # Collect all unique candidate points from all diagrams
    all_points = []
    for diagram in diagrams:
        all_points.extend(diagram.points.tolist())
    
    # Get unique points (within numerical tolerance)
    if len(all_points) == 0:
        return PersistenceDiagram(np.empty((0, 2)))
    
    all_points_array = np.array(all_points)
    
    # Remove duplicates (within tolerance)
    unique_points = []
    tol = 1e-8
    for p in all_points_array:
        is_duplicate = False
        for up in unique_points:
            if np.allclose(p, up, atol=tol):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(p)
    
    candidate_points = np.array(unique_points)
    
    if len(candidate_points) > max_candidate_points:
        return compute_barycenter(diagrams, metric_pair)
    
    n_candidates = len(candidate_points)
    n_diagrams = len(diagrams)
    best_candidate = None
    best_cost = np.inf
    
    for j, candidate in enumerate(candidate_points):
        total_cost = 0.0
        candidate_diag = PersistenceDiagram(candidate.reshape(1, -1))
        
        for diagram in diagrams:
            cost = _compute_w1_to_point(diagram, candidate)
            total_cost += cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_candidate = candidate
    if best_candidate is not None:
        return PersistenceDiagram(best_candidate.reshape(1, -1))
    else:
        return PersistenceDiagram(np.empty((0, 2)))


def _compute_w1_to_point(diagram: PersistenceDiagram, point: np.ndarray) -> float:
    """Compute W1 distance from diagram to a single point."""
    if len(diagram.points) == 0:
        return abs(point[1] - point[0]) / 2.0
    
    costs = []
    for p in diagram.points:
        cost_match = _d1_persistence_point(p, point)
        cost_diag = abs(p[1] - p[0]) / 2.0
        costs.append(min(cost_match, cost_diag))
    
    return sum(costs)


def compute_barycenter(diagrams: List[PersistenceDiagram], 
                      metric_pair: MetricPair,
                      max_iter: int = 50,
                      tol: float = 1e-6,
                      method: str = 'exact') -> PersistenceDiagram:
    """Compute W1 barycenter (Frechet mean) of persistence diagrams."""
    if method == 'exact':
        try:
            return compute_barycenter_exact(diagrams, metric_pair, max_candidate_points=50)
        except Exception:
            method = 'iterative'
    
    if method == 'iterative':
        barycenter, _ = _compute_barycenter_iterative(diagrams, metric_pair, max_iter, tol, track_objective=False)
        return barycenter
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'iterative'.")


def _compute_barycenter_iterative(diagrams: List[PersistenceDiagram],
                                  metric_pair: MetricPair,
                                  max_iter: int = 50,
                                  tol: float = 1e-6,
                                  track_objective: bool = False) -> Tuple[PersistenceDiagram, Optional[List[float]]]:
    """Iterative heuristic for computing W1 barycenter."""
    if len(diagrams) == 0:
        return (PersistenceDiagram(np.empty((0, 2))), [] if track_objective else None)
    
    if len(diagrams) == 1:
        return (diagrams[0], [0.0] if track_objective else None)
    
    sizes = [len(d) for d in diagrams]
    median_idx = np.argsort(sizes)[len(sizes) // 2]
    barycenter_points = diagrams[median_idx].points.copy()
    
    prev_objective = np.inf
    objective_history = [] if track_objective else None
    
    for iteration in range(max_iter):
        # Collect all matched points from all diagrams
        matched_points = []
        weights = []
        total_cost = 0.0
        
        for diagram in diagrams:
            if len(barycenter_points) == 0:
                for p in diagram.points:
                    total_cost += abs(p[1] - p[0]) / 2.0
                continue
            n_diag = len(diagram.points)
            n_bary = len(barycenter_points)
            cost_matrix = np.zeros((n_diag, n_bary))
            for i in range(n_diag):
                for j in range(n_bary):
                    cost_matrix[i, j] = _d1_persistence_point(
                        diagram.points[i], barycenter_points[j]
                    )
            basepoint_costs_diag = np.array([
                abs(diagram.points[i, 1] - diagram.points[i, 0]) / 2.0
                for i in range(n_diag)
            ])
            basepoint_costs_bary = np.array([
                abs(barycenter_points[j, 1] - barycenter_points[j, 0]) / 2.0
                for j in range(n_bary)
            ])
            from scipy.optimize import linear_sum_assignment
            
            extended_cost = np.zeros((n_diag + n_bary, n_diag + n_bary))
            extended_cost[:n_diag, :n_bary] = cost_matrix
            extended_cost[:n_diag, n_bary:] = np.diag(basepoint_costs_diag)
            extended_cost[n_diag:, :n_bary] = np.diag(basepoint_costs_bary)
            extended_cost[n_diag:, n_bary:] = 0  # basepoint to basepoint is free
            
            row_ind, col_ind = linear_sum_assignment(extended_cost)
            
            # Compute matching cost for this diagram
            matching_cost = extended_cost[row_ind, col_ind].sum()
            total_cost += matching_cost
            
            for i, j in zip(row_ind, col_ind):
                if i < n_diag and j < n_bary:
                    matched_points.append(diagram.points[i])
                    weights.append(1.0)
                elif i < n_diag:
                    pass
                elif j < n_bary:
                    matched_points.append(barycenter_points[j])
                    weights.append(0.0)
        
        current_objective = total_cost
        
        if len(matched_points) == 0:
            break
        
        matched_points = np.array(matched_points)
        weights = np.array(weights)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        if len(matched_points) > 0:
            non_zero_mask = weights > 0
            if np.any(non_zero_mask):
                new_barycenter = np.average(
                    matched_points[non_zero_mask], 
                    axis=0, 
                    weights=weights[non_zero_mask]
                ).reshape(1, -1)
            else:
                new_barycenter = barycenter_points.copy()
        else:
            new_barycenter = barycenter_points.copy()
        
        if len(barycenter_points) > 0 and len(new_barycenter) > 0:
            if prev_objective < np.inf:
                obj_decrease = prev_objective - current_objective
                if obj_decrease < tol:
                    break
            
            if len(barycenter_points) == len(new_barycenter):
                update_norm = np.max(np.abs(new_barycenter - barycenter_points))
                if update_norm < tol:
                    break
            elif prev_objective < np.inf:
                if abs(prev_objective - current_objective) < tol:
                    break
        
        barycenter_points = new_barycenter
        prev_objective = current_objective
    
    return (PersistenceDiagram(barycenter_points), objective_history)


def compute_class_barycenters(diagrams: List[PersistenceDiagram],
                              labels: np.ndarray,
                              metric_pair: MetricPair,
                              classes: Optional[np.ndarray] = None) -> dict:
    """Compute barycenter for each class."""
    if classes is None:
        classes = np.unique(labels)
    
    if len(diagrams) != len(labels):
        raise ValueError(f"Mismatch: {len(diagrams)} diagrams but {len(labels)} labels")
    
    barycenters = {}
    for c in classes:
        class_diagrams = [diagrams[i] for i in range(len(diagrams)) if labels[i] == c]
        if len(class_diagrams) > 0:
            barycenters[c] = compute_barycenter(class_diagrams, metric_pair)
        else:
            barycenters[c] = PersistenceDiagram(np.empty((0, 2)))
    
    return barycenters
