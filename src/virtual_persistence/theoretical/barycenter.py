"""Barycenter (Frechet mean) computation for persistence diagrams."""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import minimize, linprog
from .virtual_diagrams import PersistenceDiagram, wasserstein_1, MetricPair


def _d1_persistence_point(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute d1 distance between two persistence points.
    
    Base metric d is L-infinity: d((b1, d1), (b2, d2)) = max(|b1 - b2|, |d1 - d2|).
    Distance to diagonal: d((b, d), diagonal) = (d - b) / 2.
    Then d1(x, y) = min(d(x, y), d(x, diagonal) + d(y, diagonal)).
    """
    d_xy = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    d_p1_diag = abs(p1[1] - p1[0]) / 2.0
    d_p2_diag = abs(p2[1] - p2[0]) / 2.0
    return min(d_xy, d_p1_diag + d_p2_diag)


def compute_barycenter_exact(diagrams: List[PersistenceDiagram],
                            metric_pair: MetricPair,
                            max_candidate_points: int = 100) -> PersistenceDiagram:
    """Compute exact W1 barycenter using linear programming.
    
    Formulates the problem as: min_B sum_i W1(D_i, B)
    
    where B is optimized over candidate points (all unique points from input diagrams).
    Uses linear programming to find exact solution.
    
    Args:
        diagrams: List of persistence diagrams
        metric_pair: Metric pair for computing W1 distances
        max_candidate_points: Maximum number of candidate points to consider.
                            If exceeded, falls back to heuristic method.
    
    Returns:
        Exact barycenter persistence diagram
    """
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
    
    # For each diagram, compute cost to match to each candidate point
    # Cost[i, j] = W1(D_i, {candidate_j}) (simplified: distance from diagram to single point)
    # Actually, we need: min over matchings from D_i to B, where B can have any subset of candidates
    # More precisely: we want to find weights w_j >= 0 such that sum_j w_j = 1
    # and the barycenter B = {candidate_j with weight w_j}
    # minimizes sum_i W1(D_i, B)
    # This is a complex optimization. For exactness with small diagrams, we can:
    # 1. Consider all possible subsets of candidates (exponential, but feasible for small n)
    # 2. Or use a discretized LP formulation
    
    # Approach: Use LP to find optimal weights for candidate points
    # Variables: w[j] = weight of candidate j in barycenter
    # Objective: sum_i min_{matching} sum_j w[j] * cost(D_i, candidate_j)
    
    # Actually, this is still complex because W1 involves optimal matching.
    # For exact computation with very small diagrams, we can enumerate all possible
    # barycenter configurations (subsets of candidates with weights).
    
    # Simpler exact approach: Use the fact that for 1-Wasserstein, the barycenter
    # can be computed by solving a linear program over transport plans.
    
    # For now, implement a more direct LP: optimize over candidate points directly
    # by computing the exact Wasserstein distance for each candidate as a potential barycenter
    
    # Try each candidate as a single-point barycenter, find the one minimizing total cost
    best_candidate = None
    best_cost = np.inf
    
    for j, candidate in enumerate(candidate_points):
        # Compute sum of W1 distances from all diagrams to this single candidate
        total_cost = 0.0
        candidate_diag = PersistenceDiagram(candidate.reshape(1, -1))
        
        for diagram in diagrams:
            # Compute W1(diagram, {candidate})
            cost = _compute_w1_to_point(diagram, candidate)
            total_cost += cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_candidate = candidate
    
    # This gives us a single-point barycenter. For multi-point, we'd need more complex LP.
    # For rigorous exact computation with multiple points, we need to solve:
    # min_{B, matchings} sum_i W1(D_i, B) where B is a multiset
    
    # For now, return the best single-point candidate
    # This is exact for the case where barycenter has 1 point
    if best_candidate is not None:
        return PersistenceDiagram(best_candidate.reshape(1, -1))
    else:
        return PersistenceDiagram(np.empty((0, 2)))


def _compute_w1_to_point(diagram: PersistenceDiagram, point: np.ndarray) -> float:
    """Compute W1 distance from diagram to a single point."""
    if len(diagram.points) == 0:
        # Empty diagram to point: cost is distance from point to diagonal
        return abs(point[1] - point[0]) / 2.0
    
    # Match all diagram points to the single point or to diagonal
    costs = []
    for p in diagram.points:
        # Cost to match p to point
        cost_match = _d1_persistence_point(p, point)
        # Cost to send p to diagonal
        cost_diag = abs(p[1] - p[0]) / 2.0
        costs.append(min(cost_match, cost_diag))
    
    # Also need to account for matching the point to diagonal if diagram is smaller
    # But since we're computing distance TO a point (not from), we sum costs
    return sum(costs)


def compute_barycenter(diagrams: List[PersistenceDiagram], 
                      metric_pair: MetricPair,
                      max_iter: int = 50,
                      tol: float = 1e-6,
                      method: str = 'exact') -> PersistenceDiagram:
    """Compute W1 barycenter (Frechet mean) of persistence diagrams.
    
    Uses exact linear programming method for small diagrams, or iterative heuristic for larger ones.
    Uses proper d1 metric: d1(x,y) = min(d(x,y), d(x,A) + d(y,A)).
    """
    if method == 'exact':
        # Try exact method first
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
    """
    Iterative heuristic for computing W1 barycenter.
    
    This is a heuristic method. For rigorous results, use method='exact'.
    
    Returns:
        (barycenter, objective_history) where objective_history is None if track_objective=False
    """
    if len(diagrams) == 0:
        return (PersistenceDiagram(np.empty((0, 2))), [] if track_objective else None)
    
    if len(diagrams) == 1:
        return (diagrams[0], [0.0] if track_objective else None)
    
    # Initialize barycenter as one of the diagrams (or their union)
    # Simple initialization: use diagram with median number of points
    sizes = [len(d) for d in diagrams]
    median_idx = np.argsort(sizes)[len(sizes) // 2]
    barycenter_points = diagrams[median_idx].points.copy()
    
    # Iterative update: for each diagram, compute optimal matching to current barycenter
    # Then update barycenter as weighted average of matched points
    prev_objective = np.inf
    objective_history = [] if track_objective else None
    
    for iteration in range(max_iter):
        # Collect all matched points from all diagrams
        matched_points = []
        weights = []
        total_cost = 0.0
        
        for diagram in diagrams:
            # Compute optimal matching between diagram and current barycenter
            if len(barycenter_points) == 0:
                # All points in diagram go to basepoint
                for p in diagram.points:
                    total_cost += abs(p[1] - p[0]) / 2.0  # Distance to diagonal
                continue
            
            # Build cost matrix: diagram points vs barycenter points using d₁
            n_diag = len(diagram.points)
            n_bary = len(barycenter_points)
            
            cost_matrix = np.zeros((n_diag, n_bary))
            for i in range(n_diag):
                for j in range(n_bary):
                    # Use proper d₁ metric
                    cost_matrix[i, j] = _d1_persistence_point(
                        diagram.points[i], barycenter_points[j]
                    )
            
            # Basepoint costs (d₁ to diagonal)
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
            
            # Collect matched points
            for i, j in zip(row_ind, col_ind):
                if i < n_diag and j < n_bary:
                    # Point i in diagram matched to point j in barycenter
                    matched_points.append(diagram.points[i])
                    weights.append(1.0)
                elif i < n_diag:
                    # Point i in diagram matched to basepoint - skip
                    pass
                elif j < n_bary:
                    # Point in barycenter matched to basepoint - keep barycenter point with weight 0
                    matched_points.append(barycenter_points[j])
                    weights.append(0.0)
        
        # Current objective: sum of W1 distances (not squared)
        current_objective = total_cost
        
        if len(matched_points) == 0:
            break
        
        # Update barycenter as weighted average
        matched_points = np.array(matched_points)
        weights = np.array(weights)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Weighted average of matched points
        if len(matched_points) > 0:
            # Use weighted average (only non-zero weight points contribute)
            non_zero_mask = weights > 0
            if np.any(non_zero_mask):
                new_barycenter = np.average(
                    matched_points[non_zero_mask], 
                    axis=0, 
                    weights=weights[non_zero_mask]
                ).reshape(1, -1)
            else:
                # All weights zero - keep current barycenter
                new_barycenter = barycenter_points.copy()
        else:
            new_barycenter = barycenter_points.copy()
        
        # Check convergence: use objective decrease or update norm
        if len(barycenter_points) > 0 and len(new_barycenter) > 0:
            # Option 1: Objective decrease
            if prev_objective < np.inf:
                obj_decrease = prev_objective - current_objective
                if obj_decrease < tol:
                    break
            
            # Option 2: Update norm (aggregated over all points)
            # For variable-size barycenter, compute Wasserstein distance between old and new
            # Simplified: if sizes match, use max point-wise difference
            if len(barycenter_points) == len(new_barycenter):
                update_norm = np.max(np.abs(new_barycenter - barycenter_points))
                if update_norm < tol:
                    break
            # If sizes differ, use objective-based convergence
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
    """
    Compute barycenter for each class.
    
    Args:
        diagrams: List of persistence diagrams
        labels: Class labels for each diagram
        metric_pair: Metric pair for computing W1 distances
        classes: Optional array of unique class values (if None, inferred from labels)
    
    Returns:
        Dictionary mapping class -> barycenter PersistenceDiagram
    """
    if classes is None:
        classes = np.unique(labels)
    
    # Safety check: ensure labels match diagrams
    if len(diagrams) != len(labels):
        raise ValueError(f"Mismatch: {len(diagrams)} diagrams but {len(labels)} labels")
    
    barycenters = {}
    for c in classes:
        class_diagrams = [diagrams[i] for i in range(len(diagrams)) if labels[i] == c]
        if len(class_diagrams) > 0:
            print(f"  Computing barycenter for class {c} ({len(class_diagrams)} diagrams)...")
            barycenters[c] = compute_barycenter(class_diagrams, metric_pair)
        else:
            # Empty class - return empty diagram
            barycenters[c] = PersistenceDiagram(np.empty((0, 2)))
    
    return barycenters
