"""
MOTION ENGINE
=============
A Universe of Motion Rather Than In Motion

Six fundamental motion functions, unfolding sequentially:

1. Heat         → ●           Presence, magnitude only
2. Polarity     → ●━━━━●      Differentiation into +/-  
3. Existence    → ●━━●━━●     Middle (0) appears, persistence
4. Righteousness→ Axes        Orientation along x/y with ±
5. Order        → Labels      Enumeration via Robinson arithmetic
6. Movement     → Vector      Direction chosen (only after scalars exist)

Each function depends on prior functions being established.
An entity "unfolds" through these stages.
"""

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional


# =============================================================================
# MOTION STAGES
# =============================================================================

class Stage(IntEnum):
    """Sequential stages of motion unfolding."""
    HEAT = 0          # Presence only
    POLARITY = 1      # Differentiation (+/-)
    EXISTENCE = 2     # Persistence (middle defined)
    RIGHTEOUSNESS = 3 # Constraint (axes oriented)
    ORDER = 4         # Enumeration (labeled)
    MOVEMENT = 5      # Direction (vector chosen)


# =============================================================================
# ROBINSON ARITHMETIC
# =============================================================================

class Robinson:
    """
    Robinson Arithmetic (Q) - minimal arithmetic without induction.
    
    Provides:
    - 0 (zero)
    - S (successor)
    - Addition (defined recursively)
    - Multiplication (defined recursively)
    
    No induction axiom - enumeration without assuming infinite structure.
    """
    
    @staticmethod
    def zero() -> int:
        return 0
    
    @staticmethod
    def successor(n: int) -> int:
        """S(n) - the next number."""
        return n + 1
    
    @staticmethod
    def is_zero(n: int) -> bool:
        """Axiom: 0 is not a successor of anything."""
        return n == 0
    
    @staticmethod
    def predecessor(n: int) -> Optional[int]:
        """Inverse of successor (partial - undefined for 0)."""
        if n == 0:
            return None
        return n - 1
    
    @staticmethod
    def add(a: int, b: int) -> int:
        """Addition: a + 0 = a, a + S(b) = S(a + b)."""
        if b == 0:
            return a
        return Robinson.successor(Robinson.add(a, Robinson.predecessor(b)))
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiplication: a * 0 = 0, a * S(b) = a + (a * b)."""
        if b == 0:
            return 0
        return Robinson.add(a, Robinson.multiply(a, Robinson.predecessor(b)))


# =============================================================================
# VECTOR
# =============================================================================

@dataclass
class Vector2:
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other): return Vector2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Vector2(self.x - other.x, self.y - other.y)
    def __mul__(self, s): return Vector2(self.x * s, self.y * s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return Vector2(self.x / s, self.y / s) if s != 0 else Vector2()
    
    def magnitude(self): return math.sqrt(self.x * self.x + self.y * self.y)
    def normalized(self):
        m = self.magnitude()
        return Vector2(self.x / m, self.y / m) if m > 0 else Vector2()
    def distance_to(self, other): return (self - other).magnitude()
    def distance_sq(self, other): 
        dx, dy = self.x - other.x, self.y - other.y
        return dx * dx + dy * dy


# =============================================================================
# CONSTRAINT OPERATORS (n³+t Phase 3)
# =============================================================================

class ConstraintOperator:
    """
    Base class for constraint operators acting on structural volume V.
    
    Operators transform the state distribution over V. They are derived from
    stabilized motion function constraints (polarity, righteousness, order).
    
    Key property: operators may not commute (AB ≠ BA), which is the algebraic
    expression of irreducible incompatibility in the cube closure.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        """
        Compute constraint weights for each occupied cell.
        
        Returns a dict mapping cell -> weight where:
        - weight > 1.0: constraint amplifies this cell
        - weight = 1.0: constraint preserves this cell  
        - weight < 1.0: constraint suppresses this cell
        - weight = 0.0: constraint forbids this cell
        """
        raise NotImplementedError
    
    def apply(self, state: Dict[Tuple[int, int, int], float], 
              engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        """
        Apply this operator to a state vector.
        
        State is a dict mapping cell -> amplitude (can be complex in general,
        but we use float for real-valued constraints).
        """
        weights = self.compute_weights(engine)
        new_state = {}
        for cell, amplitude in state.items():
            weight = weights.get(cell, 1.0)
            new_amplitude = amplitude * weight
            if abs(new_amplitude) > 1e-10:  # Filter negligible amplitudes
                new_state[cell] = new_amplitude
        return new_state
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class PolarityConstraint(ConstraintOperator):
    """
    Constraint derived from polarity: opposite polarities attract, like repel.
    
    Cells containing entities with favorable polar relationships to their
    structural neighbors are amplified; unfavorable relationships are suppressed.
    """
    
    def __init__(self, attraction_bonus: float = 1.2, repulsion_penalty: float = 0.8):
        super().__init__("polarity")
        self.attraction_bonus = attraction_bonus
        self.repulsion_penalty = repulsion_penalty
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        weights = {}
        
        for cell, eid in engine.occupied_cells.items():
            entity = engine.by_id.get(eid)
            if not entity or not entity.has_polarity():
                weights[cell] = 1.0
                continue
            
            # Evaluate polar relationships with structural neighbors
            neighbors = engine.get_structural_neighbors(entity)
            polar_neighbors = [n for n in neighbors if n.has_polarity()]
            
            if not polar_neighbors:
                weights[cell] = 1.0
                continue
            
            # Count favorable (opposite) vs unfavorable (same) polarities
            favorable = 0
            unfavorable = 0
            for neighbor in polar_neighbors:
                product = entity.net_polarity * neighbor.net_polarity
                if product < -0.1:  # Opposite polarities
                    favorable += 1
                elif product > 0.1:  # Same polarities
                    unfavorable += 1
            
            # Compute weight based on balance
            total = favorable + unfavorable
            if total == 0:
                weights[cell] = 1.0
            else:
                ratio = favorable / total
                # Interpolate between repulsion_penalty and attraction_bonus
                weights[cell] = self.repulsion_penalty + ratio * (self.attraction_bonus - self.repulsion_penalty)
        
        return weights


class RighteousnessConstraint(ConstraintOperator):
    """
    Constraint derived from righteousness: structural validity and alignment.
    
    Cells containing entities with high righteousness (valid configurations)
    are amplified; low righteousness cells are suppressed.
    """
    
    def __init__(self, validity_threshold: float = 0.5, amplification: float = 1.3):
        super().__init__("righteousness")
        self.validity_threshold = validity_threshold
        self.amplification = amplification
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        weights = {}
        
        for cell, eid in engine.occupied_cells.items():
            entity = engine.by_id.get(eid)
            if not entity or not entity.has_righteousness():
                weights[cell] = 1.0
                continue
            
            # Weight based on righteousness value
            r = entity.righteousness
            if r >= self.validity_threshold:
                # Above threshold: amplify proportionally
                weights[cell] = 1.0 + (r - self.validity_threshold) * (self.amplification - 1.0) / (1.0 - self.validity_threshold)
            else:
                # Below threshold: suppress proportionally
                weights[cell] = r / self.validity_threshold
        
        return weights


class OrderConstraint(ConstraintOperator):
    """
    Constraint derived from order: bonding rules and enumeration consistency.
    
    Cells containing entities with satisfied bonds to structural neighbors
    are amplified; broken or missing bonds are penalized.
    """
    
    def __init__(self, bond_satisfaction_bonus: float = 1.25, isolation_penalty: float = 0.7):
        super().__init__("order")
        self.bond_satisfaction_bonus = bond_satisfaction_bonus
        self.isolation_penalty = isolation_penalty
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        weights = {}
        
        for cell, eid in engine.occupied_cells.items():
            entity = engine.by_id.get(eid)
            if not entity or not entity.has_order():
                weights[cell] = 1.0
                continue
            
            # Check how many bonds are satisfied by structural adjacency
            if not entity.bonds:
                # No bonds - slight isolation penalty
                weights[cell] = self.isolation_penalty
                continue
            
            structural_neighbor_ids = {n.id for n in engine.get_structural_neighbors(entity)}
            satisfied = sum(1 for bid in entity.bonds if bid in structural_neighbor_ids)
            
            satisfaction_ratio = satisfied / len(entity.bonds)
            # Interpolate between isolation_penalty and bond_satisfaction_bonus
            weights[cell] = self.isolation_penalty + satisfaction_ratio * (self.bond_satisfaction_bonus - self.isolation_penalty)
        
        return weights


class HeatConstraint(ConstraintOperator):
    """
    Constraint derived from heat: magnitude of presence.
    
    Cells with higher heat have stronger presence in the state distribution.
    This is the most fundamental constraint - pure magnitude without structure.
    """
    
    def __init__(self, heat_exponent: float = 1.0):
        super().__init__("heat")
        self.heat_exponent = heat_exponent
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        weights = {}
        
        # Find max heat for normalization
        max_heat = max((e.heat for e in engine.entities), default=1.0)
        if max_heat <= 0:
            max_heat = 1.0
        
        for cell, eid in engine.occupied_cells.items():
            entity = engine.by_id.get(eid)
            if not entity:
                weights[cell] = 1.0
                continue
            
            # Normalize heat to [0, 1] range then apply exponent
            normalized = entity.heat / max_heat
            weights[cell] = normalized ** self.heat_exponent
        
        return weights


class AxisShiftOperator(ConstraintOperator):
    """
    Operator that shifts amplitude along a specific axis.
    
    This is a non-diagonal operator: it transfers amplitude from cell (i,j,k)
    to cell (i±1,j,k) (for axis 0), creating genuine non-commutation.
    
    These operators represent the structural consequences of applying
    constraints in different orders - the cube's irreducible incompatibility.
    """
    
    def __init__(self, axis: int, direction: int = 1, transfer_rate: float = 0.3):
        """
        Args:
            axis: Which axis to shift along (0=i, 1=j, 2=k)
            direction: +1 for successor, -1 for predecessor
            transfer_rate: Fraction of amplitude to transfer (0 to 1)
        """
        axis_names = ['i', 'j', 'k']
        dir_names = {1: '+', -1: '-'}
        super().__init__(f"shift_{axis_names[axis]}{dir_names[direction]}")
        self.axis = axis
        self.direction = direction
        self.transfer_rate = transfer_rate
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        # For shift operators, weights represent what remains (1 - transfer_rate)
        return {cell: (1.0 - self.transfer_rate) for cell in engine.occupied_cells}
    
    def apply(self, state: Dict[Tuple[int, int, int], float],
              engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        """
        Apply shift: some amplitude moves from each cell to its neighbor along axis.
        """
        new_state = {}
        
        for cell, amplitude in state.items():
            # Amplitude that stays
            stay_amount = amplitude * (1.0 - self.transfer_rate)
            new_state[cell] = new_state.get(cell, 0.0) + stay_amount
            
            # Compute target cell
            target = list(cell)
            target[self.axis] += self.direction
            target = tuple(target)
            
            # Only transfer if target cell is occupied (structure exists there)
            if target in engine.occupied_cells:
                transfer_amount = amplitude * self.transfer_rate
                new_state[target] = new_state.get(target, 0.0) + transfer_amount
            else:
                # If no target, amplitude stays (reflected at boundary)
                new_state[cell] = new_state.get(cell, 0.0) + amplitude * self.transfer_rate
        
        # Filter negligible values
        return {c: a for c, a in new_state.items() if abs(a) > 1e-10}


class PolarShiftOperator(ConstraintOperator):
    """
    Polarity-driven amplitude transfer between structurally adjacent cells.
    
    Amplitude flows from cells with positive net polarity toward cells with
    negative net polarity (attraction), creating polarity-dependent dynamics.
    
    This is a non-diagonal operator that couples polarity to structural flow.
    """
    
    def __init__(self, transfer_rate: float = 0.2):
        super().__init__("polar_shift")
        self.transfer_rate = transfer_rate
    
    def apply(self, state: Dict[Tuple[int, int, int], float],
              engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        new_state = dict(state)  # Start with copy
        
        for cell, amplitude in state.items():
            eid = engine.occupied_cells.get(cell)
            if not eid:
                continue
            entity = engine.by_id.get(eid)
            if not entity or not entity.has_polarity():
                continue
            
            # Find structural neighbors with opposite polarity
            neighbors = engine.get_structural_neighbors(entity)
            attractive_neighbors = [
                n for n in neighbors 
                if n.has_polarity() and n.has_cell() and 
                   entity.net_polarity * n.net_polarity < -0.1
            ]
            
            if not attractive_neighbors:
                continue
            
            # Transfer amplitude toward attractive neighbors
            transfer_per_neighbor = (amplitude * self.transfer_rate) / len(attractive_neighbors)
            total_transfer = 0.0
            
            for neighbor in attractive_neighbors:
                new_state[neighbor.cell] = new_state.get(neighbor.cell, 0.0) + transfer_per_neighbor
                total_transfer += transfer_per_neighbor
            
            new_state[cell] -= total_transfer
        
        return {c: a for c, a in new_state.items() if abs(a) > 1e-10}


class RighteousnessFlowOperator(ConstraintOperator):
    """
    Righteousness-driven amplitude flow toward valid configurations.
    
    Amplitude flows from low-righteousness cells to high-righteousness 
    neighbors, concentrating presence in structurally valid regions.
    """
    
    def __init__(self, transfer_rate: float = 0.15):
        super().__init__("righteous_flow")
        self.transfer_rate = transfer_rate
    
    def apply(self, state: Dict[Tuple[int, int, int], float],
              engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        new_state = dict(state)
        
        for cell, amplitude in state.items():
            eid = engine.occupied_cells.get(cell)
            if not eid:
                continue
            entity = engine.by_id.get(eid)
            if not entity or not entity.has_righteousness():
                continue
            
            # Find neighbors with higher righteousness
            neighbors = engine.get_structural_neighbors(entity)
            better_neighbors = [
                n for n in neighbors
                if n.has_righteousness() and n.has_cell() and
                   n.righteousness > entity.righteousness
            ]
            
            if not better_neighbors:
                continue
            
            # Transfer proportional to righteousness difference
            total_diff = sum(n.righteousness - entity.righteousness for n in better_neighbors)
            if total_diff <= 0:
                continue
            
            transfer_budget = amplitude * self.transfer_rate
            
            for neighbor in better_neighbors:
                diff = neighbor.righteousness - entity.righteousness
                fraction = diff / total_diff
                transfer = transfer_budget * fraction
                new_state[neighbor.cell] = new_state.get(neighbor.cell, 0.0) + transfer
                new_state[cell] -= transfer
        
        return {c: a for c, a in new_state.items() if abs(a) > 1e-10}


class CompositeOperator(ConstraintOperator):
    """
    Composition of multiple constraint operators.
    
    The order of operators matters - this is where non-commutation manifests.
    CompositeOperator([A, B]) applies B first, then A: result = A(B(state))
    """
    
    def __init__(self, operators: List[ConstraintOperator]):
        names = " · ".join(op.name for op in operators)
        super().__init__(f"({names})")
        self.operators = operators
    
    def compute_weights(self, engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        # For composite, we compute effective weights by multiplying
        # This is an approximation - true composition requires full apply()
        all_cells = set(engine.occupied_cells.keys())
        combined = {cell: 1.0 for cell in all_cells}
        
        for op in reversed(self.operators):  # Apply in reverse order
            weights = op.compute_weights(engine)
            for cell in all_cells:
                combined[cell] *= weights.get(cell, 1.0)
        
        return combined
    
    def apply(self, state: Dict[Tuple[int, int, int], float],
              engine: 'MotionEngine') -> Dict[Tuple[int, int, int], float]:
        """Apply operators in sequence (rightmost first)."""
        current = state
        for op in reversed(self.operators):
            current = op.apply(current, engine)
        return current


# =============================================================================
# STATE EVOLUTION AND INCOMPATIBILITY TRACKING (n³+t Phase 4)
# =============================================================================

@dataclass
class StateSnapshot:
    """A snapshot of the state vector at a specific unfolding step."""
    step: int                                          # t in n³+t
    state: Dict[Tuple[int, int, int], float]          # |ψ(t)⟩
    operator_applied: Optional[str] = None             # Name of operator that produced this state
    entropy: float = 0.0                               # Shannon entropy of distribution
    total_amplitude: float = 0.0                       # Sum of |amplitudes|
    
    def __post_init__(self):
        if self.state:
            # Compute entropy: -Σ p_i log(p_i) where p_i = |ψ_i|²
            probs = [a * a for a in self.state.values()]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                self.entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            self.total_amplitude = sum(abs(a) for a in self.state.values())


class StateEvolution:
    """
    Tracks the evolution of the state vector through operator applications.
    
    This implements the n³+t dynamics: |ψ(t+1)⟩ = U(t)|ψ(t)⟩
    where U(t) is composed from constraint operators whose order matters.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.history: List[StateSnapshot] = []
        self.operators_applied: List[str] = []
        self.cumulative_incompatibility: float = 0.0
        
        # Record initial state
        initial = engine.get_state_vector()
        self.history.append(StateSnapshot(
            step=engine.unfolding_step,
            state=initial,
            operator_applied=None
        ))
    
    @property
    def current_state(self) -> Dict[Tuple[int, int, int], float]:
        """Get the current state vector."""
        return self.history[-1].state if self.history else {}
    
    @property
    def current_step(self) -> int:
        """Get the current unfolding step."""
        return self.history[-1].step if self.history else 0
    
    def apply(self, operator: ConstraintOperator, 
              track_incompatibility_with: Optional[ConstraintOperator] = None) -> Dict[Tuple[int, int, int], float]:
        """
        Apply an operator and record the evolution.
        
        Args:
            operator: The operator to apply
            track_incompatibility_with: If provided, measure and accumulate 
                                        incompatibility with this operator
        
        Returns:
            New state vector after application
        """
        current = self.current_state
        new_state = operator.apply(current, self.engine)
        
        # Track incompatibility if requested
        if track_incompatibility_with is not None:
            incomp = self.engine.measure_commutator(operator, track_incompatibility_with, current)
            self.cumulative_incompatibility += incomp
        
        # Record
        self.operators_applied.append(operator.name)
        self.history.append(StateSnapshot(
            step=self.current_step + 1,
            state=new_state,
            operator_applied=operator.name
        ))
        
        return new_state
    
    def apply_sequence(self, operators: List[ConstraintOperator]) -> Dict[Tuple[int, int, int], float]:
        """
        Apply a sequence of operators, tracking evolution at each step.
        
        Operators are applied left-to-right.
        """
        for i, op in enumerate(operators):
            # Track incompatibility with previous operator
            prev_op = operators[i-1] if i > 0 else None
            self.apply(op, track_incompatibility_with=prev_op)
        
        return self.current_state
    
    def get_entropy_history(self) -> List[Tuple[int, float]]:
        """Get (step, entropy) pairs for the evolution history."""
        return [(s.step, s.entropy) for s in self.history]
    
    def get_amplitude_flow(self, cell: Tuple[int, int, int]) -> List[Tuple[int, float]]:
        """Track amplitude at a specific cell through evolution."""
        return [(s.step, s.state.get(cell, 0.0)) for s in self.history]
    
    def compute_state_distance(self, step1: int, step2: int) -> float:
        """
        Compute L2 distance between states at two steps.
        
        Measures how much the state has changed.
        """
        if step1 >= len(self.history) or step2 >= len(self.history):
            return 0.0
        
        state1 = self.history[step1].state
        state2 = self.history[step2].state
        
        all_cells = set(state1.keys()) | set(state2.keys())
        diff_sq = sum((state1.get(c, 0) - state2.get(c, 0))**2 for c in all_cells)
        return math.sqrt(diff_sq)
    
    def summary(self) -> Dict[str, any]:
        """Get a summary of the evolution."""
        return {
            'total_steps': len(self.history),
            'operators_applied': len(self.operators_applied),
            'cumulative_incompatibility': self.cumulative_incompatibility,
            'initial_entropy': self.history[0].entropy if self.history else 0,
            'final_entropy': self.history[-1].entropy if self.history else 0,
            'entropy_change': (self.history[-1].entropy - self.history[0].entropy) if self.history else 0,
        }


class UncertaintyRelation:
    """
    Computes uncertainty-like relations from non-commuting operators.
    
    When [A, B] ≠ 0, measuring/constraining along A affects the ability
    to simultaneously constrain along B. This is structural, not epistemic.
    """
    
    def __init__(self, engine: 'MotionEngine', op_a: ConstraintOperator, op_b: ConstraintOperator):
        self.engine = engine
        self.op_a = op_a
        self.op_b = op_b
    
    def compute_variances(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Tuple[float, float]:
        """
        Compute variance of each operator's effect on the state.
        
        Variance = ⟨O²⟩ - ⟨O⟩²
        """
        if state is None:
            state = self.engine.get_state_vector()
        
        def variance_for(op: ConstraintOperator) -> float:
            weights = op.compute_weights(self.engine)
            
            # ⟨O⟩ = Σ |ψ_i|² × O_ii
            expectation = 0.0
            expectation_sq = 0.0
            
            for cell, amp in state.items():
                prob = amp * amp
                w = weights.get(cell, 1.0)
                expectation += prob * w
                expectation_sq += prob * w * w
            
            # Var = ⟨O²⟩ - ⟨O⟩²
            return max(0, expectation_sq - expectation * expectation)
        
        return variance_for(self.op_a), variance_for(self.op_b)
    
    def compute_commutator_bound(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Compute the commutator magnitude |⟨[A,B]⟩|.
        
        This provides a lower bound for the uncertainty product.
        """
        if state is None:
            state = self.engine.get_state_vector()
        
        return self.engine.measure_commutator(self.op_a, self.op_b, state)
    
    def compute_uncertainty_product(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Dict[str, float]:
        """
        Compute the uncertainty product and check the uncertainty relation.
        
        For non-commuting operators: ΔA × ΔB ≥ ½|⟨[A,B]⟩|
        
        Returns dict with variances, standard deviations, product, and bound.
        """
        if state is None:
            state = self.engine.get_state_vector()
        
        var_a, var_b = self.compute_variances(state)
        std_a, std_b = math.sqrt(var_a), math.sqrt(var_b)
        product = std_a * std_b
        
        commutator = self.compute_commutator_bound(state)
        bound = commutator / 2.0
        
        return {
            'variance_a': var_a,
            'variance_b': var_b,
            'std_a': std_a,
            'std_b': std_b,
            'uncertainty_product': product,
            'commutator_bound': bound,
            'relation_satisfied': product >= bound - 1e-10,
            'excess': product - bound,
        }


class IncompatibilityTracker:
    """
    Tracks accumulated incompatibility across the structural volume.
    
    As operators are applied in sequence, incompatibilities accumulate.
    This tracks which regions of V experience the most constraint conflict.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.cell_incompatibility: Dict[Tuple[int, int, int], float] = {}
        self.pair_history: List[Tuple[str, str, float]] = []
        self.total_incompatibility: float = 0.0
    
    def record_application(self, op_a: ConstraintOperator, op_b: ConstraintOperator,
                           state: Optional[Dict[Tuple[int, int, int], float]] = None):
        """
        Record the incompatibility from applying op_a after op_b.
        
        Distributes incompatibility to cells based on their state amplitude.
        """
        if state is None:
            state = self.engine.get_state_vector()
        
        # Global incompatibility
        incomp = self.engine.measure_commutator(op_a, op_b, state)
        self.total_incompatibility += incomp
        self.pair_history.append((op_a.name, op_b.name, incomp))
        
        # Distribute to cells weighted by amplitude
        total_amp = sum(abs(a) for a in state.values())
        if total_amp > 0:
            for cell, amp in state.items():
                weight = abs(amp) / total_amp
                cell_contrib = incomp * weight
                self.cell_incompatibility[cell] = self.cell_incompatibility.get(cell, 0.0) + cell_contrib
    
    def get_hotspots(self, top_n: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
        """Get cells with highest accumulated incompatibility."""
        sorted_cells = sorted(self.cell_incompatibility.items(), key=lambda x: -x[1])
        return sorted_cells[:top_n]
    
    def get_pair_statistics(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Get statistics for each operator pair."""
        from collections import defaultdict
        
        pair_data = defaultdict(list)
        for a, b, incomp in self.pair_history:
            pair_data[(a, b)].append(incomp)
        
        stats = {}
        for pair, values in pair_data.items():
            stats[pair] = {
                'count': len(values),
                'total': sum(values),
                'mean': sum(values) / len(values),
                'max': max(values),
            }
        return stats
    
    def summary(self) -> Dict[str, any]:
        """Get summary of accumulated incompatibility."""
        return {
            'total_incompatibility': self.total_incompatibility,
            'applications_tracked': len(self.pair_history),
            'cells_affected': len(self.cell_incompatibility),
            'hotspots': self.get_hotspots(3),
        }


# =============================================================================
# EMERGENT PHYSICS (n³+t Phase 5)
# =============================================================================

class MotionPhysics:
    """
    Emergent physics derived from motion function compositions.
    
    Physics is not imposed externally but emerges from:
    - Heat → Energy (magnitude of presence)
    - Polarity → Charge-like interactions
    - Existence → Mass/inertia (persistence through time)
    - Righteousness → Potential energy (structural validity)
    - Order → Quantization (discrete structure)
    - Movement → Momentum (directed motion)
    
    Conservation laws arise from symmetries in operator compositions.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.history: List[Dict[str, float]] = []
    
    # -------------------------------------------------------------------------
    # Energy-like quantities
    # -------------------------------------------------------------------------
    
    def compute_kinetic_energy(self, entity: 'MotionEntity') -> float:
        """
        Kinetic energy: derived from heat and movement.
        
        K = ½ × heat × |movement|²
        
        Heat acts as "mass" (magnitude of presence), movement as velocity.
        """
        if not entity.has_movement():
            return 0.0
        return 0.5 * entity.heat * entity.movement.magnitude() ** 2
    
    def compute_potential_energy(self, entity: 'MotionEntity') -> float:
        """
        Potential energy: derived from righteousness deficit.
        
        U = heat × (1 - righteousness)
        
        Low righteousness = high potential (unstable configuration).
        High righteousness = low potential (stable configuration).
        """
        if not entity.has_righteousness():
            return 0.0
        return entity.heat * (1.0 - entity.righteousness)
    
    def compute_polar_energy(self, entity: 'MotionEntity') -> float:
        """
        Polar interaction energy: stored in polarity magnitude.
        
        E_polar = ½ × polarity_magnitude²
        
        Represents capacity for polar interactions.
        """
        if not entity.has_polarity():
            return 0.0
        return 0.5 * entity.polarity_magnitude ** 2
    
    def compute_existence_energy(self, entity: 'MotionEntity') -> float:
        """
        Existence energy: persistence contributes to total energy.
        
        E_exist = heat × existence
        
        This is the "identity mass" from the existence paper.
        """
        if not entity.has_existence():
            return entity.heat  # Pre-existence: just heat
        return entity.heat * entity.existence
    
    def compute_structural_energy(self, entity: 'MotionEntity') -> float:
        """
        Structural energy: energy from position in n³ volume.
        
        Based on structural neighborhood density and bond satisfaction.
        """
        if not entity.has_cell():
            return 0.0
        
        neighbors = self.engine.get_structural_neighbors(entity)
        neighbor_count = len(neighbors)
        
        # Energy from structural isolation (fewer neighbors = higher energy)
        max_neighbors = 26  # 3³ - 1
        isolation_energy = entity.heat * (1.0 - neighbor_count / max_neighbors)
        
        # Energy from unsatisfied bonds
        if entity.bonds:
            structural_neighbor_ids = {n.id for n in neighbors}
            unsatisfied = sum(1 for bid in entity.bonds if bid not in structural_neighbor_ids)
            bond_energy = entity.heat * 0.1 * unsatisfied
        else:
            bond_energy = 0.0
        
        return isolation_energy + bond_energy
    
    def compute_total_energy(self, entity: 'MotionEntity') -> float:
        """
        Total energy of an entity: sum of all energy forms.
        
        E_total = K + U + E_polar + E_struct
        """
        return (self.compute_kinetic_energy(entity) +
                self.compute_potential_energy(entity) +
                self.compute_polar_energy(entity) +
                self.compute_structural_energy(entity))
    
    def compute_system_energy(self) -> Dict[str, float]:
        """
        Compute total energy of the entire system.
        
        Returns breakdown by energy type.
        """
        kinetic = sum(self.compute_kinetic_energy(e) for e in self.engine.entities)
        potential = sum(self.compute_potential_energy(e) for e in self.engine.entities)
        polar = sum(self.compute_polar_energy(e) for e in self.engine.entities)
        structural = sum(self.compute_structural_energy(e) for e in self.engine.entities)
        existence = sum(self.compute_existence_energy(e) for e in self.engine.entities)
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'polar': polar,
            'structural': structural,
            'existence': existence,
            'total': kinetic + potential + polar + structural,
        }
    
    # -------------------------------------------------------------------------
    # Momentum-like quantities
    # -------------------------------------------------------------------------
    
    def compute_momentum(self, entity: 'MotionEntity') -> Vector2:
        """
        Momentum: p = heat × movement
        
        Heat acts as mass, movement as velocity.
        """
        if not entity.has_movement():
            return Vector2()
        return entity.movement * entity.heat
    
    def compute_system_momentum(self) -> Vector2:
        """Total momentum of the system."""
        total = Vector2()
        for e in self.engine.entities:
            total = total + self.compute_momentum(e)
        return total
    
    def compute_angular_momentum(self, entity: 'MotionEntity', origin: Vector2 = None) -> float:
        """
        Angular momentum about an origin: L = r × p (scalar in 2D).
        """
        if not entity.has_movement():
            return 0.0
        
        if origin is None:
            origin = Vector2(self.engine.width / 2, self.engine.height / 2)
        
        r = entity.position - origin
        p = self.compute_momentum(entity)
        
        # Cross product in 2D gives scalar
        return r.x * p.y - r.y * p.x
    
    def compute_system_angular_momentum(self, origin: Vector2 = None) -> float:
        """Total angular momentum of the system."""
        return sum(self.compute_angular_momentum(e, origin) for e in self.engine.entities)
    
    # -------------------------------------------------------------------------
    # Conservation tracking
    # -------------------------------------------------------------------------
    
    def record_state(self):
        """Record current physical quantities for conservation tracking."""
        energy = self.compute_system_energy()
        momentum = self.compute_system_momentum()
        angular = self.compute_system_angular_momentum()
        
        self.history.append({
            'step': self.engine.unfolding_step,
            'time': self.engine.time,
            'total_energy': energy['total'],
            'kinetic': energy['kinetic'],
            'potential': energy['potential'],
            'polar': energy['polar'],
            'structural': energy['structural'],
            'momentum_x': momentum.x,
            'momentum_y': momentum.y,
            'momentum_mag': momentum.magnitude(),
            'angular_momentum': angular,
            'total_heat': self.engine.total_heat,
            'entity_count': len(self.engine.entities),
        })
    
    def get_conservation_report(self) -> Dict[str, any]:
        """
        Analyze conservation of physical quantities over recorded history.
        
        Returns variance and drift for each conserved quantity.
        """
        if len(self.history) < 2:
            return {'error': 'Need at least 2 recorded states'}
        
        def stats(values):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            drift = values[-1] - values[0]
            return {'mean': mean, 'variance': variance, 'std': math.sqrt(variance), 
                    'drift': drift, 'drift_pct': 100 * drift / (mean + 1e-10)}
        
        return {
            'total_energy': stats([h['total_energy'] for h in self.history]),
            'momentum_mag': stats([h['momentum_mag'] for h in self.history]),
            'angular_momentum': stats([h['angular_momentum'] for h in self.history]),
            'total_heat': stats([h['total_heat'] for h in self.history]),
            'entity_count': stats([h['entity_count'] for h in self.history]),
            'samples': len(self.history),
        }
    
    # -------------------------------------------------------------------------
    # Force derivation from gradients
    # -------------------------------------------------------------------------
    
    def compute_righteousness_force(self, entity: 'MotionEntity') -> Vector2:
        """
        Force derived from righteousness gradient.
        
        Entities experience force toward higher righteousness regions.
        F = -∇U where U = potential energy from righteousness.
        """
        if not entity.has_righteousness() or not entity.has_cell():
            return Vector2()
        
        neighbors = self.engine.get_structural_neighbors(entity)
        righteous_neighbors = [n for n in neighbors if n.has_righteousness()]
        
        if not righteous_neighbors:
            return Vector2()
        
        # Force points toward higher righteousness
        force = Vector2()
        for neighbor in righteous_neighbors:
            diff = neighbor.righteousness - entity.righteousness
            if diff > 0:  # Neighbor has higher righteousness
                direction = neighbor.position - entity.position
                if direction.magnitude() > 0:
                    direction = direction.normalized()
                    force = force + direction * diff * entity.heat
        
        return force
    
    def compute_polar_force(self, entity: 'MotionEntity') -> Vector2:
        """
        Force from polar interactions with structural neighbors.
        
        Opposite polarities attract, same polarities repel.
        """
        if not entity.has_polarity() or not entity.has_cell():
            return Vector2()
        
        neighbors = self.engine.get_structural_neighbors(entity)
        polar_neighbors = [n for n in neighbors if n.has_polarity()]
        
        if not polar_neighbors:
            return Vector2()
        
        force = Vector2()
        for neighbor in polar_neighbors:
            product = entity.net_polarity * neighbor.net_polarity
            direction = neighbor.position - entity.position
            dist = direction.magnitude()
            
            if dist > 0:
                direction = direction.normalized()
                if product < 0:  # Opposite polarities: attract
                    force = force + direction * abs(product)
                else:  # Same polarities: repel
                    force = force - direction * product
        
        return force
    
    def compute_structural_force(self, entity: 'MotionEntity') -> Vector2:
        """
        Force from structural volume gradients.
        
        Entities are pushed toward regions of higher structural density.
        """
        if not entity.has_cell():
            return Vector2()
        
        neighbors = self.engine.get_structural_neighbors(entity)
        if not neighbors:
            return Vector2()
        
        # Compute centroid of structural neighbors
        centroid = Vector2()
        for n in neighbors:
            centroid = centroid + n.position
        centroid = centroid / len(neighbors)
        
        # Force toward centroid (cohesion with structure)
        direction = centroid - entity.position
        if direction.magnitude() > 0:
            return direction.normalized() * 0.1 * entity.heat
        return Vector2()
    
    def compute_total_force(self, entity: 'MotionEntity') -> Vector2:
        """
        Total emergent force on an entity.
        """
        return (self.compute_righteousness_force(entity) +
                self.compute_polar_force(entity) +
                self.compute_structural_force(entity))
    
    # -------------------------------------------------------------------------
    # Symmetry and conservation laws
    # -------------------------------------------------------------------------
    
    def check_translation_symmetry(self) -> Dict[str, float]:
        """
        Check translational symmetry by measuring momentum conservation.
        
        If the system has translation symmetry, momentum should be conserved.
        """
        if len(self.history) < 2:
            return {'symmetric': True, 'momentum_drift': 0.0}
        
        p_initial = Vector2(self.history[0]['momentum_x'], self.history[0]['momentum_y'])
        p_final = Vector2(self.history[-1]['momentum_x'], self.history[-1]['momentum_y'])
        
        drift = (p_final - p_initial).magnitude()
        initial_mag = p_initial.magnitude() + 1e-10
        
        return {
            'symmetric': drift / initial_mag < 0.1,
            'momentum_drift': drift,
            'drift_fraction': drift / initial_mag,
        }
    
    def check_rotation_symmetry(self) -> Dict[str, float]:
        """
        Check rotational symmetry by measuring angular momentum conservation.
        
        If the system has rotation symmetry, angular momentum should be conserved.
        """
        if len(self.history) < 2:
            return {'symmetric': True, 'angular_drift': 0.0}
        
        L_initial = self.history[0]['angular_momentum']
        L_final = self.history[-1]['angular_momentum']
        
        drift = abs(L_final - L_initial)
        initial_mag = abs(L_initial) + 1e-10
        
        return {
            'symmetric': drift / initial_mag < 0.1,
            'angular_drift': drift,
            'drift_fraction': drift / initial_mag,
        }
    
    def check_heat_conservation(self) -> Dict[str, float]:
        """
        Check heat conservation.
        
        Total heat should be conserved if heat_conserve=True in engine params.
        """
        if len(self.history) < 2:
            return {'conserved': True, 'heat_drift': 0.0}
        
        h_initial = self.history[0]['total_heat']
        h_final = self.history[-1]['total_heat']
        
        drift = abs(h_final - h_initial)
        initial_mag = abs(h_initial) + 1e-10
        
        return {
            'conserved': drift / initial_mag < 0.05,
            'heat_drift': drift,
            'drift_fraction': drift / initial_mag,
        }


class EmergentQuantization:
    """
    Quantization effects emerging from discrete structural volume.
    
    The n³ structure naturally introduces discreteness. This class
    computes quantized quantities and energy level-like structures.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
    
    def get_structural_levels(self) -> Dict[int, List['MotionEntity']]:
        """
        Group entities by their structural "level" (sum of cell indices).
        
        This creates discrete energy-like levels from structure.
        """
        levels = {}
        for e in self.engine.entities:
            if e.has_cell():
                level = sum(e.cell)
                if level not in levels:
                    levels[level] = []
                levels[level].append(e)
        return levels
    
    def get_axis_occupancy(self) -> Dict[str, Dict[int, int]]:
        """
        Get occupancy counts for each axis value.
        
        Returns number of entities at each i, j, k value.
        """
        occupancy = {'i': {}, 'j': {}, 'k': {}}
        
        for e in self.engine.entities:
            if e.has_cell():
                i, j, k = e.cell
                occupancy['i'][i] = occupancy['i'].get(i, 0) + 1
                occupancy['j'][j] = occupancy['j'].get(j, 0) + 1
                occupancy['k'][k] = occupancy['k'].get(k, 0) + 1
        
        return occupancy
    
    def compute_level_energies(self, physics: 'MotionPhysics') -> Dict[int, float]:
        """
        Compute average energy at each structural level.
        
        This reveals energy level-like structure from the geometry.
        """
        levels = self.get_structural_levels()
        level_energies = {}
        
        for level, entities in levels.items():
            if entities:
                total_energy = sum(physics.compute_total_energy(e) for e in entities)
                level_energies[level] = total_energy / len(entities)
        
        return level_energies
    
    def compute_occupation_numbers(self) -> Dict[Tuple[int, int, int], int]:
        """
        Get occupation number for each cell (0 or 1 in current model).
        
        Returns cell -> occupation count mapping.
        """
        occupation = {}
        for cell in self.engine.occupied_cells:
            occupation[cell] = 1
        return occupation
    
    def compute_density_of_states(self, energy_bins: int = 10) -> List[Tuple[float, int]]:
        """
        Compute density of states: count of entities in each energy range.
        
        Returns list of (energy_midpoint, count) tuples.
        """
        physics = MotionPhysics(self.engine)
        
        energies = [physics.compute_total_energy(e) for e in self.engine.entities]
        if not energies:
            return []
        
        min_e, max_e = min(energies), max(energies)
        if max_e <= min_e:
            return [(min_e, len(energies))]
        
        bin_width = (max_e - min_e) / energy_bins
        bins = [0] * energy_bins
        
        for e in energies:
            bin_idx = min(int((e - min_e) / bin_width), energy_bins - 1)
            bins[bin_idx] += 1
        
        return [(min_e + (i + 0.5) * bin_width, count) for i, count in enumerate(bins)]


class MotionDynamics:
    """
    Full dynamics layer connecting engine, operators, and physics.
    
    This integrates:
    - Engine's 6 motion functions (structure)
    - n³+t operators (state evolution)
    - Emergent physics (forces, energy, conservation)
    
    Provides the complete evolution: structure + operators + forces.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.physics = MotionPhysics(engine)
        self.quantization = EmergentQuantization(engine)
        self.state_evolution: Optional[StateEvolution] = None
        self.incompatibility: Optional[IncompatibilityTracker] = None
    
    def initialize_tracking(self):
        """Initialize all tracking systems."""
        self.state_evolution = StateEvolution(self.engine)
        self.incompatibility = IncompatibilityTracker(self.engine)
        self.physics.record_state()
    
    def step_with_operators(self, operators: List[ConstraintOperator], dt: float):
        """
        Advance the system with both engine dynamics and operator evolution.
        
        1. Apply constraint operators to state vector
        2. Run engine step (6 motion functions)
        3. Record physics
        4. Track incompatibility
        """
        # Get current state
        state = self.engine.get_state_vector()
        
        # Apply operators
        if self.state_evolution is None:
            self.initialize_tracking()
        
        for i, op in enumerate(operators):
            prev_op = operators[i-1] if i > 0 else None
            self.state_evolution.apply(op, track_incompatibility_with=prev_op)
            
            if prev_op and self.incompatibility:
                self.incompatibility.record_application(op, prev_op, state)
            
            state = op.apply(state, self.engine)
        
        # Run engine step
        self.engine.step(dt)
        
        # Record physics
        self.physics.record_state()
    
    def run(self, steps: int, dt: float, 
            operators: Optional[List[ConstraintOperator]] = None):
        """
        Run full dynamics for multiple steps.
        
        If operators provided, applies them each step.
        Otherwise, just runs engine dynamics.
        """
        self.initialize_tracking()
        
        for _ in range(steps):
            if operators:
                self.step_with_operators(operators, dt)
            else:
                self.engine.step(dt)
                self.physics.record_state()
    
    def get_full_report(self) -> Dict[str, any]:
        """
        Get comprehensive report on system state and dynamics.
        """
        energy = self.physics.compute_system_energy()
        momentum = self.physics.compute_system_momentum()
        conservation = self.physics.get_conservation_report() if len(self.physics.history) >= 2 else {}
        
        levels = self.quantization.get_structural_levels()
        level_energies = self.quantization.compute_level_energies(self.physics)
        
        report = {
            'system': {
                'entities': len(self.engine.entities),
                'occupied_cells': len(self.engine.occupied_cells),
                'unfolding_step': self.engine.unfolding_step,
                'time': self.engine.time,
            },
            'energy': energy,
            'momentum': {
                'x': momentum.x,
                'y': momentum.y,
                'magnitude': momentum.magnitude(),
            },
            'angular_momentum': self.physics.compute_system_angular_momentum(),
            'conservation': conservation,
            'quantization': {
                'levels': {l: len(ents) for l, ents in levels.items()},
                'level_energies': level_energies,
            },
        }
        
        if self.state_evolution:
            report['evolution'] = self.state_evolution.summary()
        
        if self.incompatibility:
            report['incompatibility'] = self.incompatibility.summary()
        
        return report


# =============================================================================
# MOTION ENTITY
# =============================================================================

@dataclass
class MotionEntity:
    """
    An entity that unfolds through the six motion stages.
    
    Properties become meaningful only at their corresponding stage.
    """
    
    # Current stage of unfolding
    stage: Stage = Stage.HEAT
    
    # Stage 0: HEAT - magnitude of presence
    heat: float = 1.0
    
    # Stage 1: POLARITY - differentiation into two opposing ends
    polarity_positive: float = 0.0  # The + end of the line
    polarity_negative: float = 0.0  # The - end of the line
    
    @property
    def net_polarity(self) -> float:
        """Net polarity for interactions: + minus -"""
        return self.polarity_positive - self.polarity_negative
    
    @property
    def polarity_magnitude(self) -> float:
        """Total polarity strength: both ends combined"""
        return self.polarity_positive + self.polarity_negative
    
    # Stage 2: EXISTENCE - persistence, middle defined
    existence: float = 1.0  # 0 = fading, 1 = stable
    
    # Stage 3: RIGHTEOUSNESS - constraint, axes oriented
    righteousness: float = 1.0  # validity of configuration
    orientation: Vector2 = field(default_factory=lambda: Vector2(1, 0))  # axis alignment
    
    # Stage 4: ORDER - enumeration via Robinson arithmetic
    order: int = 0  # Robinson number (0, S(0), S(S(0)), ...)
    
    # n³ closure: cell assignment in structural volume V
    # Assigned when entity reaches ORDER - three independent indices (i, j, k)
    cell: Optional[Tuple[int, int, int]] = None
    
    # Stage 5: MOVEMENT - directed motion (only valid at final stage)
    movement: Vector2 = field(default_factory=Vector2)
    position: Vector2 = field(default_factory=Vector2)
    
    # Forces (accumulated during processing)
    polar_force: Vector2 = field(default_factory=Vector2)
    order_force: Vector2 = field(default_factory=Vector2)
    last_force_clamped: float = 0.0
    
    # Identity
    id: int = 0
    bonds: List[int] = field(default_factory=list)
    
    # Unfolding progress (accumulates until threshold)
    unfold_progress: float = 0.0
    
    
    def is_heat_dead(self, heat_threshold: float) -> bool:
        """Death Type 1: Heat death (presence collapses). Applies at all stages."""
        return self.heat <= heat_threshold

    def is_existence_dead(self, existence_threshold: float) -> bool:
        """Death Type 2: Existence death (persistence collapses). Applies only once existence exists."""
        return self.has_existence() and self.existence <= existence_threshold

    def is_alive(self, heat_threshold: float = 0.01, existence_threshold: float = 0.01) -> bool:
        """
        An entity has exactly two deaths:
          1) Heat death: heat <= heat_threshold  (always applicable)
          2) Existence death: stage >= EXISTENCE and existence <= existence_threshold

        Polarity is never a death condition. It may shape interactions, but it cannot erase the entity.
        """
        if self.is_heat_dead(heat_threshold):
            return False
        if self.is_existence_dead(existence_threshold):
            return False
        return True
    
    def has_polarity(self) -> bool:
        return self.stage >= Stage.POLARITY
    
    def has_existence(self) -> bool:
        return self.stage >= Stage.EXISTENCE
    
    def has_righteousness(self) -> bool:
        return self.stage >= Stage.RIGHTEOUSNESS
    
    def has_order(self) -> bool:
        return self.stage >= Stage.ORDER
    
    def has_movement(self) -> bool:
        return self.stage >= Stage.MOVEMENT
    
    def has_cell(self) -> bool:
        """Whether entity has been assigned a cell in structural volume V."""
        return self.cell is not None


# =============================================================================
# SPATIAL HASH
# =============================================================================

class SpatialHash:
    """Grid-based spatial partitioning for O(1) neighbor queries."""
    
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: Dict[Tuple[int, int], List[MotionEntity]] = {}
    
    def clear(self):
        self.cells.clear()
    
    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def insert(self, entity: MotionEntity):
        key = self._key(entity.position.x, entity.position.y)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append(entity)
    
    def query_radius(self, pos: Vector2, radius: float) -> List[MotionEntity]:
        results = []
        radius_sq = radius * radius
        cell_radius = int(math.ceil(radius / self.cell_size))
        
        cx, cy = self._key(pos.x, pos.y)
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell:
                    for e in cell:
                        if pos.distance_sq(e.position) <= radius_sq:
                            results.append(e)
        return results
    
    def get_all_pairs(self, radius: float) -> List[Tuple[MotionEntity, MotionEntity]]:
        pairs = []
        radius_sq = radius * radius
        cell_radius = int(math.ceil(radius / self.cell_size))
        
        for (cx, cy), cell in self.cells.items():
            for i, e1 in enumerate(cell):
                for e2 in cell[i+1:]:
                    if e1.position.distance_sq(e2.position) <= radius_sq:
                        pairs.append((e1, e2))
            
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if dx < 0 or (dx == 0 and dy < 0):
                        continue
                    neighbor = self.cells.get((cx + dx, cy + dy))
                    if neighbor:
                        for e1 in cell:
                            for e2 in neighbor:
                                if e1.position.distance_sq(e2.position) <= radius_sq:
                                    pairs.append((e1, e2))
        return pairs


# =============================================================================
# MOTION ENGINE
# =============================================================================

class MotionEngine:
    """
    Core engine that processes the six motion functions sequentially.
    
    Each function depends on prior functions being established.
    Entities unfold through stages over time.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.entities: List[MotionEntity] = []
        self.time: float = 0.0
        
        self._next_id: int = 0
        self._next_order: int = 0  # Robinson enumeration counter
        self.by_id: Dict[int, MotionEntity] = {}
        
        self.params = {
            # Unfolding
            'unfold_rate': 0.5,
            'unfold_threshold': 1.0,
            
            # Death thresholds (exactly two deaths)
            'heat_death_threshold': 0.01,
            'existence_death_threshold': 0.01,
            
            # Existence-death heat recycling: when persistence collapses, remaining heat returns to Stage.HEAT
            'recycle_heat_on_existence_death': True,
            
            # Heat
            'heat_transfer_rate': 0.1,
            'heat_dissipation': 0.001,
            'heat_conserve': False,
            'max_heat_transfer': 0.5,
            'heat_range': 150.0,
            
            # Polarity
            'polarity_attraction': 50.0,
            'polarity_repulsion': 30.0,
            'polarity_range': 150.0,
            
            # Existence
            'existence_decay': 0.002,
            'existence_reinforcement': 0.01,
            
            # Righteousness
            'crowd_softcap': 10.0,
            'clamp_penalty_weight': 0.5,
            
            # Order
            'bond_distance': 80.0,
            'order_strength': 20.0,
            
            # Movement
            'max_speed': 200.0,
            'max_force': 500.0,
            'damping': 0.98,
            'max_substep_dt': 0.02,
        }
        
        self.total_heat: float = 0.0
        self.spatial = SpatialHash(self.params['polarity_range'])
        
        # n³+t dimensional closure
        # V = structural volume, cells indexed by (i, j, k)
        # t = unfolding step (not continuous time, but evaluation index)
        self.occupied_cells: Dict[Tuple[int, int, int], int] = {}  # cell -> entity id
        self.unfolding_step: int = 0  # t in n³+t
    
    def spawn(self, x=None, y=None, heat=None) -> MotionEntity:
        """
        Spawn a new entity at Stage.HEAT.
        
        It begins as pure presence - a dot in darkness.
        All other properties emerge through unfolding.
        """
        entity = MotionEntity(
            stage=Stage.HEAT,
            heat=heat if heat is not None else random.uniform(0.5, 2.0),
            position=Vector2(
                x if x is not None else random.uniform(0, self.width),
                y if y is not None else random.uniform(0, self.height)
            ),
            id=self._next_id
        )
        self._next_id += 1
        self.entities.append(entity)
        self.by_id[entity.id] = entity
        self.total_heat += entity.heat
        return entity
    
    def _rebuild_spatial(self):
        self.spatial.clear()
        self.by_id.clear()
        for e in self.entities:
            self.spatial.insert(e)
            self.by_id[e.id] = e
    
    # =========================================================================
    # n³ STRUCTURAL VOLUME
    # =========================================================================
    
    def _get_adjacent_cells(self, cell: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get cells adjacent to the given cell via successor/predecessor on each axis.
        
        Adjacency is purely combinatorial (Robinson successor relations),
        not metric distance.
        """
        i, j, k = cell
        adjacent = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    # Structural volume uses non-negative indices only
                    if ni >= 0 and nj >= 0 and nk >= 0:
                        adjacent.append((ni, nj, nk))
        return adjacent
    
    def _find_available_cell(self, entity: MotionEntity) -> Tuple[int, int, int]:
        """
        Find an available cell in structural volume V for the entity.
        
        Strategy:
        1. If entity has bonded neighbors with cells, prefer adjacent to them
        2. Otherwise, find cell adjacent to existing structure
        3. If volume is empty, assign origin (0, 0, 0)
        """
        # First entity gets the origin
        if not self.occupied_cells:
            return (0, 0, 0)
        
        # Check if bonded neighbors have cells - prefer adjacency to bonds
        candidate_cells = set()
        for bond_id in entity.bonds:
            bonded = self.by_id.get(bond_id)
            if bonded and bonded.has_cell():
                for adj in self._get_adjacent_cells(bonded.cell):
                    if adj not in self.occupied_cells:
                        candidate_cells.add(adj)
        
        if candidate_cells:
            # Pick the candidate closest to the centroid of bonded neighbors
            bonded_cells = [self.by_id[bid].cell for bid in entity.bonds 
                           if bid in self.by_id and self.by_id[bid].has_cell()]
            if bonded_cells:
                ci = sum(c[0] for c in bonded_cells) / len(bonded_cells)
                cj = sum(c[1] for c in bonded_cells) / len(bonded_cells)
                ck = sum(c[2] for c in bonded_cells) / len(bonded_cells)
                return min(candidate_cells, 
                          key=lambda c: (c[0]-ci)**2 + (c[1]-cj)**2 + (c[2]-ck)**2)
            return candidate_cells.pop()
        
        # No bonded neighbors with cells - find any adjacent to existing structure
        for occupied in self.occupied_cells:
            for adj in self._get_adjacent_cells(occupied):
                if adj not in self.occupied_cells:
                    return adj
        
        # Fallback: expand the volume (find min unoccupied on boundary)
        max_coord = max(max(c) for c in self.occupied_cells) + 1
        return (max_coord, 0, 0)
    
    def _assign_cell(self, entity: MotionEntity):
        """
        Assign a cell in structural volume V to the entity.
        
        Called when entity advances to ORDER stage.
        """
        cell = self._find_available_cell(entity)
        entity.cell = cell
        self.occupied_cells[cell] = entity.id
    
    def _release_cell(self, entity: MotionEntity):
        """Release entity's cell back to the volume (on death)."""
        if entity.cell is not None and entity.cell in self.occupied_cells:
            del self.occupied_cells[entity.cell]
    
    def get_structural_state(self) -> Dict[Tuple[int, int, int], int]:
        """
        Return current state of structural volume V.
        
        State(t) ⊆ V — mapping of occupied cells to entity ids.
        """
        return dict(self.occupied_cells)
    
    # -------------------------------------------------------------------------
    # Phase 2: Adjacency queries and structural relations within V
    # -------------------------------------------------------------------------
    
    def get_structural_neighbors(self, entity: MotionEntity) -> List[MotionEntity]:
        """
        Get entities that are structurally adjacent in V.
        
        This is adjacency via successor/predecessor relations on the three
        independent order indices (i, j, k) — NOT spatial proximity.
        """
        if not entity.has_cell():
            return []
        
        neighbors = []
        for adj_cell in self._get_adjacent_cells(entity.cell):
            if adj_cell in self.occupied_cells:
                neighbor_id = self.occupied_cells[adj_cell]
                if neighbor_id in self.by_id:
                    neighbors.append(self.by_id[neighbor_id])
        return neighbors
    
    def get_axis_neighbors(self, entity: MotionEntity, axis: int) -> Tuple[Optional[MotionEntity], Optional[MotionEntity]]:
        """
        Get the predecessor and successor neighbors along a single axis.
        
        Args:
            entity: The entity to query from
            axis: 0 for i-axis, 1 for j-axis, 2 for k-axis
            
        Returns:
            (predecessor, successor) — either may be None if cell unoccupied
        """
        if not entity.has_cell():
            return (None, None)
        
        cell = list(entity.cell)
        
        # Predecessor (subtract 1 on axis)
        pred = None
        if cell[axis] > 0:
            pred_cell = tuple(cell[a] - (1 if a == axis else 0) for a in range(3))
            if pred_cell in self.occupied_cells:
                pred_id = self.occupied_cells[pred_cell]
                pred = self.by_id.get(pred_id)
        
        # Successor (add 1 on axis)
        succ_cell = tuple(cell[a] + (1 if a == axis else 0) for a in range(3))
        succ = None
        if succ_cell in self.occupied_cells:
            succ_id = self.occupied_cells[succ_cell]
            succ = self.by_id.get(succ_id)
        
        return (pred, succ)
    
    def structural_distance(self, cell1: Tuple[int, int, int], cell2: Tuple[int, int, int]) -> int:
        """
        Compute combinatorial distance between two cells in V.
        
        This is the Chebyshev distance (maximum absolute difference along any axis),
        which equals the minimum number of adjacency steps to travel between cells.
        
        This is structural distance, NOT metric distance.
        """
        return max(abs(cell1[0] - cell2[0]), 
                   abs(cell1[1] - cell2[1]), 
                   abs(cell1[2] - cell2[2]))
    
    def structural_manhattan(self, cell1: Tuple[int, int, int], cell2: Tuple[int, int, int]) -> int:
        """
        Compute Manhattan distance in V — sum of absolute differences.
        
        Useful for measuring "total displacement" across all axes.
        """
        return (abs(cell1[0] - cell2[0]) + 
                abs(cell1[1] - cell2[1]) + 
                abs(cell1[2] - cell2[2]))
    
    def are_structurally_adjacent(self, e1: MotionEntity, e2: MotionEntity) -> bool:
        """Check if two entities are adjacent in structural volume V."""
        if not (e1.has_cell() and e2.has_cell()):
            return False
        return self.structural_distance(e1.cell, e2.cell) == 1
    
    def get_shared_axes(self, cell1: Tuple[int, int, int], cell2: Tuple[int, int, int]) -> List[int]:
        """
        Return which axes have the same index value between two cells.
        
        If cells share 2 axes, they differ along exactly 1 axis (axial neighbors).
        If cells share 1 axis, they differ along 2 axes (planar diagonal).
        If cells share 0 axes, they differ along all 3 (space diagonal).
        """
        shared = []
        if cell1[0] == cell2[0]:
            shared.append(0)  # i-axis shared
        if cell1[1] == cell2[1]:
            shared.append(1)  # j-axis shared
        if cell1[2] == cell2[2]:
            shared.append(2)  # k-axis shared
        return shared
    
    def get_axis_slice(self, axis: int, value: int) -> List[MotionEntity]:
        """
        Get all entities in a slice of V where one axis has a fixed value.
        
        For example, get_axis_slice(0, 2) returns all entities with i=2.
        This gives a 2D "plane" within the 3D structural volume.
        """
        entities = []
        for cell, eid in self.occupied_cells.items():
            if cell[axis] == value:
                if eid in self.by_id:
                    entities.append(self.by_id[eid])
        return entities
    
    def get_axis_line(self, fixed_axes: Dict[int, int]) -> List[MotionEntity]:
        """
        Get all entities along a line in V where two axes are fixed.
        
        Args:
            fixed_axes: Dict mapping axis index to fixed value.
                        e.g., {0: 1, 1: 2} gives line at i=1, j=2 (varying k)
        
        Returns:
            List of entities along that line, sorted by the varying axis.
        """
        if len(fixed_axes) != 2:
            return []
        
        # Find the varying axis
        varying_axis = [a for a in range(3) if a not in fixed_axes][0]
        
        entities = []
        for cell, eid in self.occupied_cells.items():
            match = all(cell[ax] == val for ax, val in fixed_axes.items())
            if match and eid in self.by_id:
                entities.append((cell[varying_axis], self.by_id[eid]))
        
        # Sort by position on varying axis
        entities.sort(key=lambda x: x[0])
        return [e for _, e in entities]
    
    def get_structural_neighborhood(self, entity: MotionEntity, radius: int = 1) -> List[MotionEntity]:
        """
        Get all entities within a given structural distance.
        
        radius=1 gives immediate neighbors (same as get_structural_neighbors).
        radius=2 gives neighbors and neighbors-of-neighbors, etc.
        """
        if not entity.has_cell():
            return []
        
        ci, cj, ck = entity.cell
        neighbors = []
        
        for cell, eid in self.occupied_cells.items():
            if cell == entity.cell:
                continue
            if self.structural_distance(entity.cell, cell) <= radius:
                if eid in self.by_id:
                    neighbors.append(self.by_id[eid])
        
        return neighbors
    
    def get_boundary_cells(self) -> List[Tuple[int, int, int]]:
        """
        Get all occupied cells that have at least one unoccupied adjacent cell.
        
        These are the "surface" of the current structural volume.
        """
        boundary = []
        for cell in self.occupied_cells:
            for adj in self._get_adjacent_cells(cell):
                if adj not in self.occupied_cells:
                    boundary.append(cell)
                    break
        return boundary
    
    def get_interior_cells(self) -> List[Tuple[int, int, int]]:
        """
        Get all occupied cells that are completely surrounded by other occupied cells.
        
        These have an "inside" — fully interior to the structure.
        """
        interior = []
        for cell in self.occupied_cells:
            all_neighbors_occupied = True
            for adj in self._get_adjacent_cells(cell):
                if adj not in self.occupied_cells:
                    all_neighbors_occupied = False
                    break
            if all_neighbors_occupied:
                interior.append(cell)
        return interior
    
    # -------------------------------------------------------------------------
    # Phase 3: Constraint operators on structural volume V
    # -------------------------------------------------------------------------
    
    def get_state_vector(self) -> Dict[Tuple[int, int, int], float]:
        """
        Build the state vector |ψ(t)⟩ over structural volume V.
        
        The state vector represents the distribution of "presence" across
        occupied cells. Initial amplitudes are based on heat (magnitude).
        
        Returns:
            Dict mapping cell -> amplitude (normalized so sum of squares = 1)
        """
        if not self.occupied_cells:
            return {}
        
        # Build unnormalized state from heat values
        state = {}
        for cell, eid in self.occupied_cells.items():
            entity = self.by_id.get(eid)
            if entity:
                state[cell] = entity.heat
            else:
                state[cell] = 1.0
        
        # Normalize (sum of squares = 1)
        norm_sq = sum(v * v for v in state.values())
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            state = {cell: amp / norm for cell, amp in state.items()}
        
        return state
    
    def apply_operator(self, operator: 'ConstraintOperator', 
                       state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Dict[Tuple[int, int, int], float]:
        """
        Apply a constraint operator to a state vector.
        
        Args:
            operator: The constraint operator to apply
            state: Optional state vector; if None, uses current state from get_state_vector()
            
        Returns:
            New state vector after operator application
        """
        if state is None:
            state = self.get_state_vector()
        return operator.apply(state, self)
    
    def apply_operator_sequence(self, operators: List['ConstraintOperator'],
                                state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Dict[Tuple[int, int, int], float]:
        """
        Apply a sequence of constraint operators to a state vector.
        
        Operators are applied left-to-right: [A, B, C] computes C(B(A(state)))
        
        This is the U(t) update: |ψ(t+1)⟩ = U(t)|ψ(t)⟩
        where U(t) is composed from constraint operators.
        """
        if state is None:
            state = self.get_state_vector()
        
        current = state
        for op in operators:
            current = op.apply(current, self)
        return current
    
    def measure_commutator(self, op_a: 'ConstraintOperator', op_b: 'ConstraintOperator',
                           state: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Measure the non-commutation of two operators: ||AB - BA||
        
        Returns a scalar measuring how much the operators fail to commute.
        Zero means they commute; larger values mean stronger incompatibility.
        
        This is the algebraic expression of irreducible incompatibility
        introduced by cube closure.
        """
        if state is None:
            state = self.get_state_vector()
        
        # Compute AB|ψ⟩
        ab_state = op_a.apply(op_b.apply(state, self), self)
        
        # Compute BA|ψ⟩
        ba_state = op_b.apply(op_a.apply(state, self), self)
        
        # Compute ||AB - BA|| (L2 norm of difference)
        all_cells = set(ab_state.keys()) | set(ba_state.keys())
        diff_sq = 0.0
        for cell in all_cells:
            ab_val = ab_state.get(cell, 0.0)
            ba_val = ba_state.get(cell, 0.0)
            diff_sq += (ab_val - ba_val) ** 2
        
        return math.sqrt(diff_sq)
    
    def get_constraint_operators(self) -> Dict[str, 'ConstraintOperator']:
        """
        Get the standard constraint operators derived from motion functions.
        
        Returns dict with keys: 'heat', 'polarity', 'righteousness', 'order'
        These are diagonal operators (scaling only).
        """
        return {
            'heat': HeatConstraint(),
            'polarity': PolarityConstraint(),
            'righteousness': RighteousnessConstraint(),
            'order': OrderConstraint(),
        }
    
    def get_shift_operators(self) -> Dict[str, 'ConstraintOperator']:
        """
        Get the axis shift operators (non-diagonal, cause non-commutation).
        
        These operators transfer amplitude between structurally adjacent cells,
        creating genuine non-commutation: shift_i · shift_j ≠ shift_j · shift_i
        
        Returns dict with keys for each axis direction: 'i+', 'i-', 'j+', 'j-', 'k+', 'k-'
        """
        return {
            'i+': AxisShiftOperator(axis=0, direction=1),
            'i-': AxisShiftOperator(axis=0, direction=-1),
            'j+': AxisShiftOperator(axis=1, direction=1),
            'j-': AxisShiftOperator(axis=1, direction=-1),
            'k+': AxisShiftOperator(axis=2, direction=1),
            'k-': AxisShiftOperator(axis=2, direction=-1),
        }
    
    def get_flow_operators(self) -> Dict[str, 'ConstraintOperator']:
        """
        Get physics-derived flow operators (non-diagonal).
        
        These couple motion function properties to amplitude flow:
        - polar_shift: amplitude flows toward opposite polarities
        - righteous_flow: amplitude flows toward higher righteousness
        """
        return {
            'polar_shift': PolarShiftOperator(),
            'righteous_flow': RighteousnessFlowOperator(),
        }
    
    def get_all_operators(self) -> Dict[str, 'ConstraintOperator']:
        """Get all constraint operators (diagonal + shift + flow)."""
        ops = {}
        ops.update(self.get_constraint_operators())
        ops.update(self.get_shift_operators())
        ops.update(self.get_flow_operators())
        return ops
    
    def get_cell_amplitudes(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> List[Tuple[Tuple[int, int, int], int, float]]:
        """
        Get amplitudes for all cells, sorted by amplitude (descending).
        
        Returns list of (cell, entity_id, amplitude) tuples.
        Useful for inspecting which cells are most "present" in the state.
        """
        if state is None:
            state = self.get_state_vector()
        
        result = []
        for cell, amp in state.items():
            eid = self.occupied_cells.get(cell, -1)
            result.append((cell, eid, amp))
        
        result.sort(key=lambda x: -abs(x[2]))
        return result
    
    def compute_expectation(self, operator: 'ConstraintOperator',
                            state: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ for an operator.
        
        This gives the "average" effect of the constraint over the state.
        """
        if state is None:
            state = self.get_state_vector()
        
        weights = operator.compute_weights(self)
        
        # ⟨ψ|O|ψ⟩ = Σ |ψ_i|² × O_ii (diagonal expectation)
        expectation = 0.0
        for cell, amp in state.items():
            weight = weights.get(cell, 1.0)
            expectation += amp * amp * weight
        
        return expectation
    
    def get_incompatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise non-commutation for all standard constraint operators.
        
        Returns dict mapping (op_name1, op_name2) -> commutator norm.
        Larger values indicate stronger incompatibility between constraints.
        """
        operators = self.get_constraint_operators()
        state = self.get_state_vector()
        
        matrix = {}
        op_names = list(operators.keys())
        
        for i, name_a in enumerate(op_names):
            for name_b in op_names[i+1:]:
                commutator = self.measure_commutator(
                    operators[name_a], 
                    operators[name_b],
                    state
                )
                matrix[(name_a, name_b)] = commutator
                matrix[(name_b, name_a)] = commutator  # Symmetric
        
        return matrix
    
    # -------------------------------------------------------------------------
    # Phase 4: State evolution, uncertainty, and incompatibility tracking
    # -------------------------------------------------------------------------
    
    def create_state_evolution(self) -> 'StateEvolution':
        """
        Create a new StateEvolution tracker for this engine.
        
        Use this to track state changes as operators are applied:
        
            evolution = engine.create_state_evolution()
            evolution.apply(shift_i)
            evolution.apply(shift_j)
            print(evolution.summary())
        """
        return StateEvolution(self)
    
    def create_incompatibility_tracker(self) -> 'IncompatibilityTracker':
        """
        Create a new IncompatibilityTracker for this engine.
        
        Use this to track accumulated incompatibility across cells:
        
            tracker = engine.create_incompatibility_tracker()
            tracker.record_application(op_a, op_b)
            print(tracker.get_hotspots())
        """
        return IncompatibilityTracker(self)
    
    def compute_uncertainty_relation(self, op_a: 'ConstraintOperator', op_b: 'ConstraintOperator',
                                     state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Dict[str, float]:
        """
        Compute uncertainty relation between two operators.
        
        For non-commuting operators: ΔA × ΔB ≥ ½|⟨[A,B]⟩|
        
        Returns dict with variances, uncertainties, product, and bound.
        """
        relation = UncertaintyRelation(self, op_a, op_b)
        return relation.compute_uncertainty_product(state)
    
    def evolve_state(self, operators: List['ConstraintOperator'], 
                     initial_state: Optional[Dict[Tuple[int, int, int], float]] = None) -> 'StateEvolution':
        """
        Evolve state through a sequence of operators, tracking history.
        
        This is the primary method for n³+t dynamics:
        |ψ(t+n)⟩ = U_n · U_{n-1} · ... · U_1 |ψ(t)⟩
        
        Returns the StateEvolution object with full history.
        """
        evolution = StateEvolution(self)
        
        # Override initial state if provided
        if initial_state is not None:
            evolution.history[0] = StateSnapshot(
                step=self.unfolding_step,
                state=initial_state,
                operator_applied=None
            )
        
        evolution.apply_sequence(operators)
        return evolution
    
    def measure_total_incompatibility(self, operators: List['ConstraintOperator'],
                                      state: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Measure total incompatibility from applying a sequence of operators.
        
        Sums ||[A_i, A_{i+1}]|| for consecutive pairs in the sequence.
        """
        if state is None:
            state = self.get_state_vector()
        
        total = 0.0
        for i in range(len(operators) - 1):
            total += self.measure_commutator(operators[i], operators[i+1], state)
        return total
    
    def find_minimum_incompatibility_order(self, operators: List['ConstraintOperator'],
                                           state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Tuple[List['ConstraintOperator'], float]:
        """
        Find the ordering of operators that minimizes total incompatibility.
        
        For small numbers of operators, tests all permutations.
        Returns (best_ordering, min_incompatibility).
        
        Note: This is computationally expensive for >6 operators.
        """
        from itertools import permutations
        
        if state is None:
            state = self.get_state_vector()
        
        if len(operators) > 8:
            # Too many permutations - return original order
            return operators, self.measure_total_incompatibility(operators, state)
        
        best_order = operators
        best_incomp = float('inf')
        
        for perm in permutations(operators):
            incomp = self.measure_total_incompatibility(list(perm), state)
            if incomp < best_incomp:
                best_incomp = incomp
                best_order = list(perm)
        
        return best_order, best_incomp
    
    def compute_state_entropy(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Compute Shannon entropy of the state distribution.
        
        Higher entropy means more spread across cells.
        Lower entropy means more concentrated (localized).
        """
        if state is None:
            state = self.get_state_vector()
        
        if not state:
            return 0.0
        
        # Convert amplitudes to probabilities (|ψ|²)
        probs = [a * a for a in state.values()]
        total = sum(probs)
        if total <= 0:
            return 0.0
        
        probs = [p / total for p in probs]
        
        # Shannon entropy: -Σ p_i log(p_i)
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        return entropy
    
    def compute_localization(self, state: Optional[Dict[Tuple[int, int, int], float]] = None) -> Dict[str, float]:
        """
        Compute localization measures for the state.
        
        Returns:
            - participation_ratio: effective number of occupied cells (1 = fully localized)
            - max_probability: highest single-cell probability
            - entropy: Shannon entropy
            - spread: standard deviation of position
        """
        if state is None:
            state = self.get_state_vector()
        
        if not state:
            return {'participation_ratio': 0, 'max_probability': 0, 'entropy': 0, 'spread': 0}
        
        probs = {cell: a * a for cell, a in state.items()}
        total = sum(probs.values())
        if total <= 0:
            return {'participation_ratio': 0, 'max_probability': 0, 'entropy': 0, 'spread': 0}
        
        probs = {cell: p / total for cell, p in probs.items()}
        
        # Participation ratio: 1 / Σ p_i²
        sum_p_sq = sum(p * p for p in probs.values())
        participation = 1.0 / sum_p_sq if sum_p_sq > 0 else 0
        
        # Max probability
        max_prob = max(probs.values())
        
        # Entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)
        
        # Spread (standard deviation of position)
        mean_i = sum(cell[0] * p for cell, p in probs.items())
        mean_j = sum(cell[1] * p for cell, p in probs.items())
        mean_k = sum(cell[2] * p for cell, p in probs.items())
        
        var = sum(
            ((cell[0] - mean_i)**2 + (cell[1] - mean_j)**2 + (cell[2] - mean_k)**2) * p
            for cell, p in probs.items()
        )
        spread = math.sqrt(var)
        
        return {
            'participation_ratio': participation,
            'max_probability': max_prob,
            'entropy': entropy,
            'spread': spread,
        }
    
    # -------------------------------------------------------------------------
    # Phase 5: Emergent physics interface
    # -------------------------------------------------------------------------
    
    def create_physics(self) -> 'MotionPhysics':
        """
        Create a MotionPhysics instance for this engine.
        
        Use for computing energy, momentum, and conservation laws.
        """
        return MotionPhysics(self)
    
    def create_dynamics(self) -> 'MotionDynamics':
        """
        Create a full MotionDynamics instance for this engine.
        
        Integrates engine, operators, and physics for complete evolution.
        """
        return MotionDynamics(self)
    
    def create_quantization(self) -> 'EmergentQuantization':
        """
        Create an EmergentQuantization instance for this engine.
        
        Use for analyzing discrete structure and energy levels.
        """
        return EmergentQuantization(self)
    
    def compute_system_energy(self) -> Dict[str, float]:
        """
        Compute total energy of the system (convenience method).
        
        Returns breakdown: kinetic, potential, polar, structural, total.
        """
        physics = MotionPhysics(self)
        return physics.compute_system_energy()
    
    def compute_system_momentum(self) -> Tuple[float, float, float]:
        """
        Compute total momentum of the system (convenience method).
        
        Returns (px, py, magnitude).
        """
        physics = MotionPhysics(self)
        p = physics.compute_system_momentum()
        return (p.x, p.y, p.magnitude())
    
    def get_physics_snapshot(self) -> Dict[str, any]:
        """
        Get a complete physics snapshot of the current state.
        
        Useful for diagnostics and tracking.
        """
        physics = MotionPhysics(self)
        quant = EmergentQuantization(self)
        
        energy = physics.compute_system_energy()
        momentum = physics.compute_system_momentum()
        levels = quant.get_structural_levels()
        
        return {
            'energy': energy,
            'momentum': {'x': momentum.x, 'y': momentum.y, 'magnitude': momentum.magnitude()},
            'angular_momentum': physics.compute_system_angular_momentum(),
            'structural_levels': {l: len(ents) for l, ents in levels.items()},
            'total_heat': self.total_heat,
            'entity_count': len(self.entities),
        }
    
    # =========================================================================
    # STAGE 0: HEAT - Magnitude of Motion
    # =========================================================================
    
    def process_heat(self, dt):
        """
        HEAT: The first function. Pure presence, magnitude only.
        
        A dot appears in darkness. You know it has motion because 
        it is present, but no idea what motion it consists of.
        
        Heat transfers between nearby entities.
        """
        heat_range = self.params['heat_range']
        pairs = self.spatial.get_all_pairs(heat_range)
        max_transfer = self.params['max_heat_transfer']
        
        for e1, e2 in pairs:
            dist = e1.position.distance_to(e2.position)
            if dist > 0:
                falloff = 1 - dist / heat_range
                
                # If both have polarity, interaction is stronger
                if e1.has_polarity() and e2.has_polarity():
                    interaction = falloff * (1.0 + e1.polarity_magnitude * e2.polarity_magnitude)
                else:
                    interaction = falloff
                
                transfer = (e1.heat - e2.heat) * self.params['heat_transfer_rate'] * dt * interaction
                transfer = max(-max_transfer, min(max_transfer, transfer))
                e1.heat -= transfer
                e2.heat += transfer
        
        if not self.params['heat_conserve']:
            for e in self.entities:
                e.heat = max(0, e.heat - e.heat * self.params['heat_dissipation'] * dt)
    
    # =========================================================================
    # STAGE 1: POLARITY - Differentiation of Motion
    # =========================================================================
    
    def process_polarity(self, dt):
        """
        POLARITY: The dot becomes a line with 2 opposing ends.
        
        A positive and a negative emerge from undifferentiated heat.
        Only entities at Stage.POLARITY or beyond participate.
        """
        for e in self.entities:
            if e.has_polarity():
                e.polar_force = Vector2()
        
        pairs = self.spatial.get_all_pairs(self.params['polarity_range'])
        
        for e1, e2 in pairs:
            if not (e1.has_polarity() and e2.has_polarity()):
                continue
            
            delta = e2.position - e1.position
            dist = max(1, delta.magnitude())
            
            direction = delta.normalized()
            strength = (1 - dist / self.params['polarity_range']) ** 2
            product = e1.net_polarity * e2.net_polarity
            
            if product < 0:
                force_mag = -product * self.params['polarity_attraction'] * strength
                force = direction * force_mag
            else:
                force_mag = product * self.params['polarity_repulsion'] * strength
                force = direction * (-force_mag)
            
            e1.polar_force = e1.polar_force + force
            e2.polar_force = e2.polar_force - force
    
    # =========================================================================
    # STAGE 2: EXISTENCE - Persistence of Motion
    # =========================================================================
    
    def process_existence(self, dt):
        """
        EXISTENCE: A middle appears at 0.
        
        The polarity line gains a center - persistence is defined.
        Existence decays naturally, reinforced by righteousness.
        """
        for e in self.entities:
            if not e.has_existence():
                continue
            
            e.existence -= self.params['existence_decay'] * dt
            
            if e.has_righteousness():
                e.existence += e.righteousness * self.params['existence_reinforcement'] * dt
            
            e.existence = max(0, min(2.0, e.existence))
        
        
        heat_thr = self.params['heat_death_threshold']
        exist_thr = self.params['existence_death_threshold']
        recycle = self.params.get('recycle_heat_on_existence_death', True)

        alive: List[MotionEntity] = []
        heat_deaths: List[MotionEntity] = []
        existence_deaths: List[MotionEntity] = []

        for e in self.entities:
            if e.is_heat_dead(heat_thr):
                heat_deaths.append(e)
            elif e.is_existence_dead(exist_thr):
                existence_deaths.append(e)
            else:
                alive.append(e)

        if heat_deaths or existence_deaths:
            # Release cells in structural volume V before removing entities
            for e in heat_deaths:
                self._release_cell(e)
            for e in existence_deaths:
                self._release_cell(e)
            
            # Remove the dead entities
            removed_heat = sum(e.heat for e in heat_deaths)
            if not recycle:
                removed_heat += sum(e.heat for e in existence_deaths)

            self.total_heat -= removed_heat
            self.entities = alive
            self._rebuild_spatial()

            # If existence collapses but heat remains, return that heat to the universe as Stage.HEAT entities.
            if recycle:
                for e in existence_deaths:
                    # Ensure the recycled entity is not immediately heat-dead
                    recycled_heat = max(e.heat, heat_thr * 2.0)
                    self.spawn(x=e.position.x, y=e.position.y, heat=recycled_heat)
    
    # =========================================================================
    # STAGE 3: RIGHTEOUSNESS - Constraint of Motion
    # =========================================================================
    
    def process_righteousness(self, dt):
        """
        RIGHTEOUSNESS: The middle defines axes.
        
        Upon becoming, the becoming will orient along 2 axes of x and y
        with a positive and negative connotation.
        """
        for e in self.entities:
            if not e.has_righteousness():
                continue
            
            neighbors = self.spatial.query_radius(e.position, self.params['polarity_range'])
            neighbors = [n for n in neighbors if n.id != e.id and n.has_righteousness()]
            
            if neighbors:
                polar_neighbors = [n for n in neighbors if n.has_polarity()]
                if polar_neighbors:
                    polarity_balance = 1 - abs(sum(n.net_polarity for n in polar_neighbors) / len(polar_neighbors))
                else:
                    polarity_balance = 0.5
                
                heat_variance = sum((e.heat - n.heat)**2 for n in neighbors) / len(neighbors)
                heat_harmony = 1 / (1 + heat_variance)
                base = 0.5 * polarity_balance + 0.5 * heat_harmony
            else:
                base = 0.5
            
            crowd_softcap = self.params['crowd_softcap']
            crowd = len(neighbors)
            crowd_penalty = 1.0 / (1.0 + max(0.0, crowd - crowd_softcap) / crowd_softcap)
            
            clamp_weight = self.params['clamp_penalty_weight']
            clamp_penalty = 1.0 - clamp_weight * max(0.0, min(1.0, e.last_force_clamped))
            
            score = base * crowd_penalty * clamp_penalty
            e.righteousness = 0.9 * e.righteousness + 0.1 * score
            
            if e.has_polarity() and e.polar_force.magnitude() > 0.1:
                e.orientation = e.polar_force.normalized()
    
    # =========================================================================
    # STAGE 4: ORDER - Regulation of Motion
    # =========================================================================
    
    def process_order(self, dt):
        """
        ORDER: Enumeration via Robinson arithmetic.
        
        Upon connotating, the motion commits the function of order
        and labels as defined by Robinson arithmetic - the simplest
        arithmetic that does not unnecessarily define other axioms.
        """
        for e in self.entities:
            if e.has_order():
                e.bonds = []
                e.order_force = Vector2()
        
        bond_dist = self.params['bond_distance']
        ideal = bond_dist * 0.7
        k = self.params['order_strength'] * 0.01
        
        pairs = self.spatial.get_all_pairs(bond_dist)
        
        for e1, e2 in pairs:
            if not (e1.has_order() and e2.has_order()):
                continue
            
            if not (e1.has_polarity() and e2.has_polarity()):
                continue
            
            if e1.net_polarity * e2.net_polarity < -0.3:
                if e1.has_righteousness() and e2.has_righteousness():
                    if e1.righteousness > 0.5 and e2.righteousness > 0.5:
                        e1.bonds.append(e2.id)
                        e2.bonds.append(e1.id)
                        
                        delta = e2.position - e1.position
                        dist = delta.magnitude()
                        if dist > 0:
                            force = delta.normalized() * ((dist - ideal) * k)
                            e1.order_force = e1.order_force + force
                            e2.order_force = e2.order_force - force
    
    # =========================================================================
    # STAGE 5: MOVEMENT - Direction of Motion
    # =========================================================================
    
    def process_movement(self, dt):
        """
        MOVEMENT: The final function. Direction chosen.
        
        After all scalar values are established, the motion may
        choose a vector by choice or by random.
        """
        max_force = self.params['max_force']
        
        for e in self.entities:
            if not e.has_movement():
                continue
            
            force = Vector2()
            
            if e.has_polarity():
                force = force + e.polar_force
            
            if e.has_order():
                force = force + e.order_force
            
            scale = e.heat
            if e.has_existence():
                scale *= e.existence
            if e.has_righteousness():
                scale *= (0.5 + 0.5 * e.righteousness)
            
            force = force * scale
            
            force_mag = force.magnitude()
            if force_mag > max_force:
                e.last_force_clamped = min(1.0, (force_mag - max_force) / max_force)
                force = force.normalized() * max_force
            else:
                e.last_force_clamped = 0.0
            
            e.movement = e.movement * self.params['damping'] + force * dt
            
            speed = e.movement.magnitude()
            if speed > self.params['max_speed']:
                e.movement = e.movement.normalized() * self.params['max_speed']
            
            e.position = e.position + e.movement * dt
            e.position.x = e.position.x % self.width
            e.position.y = e.position.y % self.height
    
    # =========================================================================
    # UNFOLDING
    # =========================================================================
    
    def process_unfolding(self, dt):
        """
        Process the unfolding of entities through motion stages.
        
        Entities accumulate unfold_progress based on their heat
        and advance to the next stage when threshold is reached.
        """
        for e in self.entities:
            if e.stage >= Stage.MOVEMENT:
                continue
            
            e.unfold_progress += e.heat * self.params['unfold_rate'] * dt
            
            if e.unfold_progress >= self.params['unfold_threshold']:
                e.unfold_progress = 0.0
                self._advance_stage(e)
    
    def _advance_stage(self, entity: MotionEntity):
        """Advance entity to the next stage, initializing stage-specific properties."""
        
        if entity.stage == Stage.HEAT:
            # Advancing to POLARITY: the dot becomes a line with two ends
            entity.stage = Stage.POLARITY
            entity.polarity_positive = random.uniform(0.1, 1.0)
            entity.polarity_negative = random.uniform(0.1, 1.0)
        
        elif entity.stage == Stage.POLARITY:
            entity.stage = Stage.EXISTENCE
            entity.existence = 1.0
        
        elif entity.stage == Stage.EXISTENCE:
            entity.stage = Stage.RIGHTEOUSNESS
            entity.righteousness = 1.0
            angle = random.uniform(0, 2 * math.pi)
            entity.orientation = Vector2(math.cos(angle), math.sin(angle))
        
        elif entity.stage == Stage.RIGHTEOUSNESS:
            # Advancing to ORDER: enumeration via Robinson arithmetic
            # AND assignment of cell in structural volume V (n³ closure)
            entity.stage = Stage.ORDER
            entity.order = self._next_order
            self._next_order = Robinson.successor(self._next_order)
            self._assign_cell(entity)
        
        elif entity.stage == Stage.ORDER:
            entity.stage = Stage.MOVEMENT
            if random.random() < 0.5:
                angle = random.uniform(0, 2 * math.pi)
                entity.movement = Vector2(math.cos(angle), math.sin(angle)) * entity.heat
            else:
                entity.movement = entity.orientation * entity.heat
    
    # =========================================================================
    # MAIN STEP
    # =========================================================================
    
    def _substep(self, dt: float):
        self._rebuild_spatial()
        
        self.process_heat(dt)
        self.process_polarity(dt)
        self.process_existence(dt)
        self.process_righteousness(dt)
        self.process_order(dt)
        self.process_movement(dt)
        self.process_unfolding(dt)
        
        self.total_heat = sum(e.heat for e in self.entities)
    
    def step(self, dt: float):
        max_dt = self.params['max_substep_dt']
        
        if dt <= max_dt:
            self._substep(dt)
        else:
            steps = int(math.ceil(dt / max_dt))
            sub_dt = dt / steps
            for _ in range(steps):
                self._substep(sub_dt)
        
        self.time += dt
        self.unfolding_step += 1  # t in n³+t: evaluation index advances
    
    def get_stage_counts(self) -> Dict[Stage, int]:
        counts = {s: 0 for s in Stage}
        for e in self.entities:
            counts[e.stage] += 1
        return counts
    
    def get_volume_stats(self) -> Dict[str, any]:
        """
        Return statistics about the structural volume V.
        
        Useful for inspecting the n³+t dimensional closure state.
        """
        if not self.occupied_cells:
            return {
                'occupied': 0,
                'bounds': None,
                'unfolding_step': self.unfolding_step,
            }
        
        cells = list(self.occupied_cells.keys())
        i_vals = [c[0] for c in cells]
        j_vals = [c[1] for c in cells]
        k_vals = [c[2] for c in cells]
        
        return {
            'occupied': len(self.occupied_cells),
            'bounds': {
                'i': (min(i_vals), max(i_vals)),
                'j': (min(j_vals), max(j_vals)),
                'k': (min(k_vals), max(k_vals)),
            },
            'unfolding_step': self.unfolding_step,
            'entities_with_cells': sum(1 for e in self.entities if e.has_cell()),
        }


# =============================================================================
# FEATURE 1: RENDERING HOOKS
# =============================================================================

class EntityVisual:
    """Visual representation data for a single entity."""
    
    def __init__(self, entity: MotionEntity, physics: Optional['MotionPhysics'] = None):
        self.id = entity.id
        self.position = (entity.position.x, entity.position.y)
        self.stage = entity.stage
        self.stage_name = entity.stage.name
        self.heat = entity.heat
        self.cell = entity.cell
        
        # Stage-dependent properties
        self.polarity = entity.net_polarity if entity.has_polarity() else 0.0
        self.polarity_magnitude = entity.polarity_magnitude if entity.has_polarity() else 0.0
        self.existence = entity.existence if entity.has_existence() else 1.0
        self.righteousness = entity.righteousness if entity.has_righteousness() else 1.0
        self.order = entity.order if entity.has_order() else -1
        
        # Movement
        if entity.has_movement():
            self.velocity = (entity.movement.x, entity.movement.y)
            self.speed = entity.movement.magnitude()
        else:
            self.velocity = (0.0, 0.0)
            self.speed = 0.0
        
        # Orientation
        self.orientation = (entity.orientation.x, entity.orientation.y)
        
        # Bonds
        self.bonds = list(entity.bonds)
        
        # Forces
        self.polar_force = (entity.polar_force.x, entity.polar_force.y)
        self.order_force = (entity.order_force.x, entity.order_force.y)
        
        # Energy (if physics provided)
        if physics:
            self.energy = physics.compute_total_energy(entity)
            self.kinetic = physics.compute_kinetic_energy(entity)
            self.potential = physics.compute_potential_energy(entity)
        else:
            self.energy = self.kinetic = self.potential = 0.0


class RenderSnapshot:
    """Complete visual snapshot of the engine state."""
    
    # Stage colors (RGB tuples, 0-255)
    STAGE_COLORS = {
        Stage.HEAT: (255, 100, 50),          # Orange-red (pure presence)
        Stage.POLARITY: (50, 150, 255),      # Blue (differentiation)
        Stage.EXISTENCE: (50, 255, 150),     # Green (persistence)
        Stage.RIGHTEOUSNESS: (255, 255, 50), # Yellow (constraint)
        Stage.ORDER: (200, 50, 255),         # Purple (enumeration)
        Stage.MOVEMENT: (255, 255, 255),     # White (full expression)
    }
    
    def __init__(self, engine: 'MotionEngine', include_physics: bool = True):
        self.time = engine.time
        self.unfolding_step = engine.unfolding_step
        self.width = engine.width
        self.height = engine.height
        self.total_heat = engine.total_heat
        self.stage_counts = engine.get_stage_counts()
        
        # Create physics if requested
        physics = MotionPhysics(engine) if include_physics else None
        
        # Entity visuals
        self.entities = [EntityVisual(e, physics) for e in engine.entities]
        
        # Build bond pairs for drawing lines
        self.bond_pairs = []
        id_to_pos = {ev.id: ev.position for ev in self.entities}
        for ev in self.entities:
            for bond_id in ev.bonds:
                if bond_id in id_to_pos and ev.id < bond_id:  # Avoid duplicates
                    self.bond_pairs.append((ev.position, id_to_pos[bond_id]))
        
        # Structural volume data
        self.cells = list(engine.occupied_cells.keys())
        self.cell_to_entity = dict(engine.occupied_cells)
        self.volume_bounds = engine.get_volume_stats().get('bounds')
        
        # Build adjacency edges for V visualization
        self.cell_edges = []
        for cell in self.cells:
            i, j, k = cell
            # Only add edges to neighbors with higher indices (avoid duplicates)
            neighbors = [(i+1, j, k), (i, j+1, k), (i, j, k+1)]
            for neighbor in neighbors:
                if neighbor in engine.occupied_cells:
                    self.cell_edges.append((cell, neighbor))
    
    def get_entity_color(self, entity_visual: EntityVisual) -> Tuple[int, int, int]:
        """Get color for entity based on stage."""
        base = self.STAGE_COLORS[entity_visual.stage]
        
        # Modulate by heat (brightness)
        heat_factor = min(1.0, entity_visual.heat / 2.0)
        return tuple(int(c * (0.3 + 0.7 * heat_factor)) for c in base)
    
    def get_entity_radius(self, entity_visual: EntityVisual, base_radius: float = 5.0) -> float:
        """Get radius for entity based on heat."""
        return base_radius * (0.5 + 0.5 * min(2.0, entity_visual.heat))
    
    def get_polarity_color(self, polarity: float) -> Tuple[int, int, int]:
        """Get color representing polarity: red for +, blue for -."""
        if polarity > 0:
            intensity = min(255, int(polarity * 200))
            return (255, 255 - intensity, 255 - intensity)  # Red
        else:
            intensity = min(255, int(-polarity * 200))
            return (255 - intensity, 255 - intensity, 255)  # Blue
    
    def project_cell_to_2d(self, cell: Tuple[int, int, int], 
                          scale: float = 30.0, 
                          offset: Tuple[float, float] = (100, 100)) -> Tuple[float, float]:
        """Project 3D cell to 2D using isometric projection."""
        i, j, k = cell
        # Isometric projection
        x = (i - k) * scale * 0.866 + offset[0]
        y = (i + k) * scale * 0.5 - j * scale + offset[1]
        return (x, y)


class RenderHooks:
    """
    Rendering hooks for the Motion Engine.
    
    Provides callbacks and data extraction for visualization.
    Can be used with Matplotlib, Pygame, or any rendering library.
    """
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.callbacks: List[callable] = []
        self.frame_count = 0
        self.snapshots: List[RenderSnapshot] = []
        self.record_history = False
        self.max_history = 100
    
    def add_callback(self, callback: callable):
        """
        Add a render callback. Called after each step with a RenderSnapshot.
        
        callback signature: callback(snapshot: RenderSnapshot) -> None
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: callable):
        """Remove a render callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def enable_history(self, max_frames: int = 100):
        """Enable snapshot history recording."""
        self.record_history = True
        self.max_history = max_frames
    
    def disable_history(self):
        """Disable snapshot history recording."""
        self.record_history = False
    
    def get_snapshot(self, include_physics: bool = True) -> RenderSnapshot:
        """Get current render snapshot."""
        return RenderSnapshot(self.engine, include_physics)
    
    def on_step(self):
        """Called after engine.step() - triggers callbacks."""
        snapshot = self.get_snapshot()
        self.frame_count += 1
        
        if self.record_history:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_history:
                self.snapshots.pop(0)
        
        for callback in self.callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Render callback error: {e}")
    
    def get_matplotlib_figure(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Generate a Matplotlib figure with entity positions and structural volume.
        
        Returns (fig, axes) tuple. Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("Matplotlib required for this method")
        
        snapshot = self.get_snapshot()
        
        fig = plt.figure(figsize=figsize)
        
        # Left: 2D entity positions
        ax1 = fig.add_subplot(121)
        ax1.set_xlim(0, snapshot.width)
        ax1.set_ylim(0, snapshot.height)
        ax1.set_title(f"Entity Positions (t={snapshot.time:.2f}, step={snapshot.unfolding_step})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        # Draw bonds
        for p1, p2 in snapshot.bond_pairs:
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5)
        
        # Draw entities
        for ev in snapshot.entities:
            color = tuple(c/255 for c in snapshot.get_entity_color(ev))
            radius = snapshot.get_entity_radius(ev)
            ax1.scatter(ev.position[0], ev.position[1], c=[color], s=radius**2, alpha=0.8)
            
            # Draw velocity arrow for moving entities
            if ev.speed > 1.0:
                ax1.arrow(ev.position[0], ev.position[1], 
                         ev.velocity[0]*0.5, ev.velocity[1]*0.5,
                         head_width=3, head_length=2, fc='white', ec='white', alpha=0.5)
        
        # Right: 3D structural volume
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Structural Volume V")
        ax2.set_xlabel("i")
        ax2.set_ylabel("j")
        ax2.set_zlabel("k")
        
        if snapshot.cells:
            # Draw cells
            is_vals = [c[0] for c in snapshot.cells]
            js_vals = [c[1] for c in snapshot.cells]
            ks_vals = [c[2] for c in snapshot.cells]
            
            # Color by entity stage
            colors = []
            for cell in snapshot.cells:
                eid = snapshot.cell_to_entity.get(cell, -1)
                ev = next((e for e in snapshot.entities if e.id == eid), None)
                if ev:
                    colors.append(tuple(c/255 for c in snapshot.get_entity_color(ev)))
                else:
                    colors.append((0.5, 0.5, 0.5))
            
            ax2.scatter(is_vals, js_vals, ks_vals, c=colors, s=100, alpha=0.8)
            
            # Draw edges
            for c1, c2 in snapshot.cell_edges:
                ax2.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], 
                        'gray', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def save_frame(self, filename: str):
        """Save current frame as image. Requires matplotlib."""
        fig, _ = self.get_matplotlib_figure()
        fig.savefig(filename, dpi=100, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# FEATURE 2: PARAMETER TUNING
# =============================================================================

class ParamSpec:
    """Specification for a tunable parameter."""
    
    def __init__(self, name: str, min_val: float, max_val: float, 
                 default: float, description: str = "", step: float = None):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.default = default
        self.description = description
        self.step = step or (max_val - min_val) / 100
    
    def clamp(self, value: float) -> float:
        """Clamp value to valid range."""
        return max(self.min_val, min(self.max_val, value))
    
    def validate(self, value: float) -> bool:
        """Check if value is in valid range."""
        return self.min_val <= value <= self.max_val


class ParamTuner:
    """
    Parameter tuning system for the Motion Engine.
    
    Provides structured access to engine parameters with validation,
    presets, and change tracking.
    """
    
    # Parameter specifications
    PARAM_SPECS = {
        # Unfolding
        'unfold_rate': ParamSpec('unfold_rate', 0.01, 5.0, 0.5, 
                                 "Rate at which entities accumulate unfold progress"),
        'unfold_threshold': ParamSpec('unfold_threshold', 0.1, 10.0, 1.0,
                                      "Progress required to advance stage"),
        
        # Death thresholds
        'heat_death_threshold': ParamSpec('heat_death_threshold', 0.001, 0.5, 0.01,
                                          "Heat below this = death"),
        'existence_death_threshold': ParamSpec('existence_death_threshold', 0.001, 0.5, 0.01,
                                               "Existence below this = death"),
        
        # Heat
        'heat_transfer_rate': ParamSpec('heat_transfer_rate', 0.0, 1.0, 0.1,
                                        "Rate of heat transfer between nearby entities"),
        'heat_dissipation': ParamSpec('heat_dissipation', 0.0, 0.1, 0.001,
                                      "Rate of heat loss to environment"),
        'max_heat_transfer': ParamSpec('max_heat_transfer', 0.01, 2.0, 0.5,
                                       "Maximum heat transfer per step"),
        'heat_range': ParamSpec('heat_range', 10.0, 500.0, 150.0,
                                "Range for heat interactions"),
        
        # Polarity
        'polarity_attraction': ParamSpec('polarity_attraction', 0.0, 200.0, 50.0,
                                         "Strength of opposite-polarity attraction"),
        'polarity_repulsion': ParamSpec('polarity_repulsion', 0.0, 200.0, 30.0,
                                        "Strength of same-polarity repulsion"),
        'polarity_range': ParamSpec('polarity_range', 10.0, 500.0, 150.0,
                                    "Range for polarity interactions"),
        
        # Existence
        'existence_decay': ParamSpec('existence_decay', 0.0, 0.1, 0.002,
                                     "Rate of existence decay"),
        'existence_reinforcement': ParamSpec('existence_reinforcement', 0.0, 0.1, 0.01,
                                             "Rate of existence reinforcement from righteousness"),
        
        # Righteousness
        'crowd_softcap': ParamSpec('crowd_softcap', 1.0, 50.0, 10.0,
                                   "Neighbor count before crowding penalty"),
        'clamp_penalty_weight': ParamSpec('clamp_penalty_weight', 0.0, 1.0, 0.5,
                                          "Weight of force-clamp penalty on righteousness"),
        
        # Order
        'bond_distance': ParamSpec('bond_distance', 10.0, 200.0, 80.0,
                                   "Distance for bond formation"),
        'order_strength': ParamSpec('order_strength', 0.0, 100.0, 20.0,
                                    "Strength of order-based forces"),
        
        # Movement
        'max_speed': ParamSpec('max_speed', 10.0, 1000.0, 200.0,
                               "Maximum entity speed"),
        'max_force': ParamSpec('max_force', 10.0, 2000.0, 500.0,
                               "Maximum force magnitude"),
        'damping': ParamSpec('damping', 0.8, 1.0, 0.98,
                             "Velocity damping factor"),
    }
    
    # Preset configurations
    PRESETS = {
        'default': {},  # Uses engine defaults
        'stable_universe': {
            'heat_dissipation': 0.0001,
            'existence_decay': 0.0005,
            'existence_reinforcement': 0.02,
            'damping': 0.99,
        },
        'chaotic': {
            'heat_transfer_rate': 0.3,
            'polarity_attraction': 100.0,
            'polarity_repulsion': 80.0,
            'max_force': 1000.0,
            'damping': 0.95,
        },
        'slow_unfold': {
            'unfold_rate': 0.1,
            'unfold_threshold': 2.0,
            'heat_dissipation': 0.0,
        },
        'fast_death': {
            'heat_death_threshold': 0.1,
            'existence_death_threshold': 0.1,
            'existence_decay': 0.01,
            'heat_dissipation': 0.01,
        },
        'dense_bonds': {
            'bond_distance': 150.0,
            'order_strength': 50.0,
            'polarity_attraction': 80.0,
        },
        'conserved': {
            'heat_conserve': True,
            'heat_dissipation': 0.0,
        },
    }
    
    def __init__(self, engine: 'MotionEngine'):
        self.engine = engine
        self.change_history: List[Tuple[str, float, float]] = []  # (param, old, new)
        self.callbacks: List[callable] = []
    
    def get(self, name: str) -> float:
        """Get current parameter value."""
        return self.engine.params.get(name, self.PARAM_SPECS.get(name, ParamSpec(name, 0, 1, 0)).default)
    
    def set(self, name: str, value: float, validate: bool = True) -> bool:
        """
        Set parameter value with optional validation.
        
        Returns True if successful, False if validation failed.
        """
        spec = self.PARAM_SPECS.get(name)
        
        if validate and spec:
            value = spec.clamp(value)
        
        old_value = self.engine.params.get(name)
        self.engine.params[name] = value
        
        # Record change
        self.change_history.append((name, old_value, value))
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(name, old_value, value)
            except Exception as e:
                print(f"Param callback error: {e}")
        
        return True
    
    def reset(self, name: str):
        """Reset parameter to default value."""
        spec = self.PARAM_SPECS.get(name)
        if spec:
            self.set(name, spec.default)
    
    def reset_all(self):
        """Reset all parameters to defaults."""
        for name, spec in self.PARAM_SPECS.items():
            self.engine.params[name] = spec.default
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.PRESETS.keys())}")
        
        preset = self.PRESETS[preset_name]
        for name, value in preset.items():
            self.set(name, value, validate=False)
    
    def get_all(self) -> Dict[str, float]:
        """Get all current parameter values."""
        return dict(self.engine.params)
    
    def export_json(self) -> str:
        """Export current parameters as JSON string."""
        import json
        return json.dumps(self.engine.params, indent=2)
    
    def import_json(self, json_str: str):
        """Import parameters from JSON string."""
        import json
        params = json.loads(json_str)
        for name, value in params.items():
            self.set(name, value, validate=True)
    
    def save_to_file(self, filename: str):
        """Save parameters to file."""
        with open(filename, 'w') as f:
            f.write(self.export_json())
    
    def load_from_file(self, filename: str):
        """Load parameters from file."""
        with open(filename, 'r') as f:
            self.import_json(f.read())
    
    def add_change_callback(self, callback: callable):
        """Add callback for parameter changes. Signature: callback(name, old, new)"""
        self.callbacks.append(callback)
    
    def get_param_info(self, name: str) -> Dict[str, any]:
        """Get full info about a parameter."""
        spec = self.PARAM_SPECS.get(name)
        if not spec:
            return {'name': name, 'value': self.get(name), 'spec': None}
        
        return {
            'name': name,
            'value': self.get(name),
            'min': spec.min_val,
            'max': spec.max_val,
            'default': spec.default,
            'step': spec.step,
            'description': spec.description,
        }
    
    def list_params(self) -> List[Dict[str, any]]:
        """List all tunable parameters with their specs."""
        return [self.get_param_info(name) for name in self.PARAM_SPECS]


# =============================================================================
# FEATURE 3: REAL-TIME PYGAME INTEGRATION
# =============================================================================

class PygameRenderer:
    """
    Real-time Pygame renderer for the Motion Engine.
    
    Provides interactive visualization with controls.
    
    Controls:
        SPACE: Pause/resume
        R: Reset simulation
        S: Spawn new entity at mouse position
        +/-: Adjust simulation speed
        1-6: Show only entities at that stage
        0: Show all stages
        V: Toggle volume view
        Arrow keys: Pan view
        Mouse wheel: Zoom
        Click: Inspect entity
        Q/ESC: Quit
    """
    
    def __init__(self, engine: 'MotionEngine', width: int = 1200, height: int = 800):
        self.engine = engine
        self.screen_width = width
        self.screen_height = height
        self.pygame = None
        self.screen = None
        self.clock = None
        self.font = None
        
        # View state
        self.running = False
        self.paused = False
        self.show_volume = False
        self.stage_filter = None  # None = show all
        self.sim_speed = 1.0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Selection
        self.selected_entity_id = None
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.text_color = (200, 200, 200)
        self.grid_color = (40, 40, 50)
        
        # Render hooks integration
        self.render_hooks = RenderHooks(engine)
        self.param_tuner = ParamTuner(engine)
    
    def _init_pygame(self):
        """Initialize Pygame."""
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Motion Engine - Universe of Motion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        sx = int((x + self.pan_x) * self.zoom)
        sy = int((y + self.pan_y) * self.zoom)
        return (sx, sy)
    
    def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        x = sx / self.zoom - self.pan_x
        y = sy / self.zoom - self.pan_y
        return (x, y)
    
    def _draw_entity(self, ev: EntityVisual, snapshot: RenderSnapshot):
        """Draw a single entity."""
        if self.stage_filter is not None and ev.stage.value != self.stage_filter:
            return
        
        pos = self._world_to_screen(ev.position[0], ev.position[1])
        color = snapshot.get_entity_color(ev)
        radius = max(2, int(snapshot.get_entity_radius(ev) * self.zoom))
        
        # Draw entity circle
        self.pygame.draw.circle(self.screen, color, pos, radius)
        
        # Highlight selected
        if ev.id == self.selected_entity_id:
            self.pygame.draw.circle(self.screen, (255, 255, 0), pos, radius + 3, 2)
        
        # Draw polarity indicator for POLARITY+ stages
        if ev.stage.value >= Stage.POLARITY.value and abs(ev.polarity) > 0.1:
            pol_color = snapshot.get_polarity_color(ev.polarity)
            self.pygame.draw.circle(self.screen, pol_color, pos, radius // 2)
        
        # Draw velocity arrow for MOVEMENT stage
        if ev.stage == Stage.MOVEMENT and ev.speed > 5.0:
            end_x = pos[0] + int(ev.velocity[0] * 0.3 * self.zoom)
            end_y = pos[1] + int(ev.velocity[1] * 0.3 * self.zoom)
            self.pygame.draw.line(self.screen, (255, 255, 255), pos, (end_x, end_y), 1)
    
    def _draw_bonds(self, snapshot: RenderSnapshot):
        """Draw bond lines between entities."""
        for p1, p2 in snapshot.bond_pairs:
            sp1 = self._world_to_screen(p1[0], p1[1])
            sp2 = self._world_to_screen(p2[0], p2[1])
            self.pygame.draw.line(self.screen, (100, 100, 150), sp1, sp2, 1)
    
    def _draw_volume(self, snapshot: RenderSnapshot):
        """Draw structural volume V in a side panel."""
        if not snapshot.cells:
            return
        
        # Draw in right portion of screen
        panel_x = self.screen_width - 300
        panel_y = 50
        scale = 25
        
        # Background
        self.pygame.draw.rect(self.screen, (30, 30, 40), 
                              (panel_x - 10, panel_y - 10, 280, 280))
        
        # Title
        title = self.font.render("Structural Volume V", True, self.text_color)
        self.screen.blit(title, (panel_x, panel_y - 30))
        
        # Draw cells with isometric projection
        for cell in snapshot.cells:
            x, y = snapshot.project_cell_to_2d(cell, scale, (panel_x + 100, panel_y + 150))
            
            eid = snapshot.cell_to_entity.get(cell, -1)
            ev = next((e for e in snapshot.entities if e.id == eid), None)
            if ev:
                color = snapshot.get_entity_color(ev)
            else:
                color = (100, 100, 100)
            
            self.pygame.draw.circle(self.screen, color, (int(x), int(y)), 8)
            
            # Cell label
            label = self.small_font.render(f"{cell[0]},{cell[1]},{cell[2]}", True, (150, 150, 150))
            self.screen.blit(label, (int(x) - 12, int(y) + 10))
        
        # Draw edges
        for c1, c2 in snapshot.cell_edges:
            p1 = snapshot.project_cell_to_2d(c1, scale, (panel_x + 100, panel_y + 150))
            p2 = snapshot.project_cell_to_2d(c2, scale, (panel_x + 100, panel_y + 150))
            self.pygame.draw.line(self.screen, (60, 60, 80), 
                                  (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1)
    
    def _draw_ui(self, snapshot: RenderSnapshot):
        """Draw UI overlay."""
        y = 10
        line_height = 20
        
        # Stats
        stats = [
            f"Time: {snapshot.time:.2f}  Step: {snapshot.unfolding_step}",
            f"Entities: {len(snapshot.entities)}  Heat: {snapshot.total_heat:.2f}",
            f"Speed: {self.sim_speed:.1f}x  {'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "Stage counts:",
        ]
        
        for stage, count in snapshot.stage_counts.items():
            color = RenderSnapshot.STAGE_COLORS[stage]
            stats.append(f"  {stage.name}: {count}")
        
        for i, text in enumerate(stats):
            if ":" in text and text.strip().startswith(("HEAT", "POLARITY", "EXISTENCE", "RIGHTEOUSNESS", "ORDER", "MOVEMENT")):
                stage_name = text.strip().split(":")[0]
                stage = Stage[stage_name]
                color = RenderSnapshot.STAGE_COLORS[stage]
            else:
                color = self.text_color
            
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, y + i * line_height))
        
        # Controls help
        controls = [
            "SPACE: Pause  R: Reset  S: Spawn",
            "+/-: Speed  V: Volume  0-6: Filter",
            "Click: Select  Q: Quit",
        ]
        for i, text in enumerate(controls):
            surface = self.small_font.render(text, True, (150, 150, 150))
            self.screen.blit(surface, (10, self.screen_height - 60 + i * 18))
        
        # Selected entity info
        if self.selected_entity_id is not None:
            ev = next((e for e in snapshot.entities if e.id == self.selected_entity_id), None)
            if ev:
                info = [
                    f"Selected: Entity {ev.id}",
                    f"Stage: {ev.stage_name}",
                    f"Heat: {ev.heat:.3f}",
                    f"Polarity: {ev.polarity:.3f}",
                    f"Existence: {ev.existence:.3f}",
                    f"Righteousness: {ev.righteousness:.3f}",
                    f"Cell: {ev.cell}",
                    f"Energy: {ev.energy:.3f}",
                ]
                
                panel_x = self.screen_width - 300
                panel_y = self.screen_height - 200
                self.pygame.draw.rect(self.screen, (30, 30, 40), 
                                      (panel_x - 10, panel_y - 10, 200, 180))
                
                for i, text in enumerate(info):
                    surface = self.small_font.render(text, True, self.text_color)
                    self.screen.blit(surface, (panel_x, panel_y + i * 18))
    
    def _handle_events(self):
        """Handle Pygame events."""
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE or event.key == self.pygame.K_q:
                    self.running = False
                elif event.key == self.pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == self.pygame.K_r:
                    # Reset simulation
                    self.engine.entities.clear()
                    self.engine.occupied_cells.clear()
                    self.engine.time = 0
                    self.engine.unfolding_step = 0
                    for _ in range(20):
                        self.engine.spawn()
                elif event.key == self.pygame.K_s:
                    # Spawn at mouse position
                    mx, my = self.pygame.mouse.get_pos()
                    wx, wy = self._screen_to_world(mx, my)
                    self.engine.spawn(x=wx, y=wy)
                elif event.key == self.pygame.K_v:
                    self.show_volume = not self.show_volume
                elif event.key == self.pygame.K_EQUALS or event.key == self.pygame.K_PLUS:
                    self.sim_speed = min(10.0, self.sim_speed * 1.5)
                elif event.key == self.pygame.K_MINUS:
                    self.sim_speed = max(0.1, self.sim_speed / 1.5)
                elif event.key == self.pygame.K_0:
                    self.stage_filter = None
                elif event.key in (self.pygame.K_1, self.pygame.K_2, self.pygame.K_3,
                                   self.pygame.K_4, self.pygame.K_5, self.pygame.K_6):
                    self.stage_filter = event.key - self.pygame.K_1
                elif event.key == self.pygame.K_LEFT:
                    self.pan_x += 20
                elif event.key == self.pygame.K_RIGHT:
                    self.pan_x -= 20
                elif event.key == self.pygame.K_UP:
                    self.pan_y += 20
                elif event.key == self.pygame.K_DOWN:
                    self.pan_y -= 20
            
            elif event.type == self.pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mx, my = event.pos
                    wx, wy = self._screen_to_world(mx, my)
                    
                    # Find closest entity
                    closest = None
                    closest_dist = 50  # Max selection distance
                    for e in self.engine.entities:
                        dist = math.sqrt((e.position.x - wx)**2 + (e.position.y - wy)**2)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest = e.id
                    self.selected_entity_id = closest
                
                elif event.button == 4:  # Scroll up
                    self.zoom = min(5.0, self.zoom * 1.1)
                elif event.button == 5:  # Scroll down
                    self.zoom = max(0.2, self.zoom / 1.1)
    
    def run(self, fps: int = 60, dt: float = 0.05):
        """
        Run the interactive visualization.
        
        Args:
            fps: Target frames per second
            dt: Simulation timestep per frame
        """
        self._init_pygame()
        self.running = True
        
        while self.running:
            self._handle_events()
            
            # Update simulation
            if not self.paused:
                for _ in range(int(self.sim_speed)):
                    self.engine.step(dt)
            
            # Clear screen
            self.screen.fill(self.bg_color)
            
            # Draw grid
            for x in range(0, self.engine.width, 50):
                sx, _ = self._world_to_screen(x, 0)
                self.pygame.draw.line(self.screen, self.grid_color, 
                                      (sx, 0), (sx, self.screen_height), 1)
            for y in range(0, self.engine.height, 50):
                _, sy = self._world_to_screen(0, y)
                self.pygame.draw.line(self.screen, self.grid_color,
                                      (0, sy), (self.screen_width, sy), 1)
            
            # Get snapshot and draw
            snapshot = self.render_hooks.get_snapshot()
            
            self._draw_bonds(snapshot)
            for ev in snapshot.entities:
                self._draw_entity(ev, snapshot)
            
            if self.show_volume:
                self._draw_volume(snapshot)
            
            self._draw_ui(snapshot)
            
            # Update display
            self.pygame.display.flip()
            self.clock.tick(fps)
        
        self.pygame.quit()


# =============================================================================
# FEATURE 4: NON-COMMUTATION TESTS AND DEMOS
# =============================================================================

class NonCommutationTests:
    """
    Test suite for non-commutation properties of the n³+t framework.
    
    Validates that:
    1. Diagonal operators commute (AB = BA)
    2. Shift operators don't commute (AB ≠ BA)
    3. Flow operators don't commute
    4. Commutator magnitude relates to uncertainty
    """
    
    def __init__(self, engine: Optional['MotionEngine'] = None):
        self.engine = engine or self._create_test_engine()
        self.results: List[Dict[str, any]] = []
    
    def _create_test_engine(self) -> 'MotionEngine':
        """Create a small test engine."""
        engine = MotionEngine(200, 200)
        # Spawn entities and evolve to ORDER stage
        for _ in range(8):
            engine.spawn()
        for _ in range(200):
            engine.step(0.1)
        return engine
    
    def test_diagonal_commutation(self) -> Dict[str, any]:
        """Test that diagonal operators commute."""
        ops = self.engine.get_constraint_operators()
        state = self.engine.get_state_vector()
        
        results = {
            'name': 'diagonal_commutation',
            'description': 'Diagonal operators should commute (AB = BA)',
            'pairs': [],
            'passed': True,
        }
        
        op_names = list(ops.keys())
        for i, name_a in enumerate(op_names):
            for name_b in op_names[i+1:]:
                commutator = self.engine.measure_commutator(ops[name_a], ops[name_b], state)
                pair_result = {
                    'operators': (name_a, name_b),
                    'commutator': commutator,
                    'commutes': commutator < 1e-10,
                }
                results['pairs'].append(pair_result)
                if not pair_result['commutes']:
                    results['passed'] = False
        
        self.results.append(results)
        return results
    
    def test_shift_non_commutation(self) -> Dict[str, any]:
        """Test that shift operators don't commute."""
        state = self.engine.get_state_vector()
        
        shift_i = AxisShiftOperator(axis=0, direction=1)
        shift_j = AxisShiftOperator(axis=1, direction=1)
        shift_k = AxisShiftOperator(axis=2, direction=1)
        
        results = {
            'name': 'shift_non_commutation',
            'description': 'Shift operators should NOT commute in general',
            'pairs': [],
            'has_non_commuting': False,
        }
        
        pairs = [
            ('shift_i', 'shift_j', shift_i, shift_j),
            ('shift_j', 'shift_k', shift_j, shift_k),
            ('shift_i', 'shift_k', shift_i, shift_k),
        ]
        
        for name_a, name_b, op_a, op_b in pairs:
            commutator = self.engine.measure_commutator(op_a, op_b, state)
            pair_result = {
                'operators': (name_a, name_b),
                'commutator': commutator,
                'commutes': commutator < 1e-6,
            }
            results['pairs'].append(pair_result)
            if not pair_result['commutes']:
                results['has_non_commuting'] = True
        
        results['passed'] = results['has_non_commuting']
        self.results.append(results)
        return results
    
    def test_flow_non_commutation(self) -> Dict[str, any]:
        """Test that flow operators don't commute with shifts."""
        state = self.engine.get_state_vector()
        
        shift_i = AxisShiftOperator(axis=0, direction=1)
        polar_flow = PolarShiftOperator()
        right_flow = RighteousnessFlowOperator()
        
        results = {
            'name': 'flow_non_commutation',
            'description': 'Flow operators should NOT commute with shifts',
            'pairs': [],
            'has_non_commuting': False,
        }
        
        pairs = [
            ('shift_i', 'polar_flow', shift_i, polar_flow),
            ('shift_i', 'righteous_flow', shift_i, right_flow),
            ('polar_flow', 'righteous_flow', polar_flow, right_flow),
        ]
        
        for name_a, name_b, op_a, op_b in pairs:
            commutator = self.engine.measure_commutator(op_a, op_b, state)
            pair_result = {
                'operators': (name_a, name_b),
                'commutator': commutator,
                'commutes': commutator < 1e-6,
            }
            results['pairs'].append(pair_result)
            if not pair_result['commutes']:
                results['has_non_commuting'] = True
        
        results['passed'] = results['has_non_commuting']
        self.results.append(results)
        return results
    
    def test_order_dependence(self) -> Dict[str, any]:
        """Test that operator order produces different final states."""
        state = self.engine.get_state_vector()
        
        shift_i = AxisShiftOperator(axis=0, direction=1)
        shift_j = AxisShiftOperator(axis=1, direction=1)
        polar_flow = PolarShiftOperator()
        
        results = {
            'name': 'order_dependence',
            'description': 'Different operator orders should produce different states',
            'comparisons': [],
            'passed': False,
        }
        
        # Compare AB vs BA
        state_ij = self.engine.apply_operator_sequence([shift_i, shift_j], state)
        state_ji = self.engine.apply_operator_sequence([shift_j, shift_i], state)
        
        diff_ij_ji = sum((state_ij.get(c, 0) - state_ji.get(c, 0))**2 
                        for c in set(state_ij.keys()) | set(state_ji.keys()))
        
        results['comparisons'].append({
            'sequence_a': ['shift_i', 'shift_j'],
            'sequence_b': ['shift_j', 'shift_i'],
            'difference_norm_sq': diff_ij_ji,
            'differs': diff_ij_ji > 1e-10,
        })
        
        # Compare with flow operator
        state_ip = self.engine.apply_operator_sequence([shift_i, polar_flow], state)
        state_pi = self.engine.apply_operator_sequence([polar_flow, shift_i], state)
        
        diff_ip_pi = sum((state_ip.get(c, 0) - state_pi.get(c, 0))**2
                        for c in set(state_ip.keys()) | set(state_pi.keys()))
        
        results['comparisons'].append({
            'sequence_a': ['shift_i', 'polar_flow'],
            'sequence_b': ['polar_flow', 'shift_i'],
            'difference_norm_sq': diff_ip_pi,
            'differs': diff_ip_pi > 1e-10,
        })
        
        results['passed'] = any(c['differs'] for c in results['comparisons'])
        self.results.append(results)
        return results
    
    def test_uncertainty_relation(self) -> Dict[str, any]:
        """Test uncertainty relation ΔA × ΔB ≥ ½|⟨[A,B]⟩|."""
        state = self.engine.get_state_vector()
        
        shift_i = AxisShiftOperator(axis=0, direction=1)
        shift_j = AxisShiftOperator(axis=1, direction=1)
        
        relation = UncertaintyRelation(self.engine, shift_i, shift_j)
        uncertainty = relation.compute_uncertainty_product(state)
        
        results = {
            'name': 'uncertainty_relation',
            'description': 'Uncertainty product should satisfy ΔA × ΔB ≥ ½|⟨[A,B]⟩|',
            'operators': ('shift_i', 'shift_j'),
            'variance_a': uncertainty['variance_a'],
            'variance_b': uncertainty['variance_b'],
            'uncertainty_product': uncertainty['uncertainty_product'],
            'commutator_bound': uncertainty['commutator_bound'],
            'relation_satisfied': uncertainty['relation_satisfied'],
            'passed': uncertainty['relation_satisfied'],
        }
        
        self.results.append(results)
        return results
    
    def test_incompatibility_accumulation(self) -> Dict[str, any]:
        """Test that incompatibility accumulates over operator sequences."""
        state = self.engine.get_state_vector()
        
        operators = [
            AxisShiftOperator(axis=0, direction=1),
            AxisShiftOperator(axis=1, direction=1),
            PolarShiftOperator(),
            AxisShiftOperator(axis=2, direction=1),
        ]
        
        tracker = IncompatibilityTracker(self.engine)
        current_state = state
        
        for i in range(len(operators) - 1):
            tracker.record_application(operators[i], operators[i+1], current_state)
            current_state = operators[i].apply(current_state, self.engine)
        
        results = {
            'name': 'incompatibility_accumulation',
            'description': 'Incompatibility should accumulate across operator sequence',
            'sequence_length': len(operators),
            'total_incompatibility': tracker.total_incompatibility,
            'applications_tracked': len(tracker.pair_history),
            'hotspots': tracker.get_hotspots(3),
            'passed': tracker.total_incompatibility > 0,
        }
        
        self.results.append(results)
        return results
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all non-commutation tests."""
        self.results.clear()
        
        tests = [
            self.test_diagonal_commutation,
            self.test_shift_non_commutation,
            self.test_flow_non_commutation,
            self.test_order_dependence,
            self.test_uncertainty_relation,
            self.test_incompatibility_accumulation,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.results.append({
                    'name': test.__name__,
                    'error': str(e),
                    'passed': False,
                })
        
        passed = sum(1 for r in self.results if r.get('passed', False))
        total = len(self.results)
        
        return {
            'passed': passed,
            'total': total,
            'all_passed': passed == total,
            'results': self.results,
        }
    
    def print_report(self):
        """Print a formatted test report."""
        print("\n" + "="*70)
        print("NON-COMMUTATION TEST REPORT")
        print("="*70)
        
        for result in self.results:
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            print(f"\n{status}: {result['name']}")
            print(f"  {result.get('description', '')}")
            
            if 'pairs' in result:
                for pair in result['pairs']:
                    comm = pair['commutator']
                    ops = pair['operators']
                    sym = "=" if pair.get('commutes', False) else "≠"
                    print(f"    [{ops[0]}, {ops[1]}] {sym} 0  (|[A,B]| = {comm:.6f})")
            
            if 'comparisons' in result:
                for comp in result['comparisons']:
                    diff = comp['difference_norm_sq']
                    print(f"    {comp['sequence_a']} vs {comp['sequence_b']}: ||diff||² = {diff:.6f}")
            
            if 'error' in result:
                print(f"  ERROR: {result['error']}")
        
        summary = self.run_all_tests()
        print(f"\n{'='*70}")
        print(f"SUMMARY: {summary['passed']}/{summary['total']} tests passed")
        print("="*70)


def demo_non_commutation():
    """
    Demo script showing non-commutation in action.
    
    Creates an engine, applies operators in different orders,
    and visualizes the resulting state differences.
    """
    print("\n" + "="*70)
    print("NON-COMMUTATION DEMONSTRATION")
    print("="*70)
    
    # Create engine
    engine = MotionEngine(400, 400)
    for _ in range(15):
        engine.spawn()
    
    # Evolve to ORDER stage
    print("\nEvolving entities to ORDER stage...")
    for _ in range(250):
        engine.step(0.1)
    
    print(f"  Entities: {len(engine.entities)}")
    print(f"  Occupied cells: {len(engine.occupied_cells)}")
    
    # Get initial state
    state = engine.get_state_vector()
    initial_entropy = engine.compute_state_entropy(state)
    print(f"  Initial state entropy: {initial_entropy:.4f}")
    
    # Define operators
    shift_i = AxisShiftOperator(axis=0, direction=1, transfer_rate=0.3)
    shift_j = AxisShiftOperator(axis=1, direction=1, transfer_rate=0.3)
    polar_flow = PolarShiftOperator(transfer_rate=0.25)
    
    print("\n--- Operator Order Comparison ---")
    
    # Sequence 1: i then j
    state_ij = engine.apply_operator_sequence([shift_i, shift_j], state)
    entropy_ij = engine.compute_state_entropy(state_ij)
    
    # Sequence 2: j then i
    state_ji = engine.apply_operator_sequence([shift_j, shift_i], state)
    entropy_ji = engine.compute_state_entropy(state_ji)
    
    # Compute difference
    diff_ij_ji = math.sqrt(sum((state_ij.get(c, 0) - state_ji.get(c, 0))**2 
                               for c in set(state_ij.keys()) | set(state_ji.keys())))
    
    print(f"\n  shift_i · shift_j:")
    print(f"    Final entropy: {entropy_ij:.4f}")
    print(f"  shift_j · shift_i:")
    print(f"    Final entropy: {entropy_ji:.4f}")
    print(f"  ||state_ij - state_ji||: {diff_ij_ji:.6f}")
    print(f"  -> {'DIFFERENT' if diff_ij_ji > 1e-6 else 'SAME'} final states!")
    
    # Three-operator comparison
    print("\n--- Three-Operator Sequences ---")
    
    state_ijp = engine.apply_operator_sequence([shift_i, shift_j, polar_flow], state)
    state_pij = engine.apply_operator_sequence([polar_flow, shift_i, shift_j], state)
    state_jpi = engine.apply_operator_sequence([shift_j, polar_flow, shift_i], state)
    
    diff_ijp_pij = math.sqrt(sum((state_ijp.get(c, 0) - state_pij.get(c, 0))**2 
                                  for c in set(state_ijp.keys()) | set(state_pij.keys())))
    diff_ijp_jpi = math.sqrt(sum((state_ijp.get(c, 0) - state_jpi.get(c, 0))**2 
                                  for c in set(state_ijp.keys()) | set(state_jpi.keys())))
    
    print(f"  shift_i · shift_j · polar_flow  vs")
    print(f"  polar_flow · shift_i · shift_j")
    print(f"    ||difference||: {diff_ijp_pij:.6f}")
    
    print(f"  shift_i · shift_j · polar_flow  vs")
    print(f"  shift_j · polar_flow · shift_i")
    print(f"    ||difference||: {diff_ijp_jpi:.6f}")
    
    # State evolution tracking
    print("\n--- State Evolution with Incompatibility Tracking ---")
    
    evolution = engine.create_state_evolution()
    operators = [shift_i, shift_j, polar_flow, shift_i, shift_j]
    
    for i, op in enumerate(operators):
        prev = operators[i-1] if i > 0 else None
        evolution.apply(op, track_incompatibility_with=prev)
    
    summary = evolution.summary()
    print(f"  Operators applied: {summary['operators_applied']}")
    print(f"  Cumulative incompatibility: {summary['cumulative_incompatibility']:.6f}")
    print(f"  Entropy change: {summary['entropy_change']:.4f}")
    
    # Amplitude flow visualization (text-based)
    print("\n--- Amplitude Concentration ---")
    
    initial_loc = engine.compute_localization(state)
    final_loc = engine.compute_localization(evolution.current_state)
    
    print(f"  Initial: participation={initial_loc['participation_ratio']:.2f}, spread={initial_loc['spread']:.4f}")
    print(f"  Final:   participation={final_loc['participation_ratio']:.2f}, spread={final_loc['spread']:.4f}")
    
    change = final_loc['participation_ratio'] - initial_loc['participation_ratio']
    if change < -0.5:
        print("  -> State LOCALIZED (amplitude concentrated)")
    elif change > 0.5:
        print("  -> State SPREAD (amplitude dispersed)")
    else:
        print("  -> State roughly unchanged")
    
    print("\n" + "="*70)
    print("Demo complete. Run NonCommutationTests().run_all_tests() for full validation.")
    print("="*70)
