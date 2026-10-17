/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Melissa Monfared
-/
import Mathlib

/-!
# Finite discounted MDPs, the planning layer

This file proves the 26 nodes of the priority 1 blueprint
("RL planning layer blueprint for LML") for finite MDPs with a discount factor below one.
There is no `sorry` anywhere.

Steps 1 to 3 set up the model, the Bellman operators, their contraction and monotonicity,
and the comparison principle (D1 to D5, L1 to L6). Steps 4 and 5 define values as fixed
points and prove Bellman optimality among stationary policies (D6, D7, L7 to L9, T1).
Steps 6 and 7 cover value iteration and policy iteration (L10, L11, T2, T3). Steps 8 and 9
prove the performance difference lemma, the simulation lemma and planning in a perturbed
model (D8, L12 to L15).

Labels such as D1 or L2 refer to the blueprint. Book references are to Agarwal, Brantley,
Jiang, Kakade and Sun, *Reinforcement Learning Theory and Algorithms*, draft of
27 June 2026.

Checked on 14 October 2026 with Lean 4.35.0 and the latest Mathlib.
-/

-- This only silences warnings about unused instance arguments.
set_option linter.unusedSectionVars false

open scoped NNReal

noncomputable section

namespace Learning

/-! ## Part A. Setting -/

/-- **D1.** A finite discounted MDP. Book Section 1.1. -/
structure FiniteMDP (S A : Type*) [Fintype S] where
  /-- `P s a s'` is the probability of moving to `s'` after playing `a` in `s`. -/
  P : S → A → S → ℝ
  P_nonneg : ∀ s a s', 0 ≤ P s a s'
  P_sum : ∀ s a, ∑ s', P s a s' = 1
  /-- `r s a` is the reward for playing `a` in `s`. -/
  r : S → A → ℝ
  /-- The discount factor. -/
  γ : ℝ≥0
  γ_lt_one : γ < 1

/-- **D2.** A stationary, possibly randomized, policy. Book Section 1.1.1. -/
structure StationaryPolicy (S A : Type*) [Fintype A] where
  /-- `prob s a` is the probability of playing `a` in state `s`. -/
  prob : S → A → ℝ
  nonneg : ∀ s a, 0 ≤ prob s a
  sum_eq_one : ∀ s, ∑ a, prob s a = 1

variable {S A : Type*} [Fintype S] [Fintype A] [DecidableEq A] [Nonempty A]

/-- The probability vector of the deterministic choice `d`. -/
def detProb (d : S → A) (s : S) (a : A) : ℝ := if a = d s then 1 else 0

/-- **D2.** A deterministic policy, seen as a stationary policy. -/
def StationaryPolicy.ofDet (d : S → A) : StationaryPolicy S A where
  prob := detProb d
  nonneg s a := by
    unfold detProb
    split_ifs <;> norm_num
  sum_eq_one s := by
    simp [detProb]

/-- **D3.** The average of `Q s ·` under the policy, at every state `s`. -/
def StationaryPolicy.avg (π : StationaryPolicy S A) (Q : S → A → ℝ) (s : S) : ℝ :=
  ∑ a, π.prob s a * Q s a

/-- **D5.** The greedy value `max_a Q s a`. -/
def maxQ (Q : S → A → ℝ) (s : S) : ℝ :=
  (Finset.univ : Finset A).sup' Finset.univ_nonempty (Q s)

/-- **D5.** A greedy action at every state. Ties are broken by choice. -/
def greedy (Q : S → A → ℝ) : S → A :=
  fun s => (Finset.exists_max_image Finset.univ (Q s) Finset.univ_nonempty).choose

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **D3.** The reward averaged by the policy. -/
def rewardPi (π : StationaryPolicy S A) : S → ℝ := π.avg M.r

/-- **D3.** The transition matrix of the chain that the policy induces. -/
def transPi (π : StationaryPolicy S A) (s s' : S) : ℝ := ∑ a, π.prob s a * M.P s a s'

/-- **D4.** The evaluation operator on state values. -/
def evalOp (π : StationaryPolicy S A) (V : S → ℝ) : S → ℝ :=
  fun s => M.rewardPi π s + (M.γ : ℝ) * ∑ s', M.transPi π s s' * V s'

/-- **D4.** The evaluation operator on Q-functions. -/
def evalOpQ (π : StationaryPolicy S A) (Q : S → A → ℝ) : S → A → ℝ :=
  fun s a => M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * π.avg Q s'

/-- **D5.** The Bellman optimality operator. Book equation (1.4). -/
def optOp (Q : S → A → ℝ) : S → A → ℝ :=
  fun s a => M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * maxQ Q s'

/-- The same MDP with another reward function. Used later in L12 and L14. -/
def withReward (ρ : S → A → ℝ) : FiniteMDP S A := { M with r := ρ }

end FiniteMDP

/-! ## Facts about greedy actions (support for D5) -/

/-- Every action is at most the max. -/
theorem le_maxQ (Q : S → A → ℝ) (s : S) (a : A) : Q s a ≤ maxQ Q s :=
  Finset.le_sup' (Q s) (Finset.mem_univ a)

/-- The greedy action beats every action. -/
theorem le_greedy (Q : S → A → ℝ) (s : S) (a : A) : Q s a ≤ Q s (greedy Q s) :=
  (Finset.exists_max_image Finset.univ (Q s) Finset.univ_nonempty).choose_spec.2 a
    (Finset.mem_univ a)

/-- The greedy action attains the max. -/
theorem maxQ_eq_greedy (Q : S → A → ℝ) (s : S) : maxQ Q s = Q s (greedy Q s) := by
  apply le_antisymm
  · unfold maxQ
    rw [Finset.sup'_le_iff]
    intro a _
    exact le_greedy Q s a
  · exact le_maxQ Q s (greedy Q s)

/-- A policy average is at most the max. -/
theorem StationaryPolicy.avg_le_maxQ (π : StationaryPolicy S A) (Q : S → A → ℝ) (s : S) :
    π.avg Q s ≤ maxQ Q s := by
  unfold StationaryPolicy.avg
  calc ∑ a, π.prob s a * Q s a ≤ ∑ a, π.prob s a * maxQ Q s :=
        Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (le_maxQ Q s a) (π.nonneg s a)
    _ = maxQ Q s := by rw [← Finset.sum_mul, π.sum_eq_one s, one_mul]

/-! ## Part B, first lemmas -/

/-- **L1.** Averaging against a probability vector is bounded by the sup norm. -/
theorem abs_sum_mul_le_norm {X : Type*} [Fintype X] {p : X → ℝ}
    (hp : ∀ x, 0 ≤ p x) (h1 : ∑ x, p x = 1) (f : X → ℝ) :
    |∑ x, p x * f x| ≤ ‖f‖ := by
  calc |∑ x, p x * f x| ≤ ∑ x, |p x * f x| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, p x * |f x| := by
        refine Finset.sum_congr rfl fun x _ => ?_
        rw [abs_mul, abs_of_nonneg (hp x)]
    _ ≤ ∑ x, p x * ‖f‖ := by
        refine Finset.sum_le_sum fun x _ => ?_
        have h := norm_le_pi_norm f x
        rw [Real.norm_eq_abs] at h
        exact mul_le_mul_of_nonneg_left h (hp x)
    _ = ‖f‖ := by rw [← Finset.sum_mul, h1, one_mul]

/-- **L1.** Averaging against nonnegative weights is monotone. -/
theorem sum_mul_le_sum_mul {X : Type*} [Fintype X] {p : X → ℝ}
    (hp : ∀ x, 0 ≤ p x) {f g : X → ℝ} (hfg : f ≤ g) :
    ∑ x, p x * f x ≤ ∑ x, p x * g x :=
  Finset.sum_le_sum fun x _ => mul_le_mul_of_nonneg_left (hfg x) (hp x)

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **L1.** The entries of `transPi` are nonnegative. -/
theorem transPi_nonneg (π : StationaryPolicy S A) (s s' : S) : 0 ≤ M.transPi π s s' := by
  unfold transPi
  exact Finset.sum_nonneg fun a _ => mul_nonneg (π.nonneg s a) (M.P_nonneg s a s')

/-- **L1.** The rows of `transPi` sum to one. -/
theorem sum_transPi (π : StationaryPolicy S A) (s : S) : ∑ s', M.transPi π s s' = 1 := by
  unfold transPi
  rw [Finset.sum_comm]
  calc ∑ a, ∑ s', π.prob s a * M.P s a s' = ∑ a, π.prob s a := by
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [← Finset.mul_sum, M.P_sum s a, mul_one]
    _ = 1 := π.sum_eq_one s

end FiniteMDP

/-- The max over a finite nonempty type moves by at most the sup norm of the change. -/
theorem sup'_le_sup'_add_norm {X : Type*} [Fintype X] [Nonempty X] (f g : X → ℝ) :
    (Finset.univ : Finset X).sup' Finset.univ_nonempty f ≤
      (Finset.univ : Finset X).sup' Finset.univ_nonempty g + ‖f - g‖ := by
  rw [Finset.sup'_le_iff]
  intro x _
  have h1 : f x - g x ≤ ‖f - g‖ := by
    have h := norm_le_pi_norm (f - g) x
    rw [Pi.sub_apply, Real.norm_eq_abs] at h
    exact (le_abs_self _).trans h
  have h2 : g x ≤ (Finset.univ : Finset X).sup' Finset.univ_nonempty g :=
    Finset.le_sup' g (Finset.mem_univ x)
  linarith

/-- **L2.** The max over actions is 1-Lipschitz. Inside the proof of book Lemma 1.10. -/
theorem abs_maxQ_sub_maxQ_le (Q Q' : S → A → ℝ) (s : S) :
    |maxQ Q s - maxQ Q' s| ≤ ‖Q s - Q' s‖ := by
  have h1 := sup'_le_sup'_add_norm (Q s) (Q' s)
  have h2 := sup'_le_sup'_add_norm (Q' s) (Q s)
  rw [norm_sub_rev] at h2
  unfold maxQ
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-! End of step 1. -/

/-! ## Step 2. Contraction and monotonicity (L3, L4, L5) -/

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- The discount factor is nonnegative as a real number. -/
theorem gamma_nonneg : (0 : ℝ) ≤ M.γ := NNReal.coe_nonneg _

/-- **L4.** The evaluation operator on state values is monotone. -/
theorem monotone_evalOp (π : StationaryPolicy S A) : Monotone (M.evalOp π) := by
  intro V V' h s
  have hsum : ∑ s', M.transPi π s s' * V s' ≤ ∑ s', M.transPi π s s' * V' s' :=
    sum_mul_le_sum_mul (M.transPi_nonneg π s) h
  have hmul := mul_le_mul_of_nonneg_left hsum M.gamma_nonneg
  simp only [evalOp]
  linarith

/-- **L4.** The evaluation operator on Q-functions is monotone. -/
theorem monotone_evalOpQ (π : StationaryPolicy S A) : Monotone (M.evalOpQ π) := by
  intro Q Q' h s a
  have havg : π.avg Q ≤ π.avg Q' := by
    intro s'
    exact sum_mul_le_sum_mul (π.nonneg s') (h s')
  have hsum : ∑ s', M.P s a s' * π.avg Q s' ≤ ∑ s', M.P s a s' * π.avg Q' s' :=
    sum_mul_le_sum_mul (M.P_nonneg s a) havg
  have hmul := mul_le_mul_of_nonneg_left hsum M.gamma_nonneg
  simp only [evalOpQ]
  linarith

/-- **L4.** The optimality operator is monotone. -/
theorem monotone_optOp : Monotone M.optOp := by
  intro Q Q' h s a
  have hmax : maxQ Q ≤ maxQ Q' := by
    intro s'
    unfold maxQ
    rw [Finset.sup'_le_iff]
    intro b _
    exact (h s' b).trans (le_maxQ Q' s' b)
  have hsum : ∑ s', M.P s a s' * maxQ Q s' ≤ ∑ s', M.P s a s' * maxQ Q' s' :=
    sum_mul_le_sum_mul (M.P_nonneg s a) hmax
  have hmul := mul_le_mul_of_nonneg_left hsum M.gamma_nonneg
  simp only [optOp]
  linarith

/-- **L3.** The evaluation operator on state values is a `γ`-contraction. -/
theorem dist_evalOp_le (π : StationaryPolicy S A) (V V' : S → ℝ) :
    dist (M.evalOp π V) (M.evalOp π V') ≤ (M.γ : ℝ) * dist V V' := by
  have hr : 0 ≤ (M.γ : ℝ) * dist V V' := mul_nonneg M.gamma_nonneg dist_nonneg
  rw [dist_pi_le_iff hr]
  intro s
  have key : M.evalOp π V s - M.evalOp π V' s =
      (M.γ : ℝ) * ∑ s', M.transPi π s s' * (V - V') s' := by
    simp only [evalOp, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [Real.dist_eq, key, abs_mul, abs_of_nonneg M.gamma_nonneg, dist_eq_norm]
  exact mul_le_mul_of_nonneg_left
    (abs_sum_mul_le_norm (M.transPi_nonneg π s) (M.sum_transPi π s) (V - V')) M.gamma_nonneg

/-- **L3.** The same fact in Mathlib's language of contractions. -/
theorem contractingWith_evalOp (π : StationaryPolicy S A) :
    ContractingWith M.γ (M.evalOp π) :=
  ⟨M.γ_lt_one, LipschitzWith.of_dist_le_mul fun V V' => M.dist_evalOp_le π V V'⟩

/-- **L3.** The evaluation operator on Q-functions is a `γ`-contraction. -/
theorem dist_evalOpQ_le (π : StationaryPolicy S A) (Q Q' : S → A → ℝ) :
    dist (M.evalOpQ π Q) (M.evalOpQ π Q') ≤ (M.γ : ℝ) * dist Q Q' := by
  have hd : 0 ≤ dist Q Q' := dist_nonneg
  -- the policy average moves by at most `dist Q Q'` at every state
  have havg : ‖π.avg Q - π.avg Q'‖ ≤ dist Q Q' := by
    rw [pi_norm_le_iff_of_nonneg hd]
    intro s'
    have h1 : (π.avg Q - π.avg Q') s' = ∑ a, π.prob s' a * (Q s' - Q' s') a := by
      simp only [Pi.sub_apply, StationaryPolicy.avg, mul_sub, Finset.sum_sub_distrib]
    rw [h1, Real.norm_eq_abs]
    calc |∑ a, π.prob s' a * (Q s' - Q' s') a| ≤ ‖Q s' - Q' s'‖ :=
          abs_sum_mul_le_norm (π.nonneg s') (π.sum_eq_one s') _
      _ = dist (Q s') (Q' s') := (dist_eq_norm _ _).symm
      _ ≤ dist Q Q' := dist_le_pi_dist Q Q' s'
  have hr : 0 ≤ (M.γ : ℝ) * dist Q Q' := mul_nonneg M.gamma_nonneg hd
  rw [dist_pi_le_iff hr]
  intro s
  rw [dist_pi_le_iff hr]
  intro a
  have key : M.evalOpQ π Q s a - M.evalOpQ π Q' s a =
      (M.γ : ℝ) * ∑ s', M.P s a s' * (π.avg Q - π.avg Q') s' := by
    simp only [evalOpQ, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [Real.dist_eq, key, abs_mul, abs_of_nonneg M.gamma_nonneg]
  exact mul_le_mul_of_nonneg_left
    ((abs_sum_mul_le_norm (M.P_nonneg s a) (M.P_sum s a) _).trans havg) M.gamma_nonneg

/-- **L3.** The same fact in Mathlib's language of contractions. -/
theorem contractingWith_evalOpQ (π : StationaryPolicy S A) :
    ContractingWith M.γ (M.evalOpQ π) :=
  ⟨M.γ_lt_one, LipschitzWith.of_dist_le_mul fun Q Q' => M.dist_evalOpQ_le π Q Q'⟩

/-- **L5.** The optimality operator is a `γ`-contraction. Book Lemma 1.10. -/
theorem dist_optOp_le (Q Q' : S → A → ℝ) :
    dist (M.optOp Q) (M.optOp Q') ≤ (M.γ : ℝ) * dist Q Q' := by
  have hd : 0 ≤ dist Q Q' := dist_nonneg
  -- the max over actions moves by at most `dist Q Q'` at every state, by L2
  have hmax : ‖maxQ Q - maxQ Q'‖ ≤ dist Q Q' := by
    rw [pi_norm_le_iff_of_nonneg hd]
    intro s'
    rw [Pi.sub_apply, Real.norm_eq_abs]
    calc |maxQ Q s' - maxQ Q' s'| ≤ ‖Q s' - Q' s'‖ := abs_maxQ_sub_maxQ_le Q Q' s'
      _ = dist (Q s') (Q' s') := (dist_eq_norm _ _).symm
      _ ≤ dist Q Q' := dist_le_pi_dist Q Q' s'
  have hr : 0 ≤ (M.γ : ℝ) * dist Q Q' := mul_nonneg M.gamma_nonneg hd
  rw [dist_pi_le_iff hr]
  intro s
  rw [dist_pi_le_iff hr]
  intro a
  have key : M.optOp Q s a - M.optOp Q' s a =
      (M.γ : ℝ) * ∑ s', M.P s a s' * (maxQ Q - maxQ Q') s' := by
    simp only [optOp, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
    ring
  rw [Real.dist_eq, key, abs_mul, abs_of_nonneg M.gamma_nonneg]
  exact mul_le_mul_of_nonneg_left
    ((abs_sum_mul_le_norm (M.P_nonneg s a) (M.P_sum s a) _).trans hmax) M.gamma_nonneg

/-- **L5.** The same fact in Mathlib's language of contractions. -/
theorem contractingWith_optOp : ContractingWith M.γ M.optOp :=
  ⟨M.γ_lt_one, LipschitzWith.of_dist_le_mul fun Q Q' => M.dist_optOp_le Q Q'⟩

end FiniteMDP

/-! End of step 2. -/

/-! ## Step 3. Comparison principle (L6) -/

/-- **L6.** Comparison principle. If a monotone contraction moves `x` down, its fixed point
lies below `x`. This replaces the matrix-inverse arguments of the book. -/
theorem fixedPoint_le_of_map_le {E : Type*} [MetricSpace E] [CompleteSpace E] [Nonempty E]
    [PartialOrder E] [OrderClosedTopology E] {K : ℝ≥0} {F : E → E}
    (hF : ContractingWith K F) (hmono : Monotone F) {x : E} (hx : F x ≤ x) :
    ContractingWith.fixedPoint F hF ≤ x := by
  -- every iterate of `F` stays below `x`
  have hiter : ∀ n : ℕ, F^[n] x ≤ x := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Function.iterate_succ_apply']
      exact (hmono ih).trans hx
  -- the iterates converge to the fixed point, and being below `x` survives the limit
  exact le_of_tendsto (hF.tendsto_iterate_fixedPoint x) (Filter.Eventually.of_forall hiter)

/-- **L6.** The symmetric comparison principle. If a monotone contraction moves `x` up, its
fixed point lies above `x`. -/
theorem le_fixedPoint_of_le_map {E : Type*} [MetricSpace E] [CompleteSpace E] [Nonempty E]
    [PartialOrder E] [OrderClosedTopology E] {K : ℝ≥0} {F : E → E}
    (hF : ContractingWith K F) (hmono : Monotone F) {x : E} (hx : x ≤ F x) :
    x ≤ ContractingWith.fixedPoint F hF := by
  have hiter : ∀ n : ℕ, x ≤ F^[n] x := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Function.iterate_succ_apply']
      exact hx.trans (hmono ih)
  exact ge_of_tendsto (hF.tendsto_iterate_fixedPoint x) (Filter.Eventually.of_forall hiter)

-- Checks that L6 will apply to value functions and Q-functions in step 4.
example : OrderClosedTopology (S → ℝ) := inferInstance
example : OrderClosedTopology (S → A → ℝ) := inferInstance
example : CompleteSpace (S → A → ℝ) := inferInstance

/-! End of step 3. -/

/-! ## Step 4. Values of a policy (D6, L7, L8, L9) -/

/-- The probabilities of a deterministic policy, by definition. -/
theorem StationaryPolicy.ofDet_prob (d : S → A) (s : S) (a : A) :
    (StationaryPolicy.ofDet d).prob s a = if a = d s then 1 else 0 := rfl

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **D6.** The value of a stationary policy, the unique fixed point of `evalOp`. -/
def value (π : StationaryPolicy S A) : S → ℝ :=
  ContractingWith.fixedPoint (M.evalOp π) (M.contractingWith_evalOp π)

/-- **D6.** The Q-function of a stationary policy. -/
def qValue (π : StationaryPolicy S A) : S → A → ℝ :=
  fun s a => M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s'

/-- Averaging through `transPi` is averaging through `P` and then through the policy. -/
theorem sum_transPi_mul (π : StationaryPolicy S A) (s : S) (V : S → ℝ) :
    ∑ s', M.transPi π s s' * V s' = ∑ a, π.prob s a * ∑ s', M.P s a s' * V s' := by
  unfold transPi
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun s' _ => ?_
  ring

/-- **L7.** Bellman consistency for values. Book Lemma 1.4. -/
theorem evalOp_value (π : StationaryPolicy S A) : M.evalOp π (M.value π) = M.value π :=
  (M.contractingWith_evalOp π).fixedPoint_isFixedPt

/-- **L7.** The value is the policy average of the Q-function. -/
theorem value_eq_avg_qValue (π : StationaryPolicy S A) : M.value π = π.avg (M.qValue π) := by
  funext s
  calc M.value π s = M.evalOp π (M.value π) s := (congrFun (M.evalOp_value π) s).symm
    _ = ∑ a, π.prob s a * M.r s a +
          (M.γ : ℝ) * ∑ a, π.prob s a * ∑ s', M.P s a s' * M.value π s' := by
        rw [← M.sum_transPi_mul π s (M.value π)]
        rfl
    _ = ∑ a, (π.prob s a * M.r s a +
          (M.γ : ℝ) * (π.prob s a * ∑ s', M.P s a s' * M.value π s')) := by
        rw [Finset.sum_add_distrib, Finset.mul_sum]
    _ = π.avg (M.qValue π) s := by
        change _ = ∑ a, π.prob s a * M.qValue π s a
        refine Finset.sum_congr rfl fun a _ => ?_
        change _ = π.prob s a * (M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s')
        ring

/-- **L7.** Bellman consistency for Q-functions. -/
theorem evalOpQ_qValue (π : StationaryPolicy S A) :
    M.evalOpQ π (M.qValue π) = M.qValue π := by
  funext s a
  change M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * π.avg (M.qValue π) s' =
    M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s'
  rw [← M.value_eq_avg_qValue π]

/-- **L7.** For a deterministic policy, the value is the Q-value of its own action. -/
theorem value_ofDet (d : S → A) (s : S) :
    M.value (StationaryPolicy.ofDet d) s = M.qValue (StationaryPolicy.ofDet d) s (d s) := by
  rw [M.value_eq_avg_qValue]
  simp [StationaryPolicy.avg, StationaryPolicy.ofDet_prob, ite_mul]

/-- **L8.** Any solution of the Bellman equation is the value. -/
theorem eq_value_of_evalOp_eq {π : StationaryPolicy S A} {V : S → ℝ}
    (h : M.evalOp π V = V) : V = M.value π :=
  (M.contractingWith_evalOp π).fixedPoint_unique h

/-- **L8.** Any solution of the Q-form Bellman equation is the Q-function. -/
theorem eq_qValue_of_evalOpQ_eq {π : StationaryPolicy S A} {Q : S → A → ℝ}
    (h : M.evalOpQ π Q = Q) : Q = M.qValue π := by
  have hQ := (M.contractingWith_evalOpQ π).fixedPoint_unique h
  have hq := (M.contractingWith_evalOpQ π).fixedPoint_unique (M.evalOpQ_qValue π)
  exact hQ.trans hq.symm

/-- **L8.** Iterating the evaluation operator reaches the value at rate `γ ^ k`. -/
theorem dist_iterate_evalOp_le (π : StationaryPolicy S A) (V₀ : S → ℝ) (k : ℕ) :
    dist ((M.evalOp π)^[k] V₀) (M.value π) ≤ (M.γ : ℝ) ^ k * dist V₀ (M.value π) := by
  have h := ((M.contractingWith_evalOp π).2.iterate k).dist_le_mul V₀ (M.value π)
  rw [Function.iterate_fixed (M.evalOp_value π) k, NNReal.coe_pow] at h
  exact h

/-- **L9.** Value bound. Book Section 1.1.1 and Lemma 2.3. The subtraction
`1 - (M.γ : ℝ)` is written in `ℝ` on purpose, since in `ℝ≥0` it would be truncated. -/
theorem norm_value_le (π : StationaryPolicy S A) :
    ‖M.value π‖ ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  -- from the Bellman equation at every state, `‖V‖ ≤ ‖r‖ + γ ‖V‖`
  have key : ‖M.value π‖ ≤ ‖M.r‖ + (M.γ : ℝ) * ‖M.value π‖ := by
    have hnn : 0 ≤ ‖M.r‖ + (M.γ : ℝ) * ‖M.value π‖ := by positivity
    rw [pi_norm_le_iff_of_nonneg hnn]
    intro s
    have hfix := congrFun (M.evalOp_value π) s
    rw [← hfix]
    have h1 : |M.rewardPi π s| ≤ ‖M.r‖ :=
      (abs_sum_mul_le_norm (π.nonneg s) (π.sum_eq_one s) (M.r s)).trans (norm_le_pi_norm M.r s)
    have h2 : |∑ s', M.transPi π s s' * M.value π s'| ≤ ‖M.value π‖ :=
      abs_sum_mul_le_norm (M.transPi_nonneg π s) (M.sum_transPi π s) (M.value π)
    calc ‖M.evalOp π (M.value π) s‖
        = ‖M.rewardPi π s + (M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π s'‖ := rfl
      _ ≤ ‖M.rewardPi π s‖ + ‖(M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π s'‖ :=
          norm_add_le _ _
      _ = |M.rewardPi π s| + (M.γ : ℝ) * |∑ s', M.transPi π s s' * M.value π s'| := by
          rw [norm_mul, Real.norm_of_nonneg M.gamma_nonneg, Real.norm_eq_abs (M.rewardPi π s),
            Real.norm_eq_abs (∑ s', M.transPi π s s' * M.value π s')]
      _ ≤ ‖M.r‖ + (M.γ : ℝ) * ‖M.value π‖ :=
          add_le_add h1 (mul_le_mul_of_nonneg_left h2 M.gamma_nonneg)
  rw [le_div_iff₀ (by linarith)]
  linarith

/-- **L9.** Q-function bound. -/
theorem norm_qValue_le (π : StationaryPolicy S A) :
    ‖M.qValue π‖ ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  have hV := M.norm_value_le π
  have hbound : 0 ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) := div_nonneg (norm_nonneg _) hpos.le
  rw [pi_norm_le_iff_of_nonneg hbound]
  intro s
  rw [pi_norm_le_iff_of_nonneg hbound]
  intro a
  have h2 : |∑ s', M.P s a s' * M.value π s'| ≤ ‖M.value π‖ :=
    abs_sum_mul_le_norm (M.P_nonneg s a) (M.P_sum s a) (M.value π)
  have hr : ‖M.r s a‖ ≤ ‖M.r‖ := (norm_le_pi_norm (M.r s) a).trans (norm_le_pi_norm M.r s)
  have hB : ‖M.r‖ / (1 - (M.γ : ℝ)) * (1 - (M.γ : ℝ)) = ‖M.r‖ := div_mul_cancel₀ _ hpos.ne'
  calc ‖M.qValue π s a‖
      = ‖M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s'‖ := rfl
    _ ≤ ‖M.r s a‖ + ‖(M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s'‖ := norm_add_le _ _
    _ = ‖M.r s a‖ + (M.γ : ℝ) * |∑ s', M.P s a s' * M.value π s'| := by
        rw [norm_mul, Real.norm_of_nonneg M.gamma_nonneg,
          Real.norm_eq_abs (∑ s', M.P s a s' * M.value π s')]
    _ ≤ ‖M.r‖ + (M.γ : ℝ) * (‖M.r‖ / (1 - (M.γ : ℝ))) :=
        add_le_add hr (mul_le_mul_of_nonneg_left (h2.trans hV) M.gamma_nonneg)
    _ = ‖M.r‖ / (1 - (M.γ : ℝ)) := by linarith

/-- **L9.** With rewards in `[0, 1]`, values lie in `[0, 1 / (1 - γ)]`. Both bounds come
from the comparison principle L6. -/
theorem value_mem_Icc (π : StationaryPolicy S A) (hr : ∀ s a, 0 ≤ M.r s a ∧ M.r s a ≤ 1)
    (s : S) : 0 ≤ M.value π s ∧ M.value π s ≤ 1 / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  -- the reward averaged by the policy also lies in `[0, 1]`
  have hrew : ∀ s, 0 ≤ M.rewardPi π s ∧ M.rewardPi π s ≤ 1 := by
    intro s
    constructor
    · exact Finset.sum_nonneg fun a _ => mul_nonneg (π.nonneg s a) (hr s a).1
    · calc M.rewardPi π s = ∑ a, π.prob s a * M.r s a := rfl
        _ ≤ ∑ a, π.prob s a * 1 :=
            Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hr s a).2 (π.nonneg s a)
        _ = 1 := by simp [π.sum_eq_one s]
  constructor
  · -- lower bound, because `0 ≤ evalOp 0`
    have h0 : (0 : S → ℝ) ≤ M.evalOp π 0 := by
      intro s
      simp only [evalOp, Pi.zero_apply, mul_zero, Finset.sum_const_zero, add_zero]
      exact (hrew s).1
    exact le_fixedPoint_of_le_map (M.contractingWith_evalOp π) (M.monotone_evalOp π) h0 s
  · -- upper bound, because `evalOp c ≤ c` for the constant `c = 1 / (1 - γ)`
    set c : ℝ := 1 / (1 - (M.γ : ℝ)) with hc
    have hcmul : c * (1 - (M.γ : ℝ)) = 1 := by
      rw [hc]
      exact div_mul_cancel₀ 1 hpos.ne'
    have hcγ : 1 + (M.γ : ℝ) * c = c := by linarith
    have h1 : M.evalOp π (fun _ => c) ≤ fun _ => c := by
      intro s
      simp only [evalOp]
      rw [← Finset.sum_mul, M.sum_transPi π s, one_mul]
      linarith [(hrew s).2]
    exact fixedPoint_le_of_map_le (M.contractingWith_evalOp π) (M.monotone_evalOp π) h1 s

end FiniteMDP

/-! End of step 4. -/

/-! ## Step 5. Optimality (D7, T1) -/

/-- Averaging against a deterministic policy picks out the chosen action. -/
theorem StationaryPolicy.avg_ofDet (d : S → A) (Q : S → A → ℝ) (s : S) :
    (StationaryPolicy.ofDet d).avg Q s = Q s (d s) := by
  simp [StationaryPolicy.avg, StationaryPolicy.ofDet_prob, ite_mul]

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **D7.** The optimal Q-function, the unique fixed point of `optOp`. -/
def optQ : S → A → ℝ := ContractingWith.fixedPoint M.optOp M.contractingWith_optOp

/-- **D7.** The optimal value. -/
def optV : S → ℝ := maxQ M.optQ

/-- **D7.** A greedy policy for the optimal Q-function. -/
def optPolicy : S → A := greedy M.optQ

/-- `optQ` solves the Bellman optimality equation. -/
theorem optOp_optQ : M.optOp M.optQ = M.optQ :=
  M.contractingWith_optOp.fixedPoint_isFixedPt

/-- Evaluating the greedy policy of `Q` against `Q` itself is one optimality step. -/
theorem evalOpQ_greedy (Q : S → A → ℝ) :
    M.evalOpQ (StationaryPolicy.ofDet (greedy Q)) Q = M.optOp Q := by
  have h : ∀ s', (StationaryPolicy.ofDet (greedy Q)).avg Q s' = maxQ Q s' := fun s' => by
    rw [StationaryPolicy.avg_ofDet, maxQ_eq_greedy]
  funext s a
  change M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * (StationaryPolicy.ofDet (greedy Q)).avg Q s' =
    M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * maxQ Q s'
  simp only [h]

/-- One evaluation step is at most one optimality step. -/
theorem evalOpQ_le_optOp (π : StationaryPolicy S A) (Q : S → A → ℝ) :
    M.evalOpQ π Q ≤ M.optOp Q := by
  intro s a
  have havg : π.avg Q ≤ maxQ Q := by
    intro s'
    exact π.avg_le_maxQ Q s'
  have hsum : ∑ s', M.P s a s' * π.avg Q s' ≤ ∑ s', M.P s a s' * maxQ Q s' :=
    sum_mul_le_sum_mul (M.P_nonneg s a) havg
  have hmul := mul_le_mul_of_nonneg_left hsum M.gamma_nonneg
  change M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * π.avg Q s' ≤
    M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * maxQ Q s'
  linarith

/-- **T1 (i).** The greedy policy of `optQ` has Q-function `optQ`. Book Theorem 1.8,
restricted to stationary policies. -/
theorem qValue_optPolicy : M.qValue (StationaryPolicy.ofDet M.optPolicy) = M.optQ := by
  symm
  apply M.eq_qValue_of_evalOpQ_eq
  change M.evalOpQ (StationaryPolicy.ofDet (greedy M.optQ)) M.optQ = M.optQ
  rw [M.evalOpQ_greedy, M.optOp_optQ]

/-- **T1 (i).** The same statement for values. -/
theorem value_optPolicy : M.value (StationaryPolicy.ofDet M.optPolicy) = M.optV := by
  funext s
  rw [M.value_ofDet, M.qValue_optPolicy]
  exact (maxQ_eq_greedy M.optQ s).symm

/-- **T1 (ii).** No stationary policy beats `optQ`. Proved with the comparison principle L6. -/
theorem qValue_le_optQ (π : StationaryPolicy S A) : M.qValue π ≤ M.optQ := by
  have hx : M.evalOpQ π M.optQ ≤ M.optQ := by
    have h := M.evalOpQ_le_optOp π M.optQ
    rwa [M.optOp_optQ] at h
  have hle := fixedPoint_le_of_map_le (M.contractingWith_evalOpQ π) (M.monotone_evalOpQ π) hx
  have heq : M.qValue π =
      ContractingWith.fixedPoint (M.evalOpQ π) (M.contractingWith_evalOpQ π) :=
    (M.contractingWith_evalOpQ π).fixedPoint_unique (M.evalOpQ_qValue π)
  rw [heq]
  exact hle

/-- **T1 (ii).** The same statement for values. -/
theorem value_le_optV (π : StationaryPolicy S A) : M.value π ≤ M.optV := by
  intro s
  rw [M.value_eq_avg_qValue]
  calc π.avg (M.qValue π) s ≤ maxQ (M.qValue π) s := π.avg_le_maxQ _ s
    _ ≤ maxQ M.optQ s := by
        unfold maxQ
        rw [Finset.sup'_le_iff]
        intro a _
        exact (M.qValue_le_optQ π s a).trans (le_maxQ M.optQ s a)

/-- **T1 (iii).** `optQ` is the only fixed point of `optOp`. -/
theorem eq_optQ_of_optOp_eq {Q : S → A → ℝ} (h : M.optOp Q = Q) : Q = M.optQ :=
  M.contractingWith_optOp.fixedPoint_unique h

/-- The optimal Q-function obeys the same bound as every Q-function. -/
theorem norm_optQ_le : ‖M.optQ‖ ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) := by
  rw [← M.qValue_optPolicy]
  exact M.norm_qValue_le _

end FiniteMDP

/-! End of step 5. -/

/-! ## Step 6. Error amplification and value iteration (L10, T2) -/

/-- Two Q-functions that are close in sup norm are close at every state and action. -/
theorem abs_sub_apply_le_dist (Q Q' : S → A → ℝ) (s : S) (a : A) :
    |Q s a - Q' s a| ≤ dist Q Q' := by
  rw [← Real.dist_eq]
  exact (dist_le_pi_dist (Q s) (Q' s) a).trans (dist_le_pi_dist Q Q' s)

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- `optQ` written through `optV`, from the Bellman optimality equation. -/
theorem optQ_eq (s : S) (a : A) :
    M.optQ s a = M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.optV s' := by
  have h := congrFun (congrFun M.optOp_optQ s) a
  rw [← h]
  rfl

/-- The greedy action of `Q` is `2 ε`-optimal for `optQ`, where `ε = dist Q optQ`. -/
theorem optV_le_optQ_greedy (Q : S → A → ℝ) (s : S) :
    M.optV s ≤ M.optQ s (greedy Q s) + 2 * dist Q M.optQ := by
  have h := abs_maxQ_sub_maxQ_le M.optQ Q s
  have h' : ‖M.optQ s - Q s‖ ≤ dist Q M.optQ := by
    rw [← dist_eq_norm, dist_comm (M.optQ s) (Q s)]
    exact dist_le_pi_dist Q M.optQ s
  have hmax : maxQ M.optQ s - maxQ Q s ≤ dist Q M.optQ := (le_abs_self _).trans (h.trans h')
  have hg : maxQ Q s = Q s (greedy Q s) := maxQ_eq_greedy Q s
  have hq := (abs_sub_le_iff.mp (abs_sub_apply_le_dist Q M.optQ s (greedy Q s))).1
  change maxQ M.optQ s ≤ M.optQ s (greedy Q s) + 2 * dist Q M.optQ
  linarith

/-- One Bellman step links `optQ` and the value of the greedy policy at the greedy action. -/
theorem optQ_greedy_sub_value (Q : S → A → ℝ) (s : S) :
    M.optQ s (greedy Q s) - M.value (StationaryPolicy.ofDet (greedy Q)) s =
      (M.γ : ℝ) * ∑ s', M.P s (greedy Q s) s' *
        (M.optV s' - M.value (StationaryPolicy.ofDet (greedy Q)) s') := by
  rw [M.optQ_eq, M.value_ofDet]
  change M.r s (greedy Q s) + (M.γ : ℝ) * ∑ s', M.P s (greedy Q s) s' * M.optV s' -
      (M.r s (greedy Q s) + (M.γ : ℝ) * ∑ s', M.P s (greedy Q s) s' *
        M.value (StationaryPolicy.ofDet (greedy Q)) s') = _
  simp only [mul_sub, Finset.sum_sub_distrib]
  ring

/-- **L10.** Error amplification for greedy policies. Book Lemma 1.11. -/
theorem optV_sub_le_value_greedy (Q : S → A → ℝ) (s : S) :
    M.optV s - 2 * dist Q M.optQ / (1 - (M.γ : ℝ)) ≤
      M.value (StationaryPolicy.ofDet (greedy Q)) s := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  -- the gap `optV - value` is nonnegative, by T1
  have hD_nonneg : ∀ s, 0 ≤ (M.optV - M.value (StationaryPolicy.ofDet (greedy Q))) s :=
    fun s => sub_nonneg.mpr (M.value_le_optV _ s)
  -- at every state the gap is at most `2 ε + γ ‖gap‖`
  have hstep : ∀ s, (M.optV - M.value (StationaryPolicy.ofDet (greedy Q))) s ≤
      2 * dist Q M.optQ +
        (M.γ : ℝ) * ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ := by
    intro s
    have h1 := M.optV_le_optQ_greedy Q s
    have h2 := M.optQ_greedy_sub_value Q s
    have h3 : ∑ s', M.P s (greedy Q s) s' *
        (M.optV s' - M.value (StationaryPolicy.ofDet (greedy Q)) s') ≤
          ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ :=
      (le_abs_self _).trans (abs_sum_mul_le_norm (M.P_nonneg s (greedy Q s))
        (M.P_sum s (greedy Q s)) (M.optV - M.value (StationaryPolicy.ofDet (greedy Q))))
    have h4 := mul_le_mul_of_nonneg_left h3 M.gamma_nonneg
    rw [Pi.sub_apply]
    linarith
  -- so `‖gap‖ ≤ 2 ε + γ ‖gap‖`
  have hnorm : ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ ≤
      2 * dist Q M.optQ +
        (M.γ : ℝ) * ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ := by
    have hnn : 0 ≤ 2 * dist Q M.optQ +
        (M.γ : ℝ) * ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ := by positivity
    rw [pi_norm_le_iff_of_nonneg hnn]
    intro s
    rw [Real.norm_of_nonneg (hD_nonneg s)]
    exact hstep s
  have hDs : (M.optV - M.value (StationaryPolicy.ofDet (greedy Q))) s ≤
      ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ := by
    have h := norm_le_pi_norm (M.optV - M.value (StationaryPolicy.ofDet (greedy Q))) s
    rw [Real.norm_of_nonneg (hD_nonneg s)] at h
    exact h
  have hfinal : ‖M.optV - M.value (StationaryPolicy.ofDet (greedy Q))‖ ≤
      2 * dist Q M.optQ / (1 - (M.γ : ℝ)) := by
    rw [le_div_iff₀ hpos]
    linarith
  rw [Pi.sub_apply] at hDs
  linarith

/-- Value iteration started from zero. -/
def valueIteration (k : ℕ) : S → A → ℝ := M.optOp^[k] 0

/-- **T2.** Value iteration converges geometrically. Book Theorem 1.12. -/
theorem dist_valueIteration_le (k : ℕ) :
    dist (M.valueIteration k) M.optQ ≤ (M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ)) := by
  have h := (M.contractingWith_optOp.2.iterate k).dist_le_mul 0 M.optQ
  rw [Function.iterate_fixed M.optOp_optQ k, NNReal.coe_pow] at h
  have h0 : dist (0 : S → A → ℝ) M.optQ ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) := by
    rw [dist_comm, dist_zero_right]
    exact M.norm_optQ_le
  calc dist (M.valueIteration k) M.optQ ≤ (M.γ : ℝ) ^ k * dist 0 M.optQ := h
    _ ≤ (M.γ : ℝ) ^ k * (‖M.r‖ / (1 - (M.γ : ℝ))) :=
        mul_le_mul_of_nonneg_left h0 (pow_nonneg M.gamma_nonneg k)
    _ = (M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ)) := by ring

/-- **T2.** Once `2 γ^k ‖r‖ ≤ (1 - γ)^2 ε`, the greedy policy of `valueIteration k` is
`ε`-optimal. -/
theorem value_greedy_valueIteration_ge_of_le {ε : ℝ} (k : ℕ)
    (hk : 2 * (M.γ : ℝ) ^ k * ‖M.r‖ ≤ (1 - (M.γ : ℝ)) ^ 2 * ε) (s : S) :
    M.optV s - ε ≤ M.value (StationaryPolicy.ofDet (greedy (M.valueIteration k))) s := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  have hL10 := M.optV_sub_le_value_greedy (M.valueIteration k) s
  have hdist := M.dist_valueIteration_le k
  have hsmall : 2 * dist (M.valueIteration k) M.optQ / (1 - (M.γ : ℝ)) ≤ ε := by
    rw [div_le_iff₀ hpos]
    have h2 : 2 * dist (M.valueIteration k) M.optQ ≤
        2 * ((M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ))) := by linarith
    have h3 : 2 * ((M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ))) ≤ ε * (1 - (M.γ : ℝ)) := by
      rw [show 2 * ((M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ))) =
          2 * (M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ)) by ring, div_le_iff₀ hpos]
      linarith
    linarith
  linarith

/-- **T2.** The iteration count of the book. With
`k ≥ log (2 ‖r‖ / ((1 - γ)^2 ε)) / (1 - γ)`, the greedy policy of `valueIteration k`
is `ε`-optimal. -/
theorem value_greedy_valueIteration_ge {ε : ℝ} (hε : 0 < ε) (k : ℕ)
    (hk : Real.log (2 * ‖M.r‖ / ((1 - (M.γ : ℝ)) ^ 2 * ε)) / (1 - (M.γ : ℝ)) ≤ (k : ℝ))
    (s : S) :
    M.optV s - ε ≤ M.value (StationaryPolicy.ofDet (greedy (M.valueIteration k))) s := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  refine M.value_greedy_valueIteration_ge_of_le k ?_ s
  have hc : 0 < (1 - (M.γ : ℝ)) ^ 2 * ε := by positivity
  rcases (norm_nonneg M.r).eq_or_lt with hr | hr
  · -- no reward at all, so the left side is zero
    rw [← hr, mul_zero]
    exact hc.le
  · set c := (1 - (M.γ : ℝ)) ^ 2 * ε with hc_def
    set R := ‖M.r‖ with hR_def
    have hX : 0 < 2 * R / c := div_pos (by linarith) hc
    -- `γ^k ≤ exp (k (γ - 1))`, since `γ ≤ exp (γ - 1)`
    have h1 : (M.γ : ℝ) ^ k ≤ Real.exp (k * ((M.γ : ℝ) - 1)) := by
      rw [Real.exp_nat_mul]
      gcongr
      linarith [Real.add_one_le_exp ((M.γ : ℝ) - 1)]
    -- `exp (k (γ - 1)) ≤ c / (2 R)`, by the choice of `k`
    have h2 : Real.exp (k * ((M.γ : ℝ) - 1)) ≤ c / (2 * R) := by
      have hlog : Real.log (2 * R / c) ≤ (1 - (M.γ : ℝ)) * k := by
        rw [div_le_iff₀ hpos] at hk
        linarith
      calc Real.exp (k * ((M.γ : ℝ) - 1)) ≤ Real.exp (-Real.log (2 * R / c)) :=
            Real.exp_le_exp.mpr (by linarith)
        _ = c / (2 * R) := by rw [Real.exp_neg, Real.exp_log hX, inv_div]
    have h2R : 2 * R ≠ 0 := (by linarith : (0 : ℝ) < 2 * R).ne'
    calc 2 * (M.γ : ℝ) ^ k * R ≤ 2 * (c / (2 * R)) * R :=
          mul_le_mul_of_nonneg_right
            (mul_le_mul_of_nonneg_left (h1.trans h2) (by norm_num : (0 : ℝ) ≤ 2)) hr.le
      _ = c / (2 * R) * (2 * R) := by ring
      _ = c := div_mul_cancel₀ c h2R

end FiniteMDP

/-! End of step 6. -/

/-! ## Step 7. Policy iteration (L11, T3) -/

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- One step of policy iteration, the greedy policy of the current Q-function. -/
def piStep (d : S → A) : S → A := greedy (M.qValue (StationaryPolicy.ofDet d))

/-- Policy iteration started from `d₀`. -/
def policyIteration (d₀ : S → A) (k : ℕ) : S → A := M.piStep^[k] d₀

theorem policyIteration_succ (d₀ : S → A) (k : ℕ) :
    M.policyIteration d₀ (k + 1) = M.piStep (M.policyIteration d₀ k) := by
  unfold policyIteration
  rw [Function.iterate_succ_apply']

/-- **L11.** One optimality step does not decrease a Q-function. Book Lemma 1.13, part 1. -/
theorem qValue_le_optOp_qValue (π : StationaryPolicy S A) :
    M.qValue π ≤ M.optOp (M.qValue π) := by
  have h := M.evalOpQ_le_optOp π (M.qValue π)
  rwa [M.evalOpQ_qValue] at h

/-- **L11.** The improved policy is at least one optimality step better. The proof uses the
comparison principle L6. -/
theorem optOp_qValue_le_qValue_piStep (d : S → A) :
    M.optOp (M.qValue (StationaryPolicy.ofDet d)) ≤
      M.qValue (StationaryPolicy.ofDet (M.piStep d)) := by
  have hT : M.evalOpQ (StationaryPolicy.ofDet (M.piStep d)) (M.qValue (StationaryPolicy.ofDet d)) =
      M.optOp (M.qValue (StationaryPolicy.ofDet d)) :=
    M.evalOpQ_greedy (M.qValue (StationaryPolicy.ofDet d))
  have hQ : M.qValue (StationaryPolicy.ofDet d) ≤
      M.evalOpQ (StationaryPolicy.ofDet (M.piStep d)) (M.qValue (StationaryPolicy.ofDet d)) := by
    rw [hT]
    exact M.qValue_le_optOp_qValue _
  have hfix : M.qValue (StationaryPolicy.ofDet (M.piStep d)) =
      ContractingWith.fixedPoint (M.evalOpQ (StationaryPolicy.ofDet (M.piStep d)))
        (M.contractingWith_evalOpQ _) :=
    (M.contractingWith_evalOpQ _).fixedPoint_unique (M.evalOpQ_qValue _)
  have hle : M.qValue (StationaryPolicy.ofDet d) ≤
      M.qValue (StationaryPolicy.ofDet (M.piStep d)) := by
    rw [hfix]
    exact le_fixedPoint_of_le_map (M.contractingWith_evalOpQ _) (M.monotone_evalOpQ _) hQ
  calc M.optOp (M.qValue (StationaryPolicy.ofDet d))
      = M.evalOpQ (StationaryPolicy.ofDet (M.piStep d))
          (M.qValue (StationaryPolicy.ofDet d)) := hT.symm
    _ ≤ M.evalOpQ (StationaryPolicy.ofDet (M.piStep d))
          (M.qValue (StationaryPolicy.ofDet (M.piStep d))) := M.monotone_evalOpQ _ hle
    _ = M.qValue (StationaryPolicy.ofDet (M.piStep d)) := M.evalOpQ_qValue _

/-- **T3.** Each round of policy iteration shrinks the gap to `optQ` by `γ`.
Book Lemma 1.13, part 2. -/
theorem dist_policyIteration_succ_le (d₀ : S → A) (k : ℕ) :
    dist M.optQ (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ (k + 1)))) ≤
      (M.γ : ℝ) * dist M.optQ (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ k))) := by
  rw [M.policyIteration_succ]
  set d := M.policyIteration d₀ k
  -- the new Q-function sits between one optimality step and `optQ`
  have hlow := M.optOp_qValue_le_qValue_piStep d
  have hup := M.qValue_le_optQ (StationaryPolicy.ofDet (M.piStep d))
  have hcon := M.dist_optOp_le M.optQ (M.qValue (StationaryPolicy.ofDet d))
  rw [M.optOp_optQ] at hcon
  refine le_trans ?_ hcon
  have hr : 0 ≤ dist M.optQ (M.optOp (M.qValue (StationaryPolicy.ofDet d))) := dist_nonneg
  rw [dist_pi_le_iff hr]
  intro s
  rw [dist_pi_le_iff hr]
  intro a
  rw [Real.dist_eq, abs_of_nonneg (sub_nonneg.mpr (hup s a))]
  have h1 := abs_sub_apply_le_dist M.optQ (M.optOp (M.qValue (StationaryPolicy.ofDet d))) s a
  have h2 := hlow s a
  have h3 := le_abs_self (M.optQ s a - M.optOp (M.qValue (StationaryPolicy.ofDet d)) s a)
  linarith

/-- **T3.** Policy iteration converges geometrically. Book Theorem 1.14. -/
theorem dist_policyIteration_le (d₀ : S → A) (k : ℕ) :
    dist M.optQ (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ k))) ≤
      2 * (M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ)) := by
  induction k with
  | zero =>
    -- both Q-functions are bounded by `‖r‖ / (1 - γ)`
    have h1 := M.norm_optQ_le
    have h2 := M.norm_qValue_le (StationaryPolicy.ofDet (M.policyIteration d₀ 0))
    calc dist M.optQ (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ 0)))
        ≤ ‖M.optQ‖ + ‖M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ 0))‖ :=
          dist_le_norm_add_norm _ _
      _ ≤ ‖M.r‖ / (1 - (M.γ : ℝ)) + ‖M.r‖ / (1 - (M.γ : ℝ)) := add_le_add h1 h2
      _ = 2 * (M.γ : ℝ) ^ 0 * ‖M.r‖ / (1 - (M.γ : ℝ)) := by ring
  | succ k ih =>
    calc dist M.optQ (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ (k + 1))))
        ≤ (M.γ : ℝ) * dist M.optQ
            (M.qValue (StationaryPolicy.ofDet (M.policyIteration d₀ k))) :=
          M.dist_policyIteration_succ_le d₀ k
      _ ≤ (M.γ : ℝ) * (2 * (M.γ : ℝ) ^ k * ‖M.r‖ / (1 - (M.γ : ℝ))) :=
          mul_le_mul_of_nonneg_left ih M.gamma_nonneg
      _ = 2 * (M.γ : ℝ) ^ (k + 1) * ‖M.r‖ / (1 - (M.γ : ℝ)) := by ring

end FiniteMDP

/-! End of step 7. -/

/-! ## Step 8. Performance difference (D8, L12, L13) -/

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **D8.** The advantage function. Book Section 1.5. -/
def advantage (π : StationaryPolicy S A) : S → A → ℝ :=
  fun s a => M.qValue π s a - M.value π s

/-- Averaging one policy's Q-function under another policy is one evaluation step. -/
theorem avg_qValue_eq_evalOp (π π' : StationaryPolicy S A) (s : S) :
    π.avg (M.qValue π') s = M.evalOp π (M.value π') s := by
  change ∑ a, π.prob s a * (M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π' s') =
    ∑ a, π.prob s a * M.r s a + (M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π' s'
  rw [M.sum_transPi_mul π s (M.value π'), Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun a _ => ?_
  ring

/-- **L12.** Performance difference, value form. Book Lemma 1.16 without occupancy measures.
The difference of two values is the value of the first policy once the reward is replaced
by the advantage of the second. -/
theorem value_sub_value_eq (π π' : StationaryPolicy S A) :
    M.value π - M.value π' = (M.withReward (M.advantage π')).value π := by
  apply (M.withReward (M.advantage π')).eq_value_of_evalOp_eq
  funext s
  have h1 : M.rewardPi π s + (M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π s' =
      M.value π s := congrFun (M.evalOp_value π) s
  have h2 := M.avg_qValue_eq_evalOp π π' s
  have hsplit : (M.withReward (M.advantage π')).rewardPi π s =
      π.avg (M.qValue π') s - M.value π' s := by
    change ∑ a, π.prob s a * (M.qValue π' s a - M.value π' s) = _
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, π.sum_eq_one s, one_mul]
    rfl
  have h3 : ∑ s', M.transPi π s s' * (M.value π - M.value π') s' =
      ∑ s', M.transPi π s s' * M.value π s' - ∑ s', M.transPi π s s' * M.value π' s' := by
    simp only [Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
  have h4 : M.evalOp π (M.value π') s =
      M.rewardPi π s + (M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π' s' := rfl
  change (M.withReward (M.advantage π')).rewardPi π s +
      (M.γ : ℝ) * ∑ s', M.transPi π s s' * (M.value π - M.value π') s' =
    M.value π s - M.value π' s
  rw [hsplit, h2, h3, h4]
  linarith

/-- **L13.** A policy whose averaged advantage over `π'` is nonnegative is at least as good
as `π'`. -/
theorem value_le_value_of_avg_advantage_nonneg {π π' : StationaryPolicy S A}
    (h : 0 ≤ π.avg (M.advantage π')) : M.value π' ≤ M.value π := by
  have hpd := M.value_sub_value_eq π π'
  -- the value of `π` with reward `advantage π'` is nonnegative, by L6
  have hnn : 0 ≤ (M.withReward (M.advantage π')).value π := by
    have h0 : (0 : S → ℝ) ≤ (M.withReward (M.advantage π')).evalOp π 0 := by
      intro s
      simp only [evalOp, Pi.zero_apply, mul_zero, Finset.sum_const_zero, add_zero]
      exact h s
    exact le_fixedPoint_of_le_map ((M.withReward (M.advantage π')).contractingWith_evalOp π)
      ((M.withReward (M.advantage π')).monotone_evalOp π) h0
  intro s
  have h1 : M.value π s - M.value π' s = (M.withReward (M.advantage π')).value π s :=
    congrFun hpd s
  have h2 : 0 ≤ (M.withReward (M.advantage π')).value π s := hnn s
  linarith

/-- **L13.** The greedy policy of `qValue π` improves on `π`. -/
theorem value_le_value_greedy (π : StationaryPolicy S A) :
    M.value π ≤ M.value (StationaryPolicy.ofDet (greedy (M.qValue π))) := by
  apply M.value_le_value_of_avg_advantage_nonneg
  intro s
  rw [StationaryPolicy.avg_ofDet]
  change 0 ≤ M.qValue π s (greedy (M.qValue π) s) - M.value π s
  have hV : M.value π s = π.avg (M.qValue π) s := congrFun (M.value_eq_avg_qValue π) s
  have h1 := π.avg_le_maxQ (M.qValue π) s
  have h2 := maxQ_eq_greedy (M.qValue π) s
  linarith

end FiniteMDP

/-! End of step 8. -/

/-! ## Step 9. Simulation lemma and planning in a perturbed model (L14, L15) -/

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **L14.** Simulation lemma, operator form. Book Lemmas 2.2 and 2.3. The MDP `M'` has the
same rewards and discount as `M` and its own transitions. -/
theorem qValue_sub_qValue_eq (M' : FiniteMDP S A) (hr : M'.r = M.r) (hγ : M'.γ = M.γ)
    (π : StationaryPolicy S A) :
    M.qValue π - M'.qValue π =
      (M'.withReward fun s a =>
        (M.γ : ℝ) * ∑ s', (M.P s a s' - M'.P s a s') * M.value π s').qValue π := by
  apply (M'.withReward fun s a =>
    (M.γ : ℝ) * ∑ s', (M.P s a s' - M'.P s a s') * M.value π s').eq_qValue_of_evalOpQ_eq
  funext s a
  have havg : ∀ s', π.avg (M.qValue π - M'.qValue π) s' = M.value π s' - M'.value π s' := by
    intro s'
    rw [congrFun (M.value_eq_avg_qValue π) s', congrFun (M'.value_eq_avg_qValue π) s']
    simp only [StationaryPolicy.avg, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
  change (M.γ : ℝ) * ∑ s', (M.P s a s' - M'.P s a s') * M.value π s' +
      (M'.γ : ℝ) * ∑ s', M'.P s a s' * π.avg (M.qValue π - M'.qValue π) s' =
    M.qValue π s a - M'.qValue π s a
  simp only [havg]
  change _ = (M.r s a + (M.γ : ℝ) * ∑ s', M.P s a s' * M.value π s') -
    (M'.r s a + (M'.γ : ℝ) * ∑ s', M'.P s a s' * M'.value π s')
  rw [hr, hγ]
  simp only [sub_mul, mul_sub, Finset.sum_sub_distrib]
  ring

/-- **L14.** The resulting bound in sup norm. -/
theorem norm_qValue_sub_qValue_le (M' : FiniteMDP S A) (hr : M'.r = M.r) (hγ : M'.γ = M.γ)
    (π : StationaryPolicy S A) {δ : ℝ} (hδ : ∀ s a, ∑ s', |M.P s a s' - M'.P s a s'| ≤ δ) :
    ‖M.qValue π - M'.qValue π‖ ≤ (M.γ : ℝ) / (1 - (M.γ : ℝ)) * δ * ‖M.value π‖ := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  rw [M.qValue_sub_qValue_eq M' hr hγ π]
  set b : S → A → ℝ := fun s a =>
    (M.γ : ℝ) * ∑ s', (M.P s a s' - M'.P s a s') * M.value π s' with hb_def
  -- `δ ‖V‖` is nonnegative, also when there are no states
  have hδV : 0 ≤ δ * ‖M.value π‖ := by
    rcases isEmpty_or_nonempty S with hS | ⟨⟨s0⟩⟩
    · have hV0 : M.value π = 0 := Subsingleton.elim _ _
      rw [hV0, norm_zero, mul_zero]
    · obtain ⟨a0⟩ := ‹Nonempty A›
      have h0 : 0 ≤ δ := (Finset.sum_nonneg fun s' _ => abs_nonneg _).trans (hδ s0 a0)
      exact mul_nonneg h0 (norm_nonneg _)
  -- the new reward is at most `γ δ ‖V‖`
  have hb : ‖b‖ ≤ (M.γ : ℝ) * δ * ‖M.value π‖ := by
    have hnn : 0 ≤ (M.γ : ℝ) * δ * ‖M.value π‖ := by
      rw [mul_assoc]
      exact mul_nonneg M.gamma_nonneg hδV
    rw [pi_norm_le_iff_of_nonneg hnn]
    intro s
    rw [pi_norm_le_iff_of_nonneg hnn]
    intro a
    change ‖(M.γ : ℝ) * ∑ s', (M.P s a s' - M'.P s a s') * M.value π s'‖ ≤
      (M.γ : ℝ) * δ * ‖M.value π‖
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg M.gamma_nonneg, mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ M.gamma_nonneg
    calc |∑ s', (M.P s a s' - M'.P s a s') * M.value π s'|
        ≤ ∑ s', |(M.P s a s' - M'.P s a s') * M.value π s'| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ s', |M.P s a s' - M'.P s a s'| * ‖M.value π‖ := by
          refine Finset.sum_le_sum fun s' _ => ?_
          rw [abs_mul]
          refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
          have h := norm_le_pi_norm (M.value π) s'
          rwa [Real.norm_eq_abs] at h
      _ = (∑ s', |M.P s a s' - M'.P s a s'|) * ‖M.value π‖ := by rw [Finset.sum_mul]
      _ ≤ δ * ‖M.value π‖ := mul_le_mul_of_nonneg_right (hδ s a) (norm_nonneg _)
  -- L9 in the MDP with reward `b`
  have hQ : ‖(M'.withReward b).qValue π‖ ≤ ‖b‖ / (1 - (M'.γ : ℝ)) :=
    (M'.withReward b).norm_qValue_le π
  rw [hγ] at hQ
  calc ‖(M'.withReward b).qValue π‖ ≤ ‖b‖ / (1 - (M.γ : ℝ)) := hQ
    _ ≤ (M.γ : ℝ) * δ * ‖M.value π‖ / (1 - (M.γ : ℝ)) := by
        rw [div_eq_mul_inv, div_eq_mul_inv]
        exact mul_le_mul_of_nonneg_right hb (inv_nonneg.mpr hpos.le)
    _ = (M.γ : ℝ) / (1 - (M.γ : ℝ)) * δ * ‖M.value π‖ := by ring

/-- **L15.** Planning in a perturbed model, optimal Q-functions. Book Proposition 2.1, third
bullet. Accuracy is only needed for the two optimal policies. -/
theorem dist_optQ_le_of_dist_le (M' : FiniteMDP S A) {ε : ℝ}
    (h₁ : dist (M.qValue (StationaryPolicy.ofDet M.optPolicy))
      (M'.qValue (StationaryPolicy.ofDet M.optPolicy)) ≤ ε)
    (h₂ : dist (M.qValue (StationaryPolicy.ofDet M'.optPolicy))
      (M'.qValue (StationaryPolicy.ofDet M'.optPolicy)) ≤ ε) :
    dist M.optQ M'.optQ ≤ ε := by
  have hε : 0 ≤ ε := dist_nonneg.trans h₁
  rw [dist_pi_le_iff hε]
  intro s
  rw [dist_pi_le_iff hε]
  intro a
  rw [Real.dist_eq, abs_sub_le_iff]
  have hq1 := congrFun (congrFun M.qValue_optPolicy s) a
  have hq2 := congrFun (congrFun M'.qValue_optPolicy s) a
  have hle1 := M'.qValue_le_optQ (StationaryPolicy.ofDet M.optPolicy) s a
  have hle2 := M.qValue_le_optQ (StationaryPolicy.ofDet M'.optPolicy) s a
  have hd1 := abs_sub_le_iff.mp (abs_sub_apply_le_dist
    (M.qValue (StationaryPolicy.ofDet M.optPolicy))
    (M'.qValue (StationaryPolicy.ofDet M.optPolicy)) s a)
  have hd2 := abs_sub_le_iff.mp (abs_sub_apply_le_dist
    (M.qValue (StationaryPolicy.ofDet M'.optPolicy))
    (M'.qValue (StationaryPolicy.ofDet M'.optPolicy)) s a)
  constructor
  · linarith [hd1.1]
  · linarith [hd2.2]

/-- **L15.** The optimal policy of `M'` is `2 ε`-optimal in `M`. -/
theorem dist_optQ_qValue_le_of_dist_le (M' : FiniteMDP S A) {ε : ℝ}
    (h₁ : dist (M.qValue (StationaryPolicy.ofDet M.optPolicy))
      (M'.qValue (StationaryPolicy.ofDet M.optPolicy)) ≤ ε)
    (h₂ : dist (M.qValue (StationaryPolicy.ofDet M'.optPolicy))
      (M'.qValue (StationaryPolicy.ofDet M'.optPolicy)) ≤ ε) :
    dist M.optQ (M.qValue (StationaryPolicy.ofDet M'.optPolicy)) ≤ 2 * ε := by
  calc dist M.optQ (M.qValue (StationaryPolicy.ofDet M'.optPolicy))
      ≤ dist M.optQ M'.optQ + dist M'.optQ (M.qValue (StationaryPolicy.ofDet M'.optPolicy)) :=
        dist_triangle _ _ _
    _ ≤ ε + ε := by
        refine add_le_add (M.dist_optQ_le_of_dist_le M' h₁ h₂) ?_
        rw [← M'.qValue_optPolicy, dist_comm]
        exact h₂
    _ = 2 * ε := by ring

end FiniteMDP

/-! End of step 9. All 26 nodes of priority 1 are proved. -/








end Learning
