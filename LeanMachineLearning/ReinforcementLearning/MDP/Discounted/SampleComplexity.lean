/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Melissa Monfared
-/
import LeanMachineLearning.ReinforcementLearning.MDP.Discounted.Planning

/-!
# Sample complexity with a generative model

Priority 2 of the RL plan. This file grows one step at a time, like the planning file.

Step 1 contains R2 and R3, the deterministic half. They turn accuracy of the model in
l1 into accuracy of every Q-function and into near-optimal planning.
-/

set_option linter.unusedSectionVars false

open scoped NNReal

noncomputable section

namespace Learning

variable {S A : Type*} [Fintype S] [Fintype A] [DecidableEq A] [Nonempty A]

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **R2.** Uniform value accuracy from model accuracy. Book proof of Proposition 2.1,
second bullet. `M'` has the same rewards and discount as `M` and its own transitions. -/
theorem norm_qValue_sub_le_of_l1_le [Nonempty S] (M' : FiniteMDP S A) (hr : M'.r = M.r)
    (hγ : M'.γ = M.γ) {δ : ℝ} (hδ : ∀ s a, ∑ s', |M.P s a s' - M'.P s a s'| ≤ δ)
    (π : StationaryPolicy S A) :
    ‖M.qValue π - M'.qValue π‖ ≤ (M.γ : ℝ) * δ * ‖M.r‖ / (1 - (M.γ : ℝ)) ^ 2 := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  have hδ0 : 0 ≤ δ := by
    obtain ⟨s0⟩ := ‹Nonempty S›
    obtain ⟨a0⟩ := ‹Nonempty A›
    exact (Finset.sum_nonneg fun s' _ => abs_nonneg _).trans (hδ s0 a0)
  have h14 := M.norm_qValue_sub_qValue_le M' hr hγ π hδ
  have h9 := M.norm_value_le π
  calc ‖M.qValue π - M'.qValue π‖
      ≤ (M.γ : ℝ) / (1 - (M.γ : ℝ)) * δ * ‖M.value π‖ := h14
    _ ≤ (M.γ : ℝ) / (1 - (M.γ : ℝ)) * δ * (‖M.r‖ / (1 - (M.γ : ℝ))) := by
        apply mul_le_mul_of_nonneg_left h9
        exact mul_nonneg (div_nonneg M.gamma_nonneg hpos.le) hδ0
    _ = (M.γ : ℝ) * δ * ‖M.r‖ / (1 - (M.γ : ℝ)) ^ 2 := by
        rw [div_mul_eq_mul_div, div_mul_div_comm, ← sq]

/-- **R3.** Model accuracy gives accuracy of the optimal Q-function. Book Proposition 2.1,
third bullet. -/
theorem dist_optQ_le_of_l1_le [Nonempty S] (M' : FiniteMDP S A) (hr : M'.r = M.r)
    (hγ : M'.γ = M.γ) {δ : ℝ} (hδ : ∀ s a, ∑ s', |M.P s a s' - M'.P s a s'| ≤ δ) :
    dist M.optQ M'.optQ ≤ (M.γ : ℝ) * δ * ‖M.r‖ / (1 - (M.γ : ℝ)) ^ 2 :=
  M.dist_optQ_le_of_dist_le M'
    (by rw [dist_eq_norm]; exact M.norm_qValue_sub_le_of_l1_le M' hr hγ hδ _)
    (by rw [dist_eq_norm]; exact M.norm_qValue_sub_le_of_l1_le M' hr hγ hδ _)

/-- **R3.** The optimal policy of the perturbed model is nearly optimal in `M`. -/
theorem dist_optQ_qValue_le_of_l1_le [Nonempty S] (M' : FiniteMDP S A) (hr : M'.r = M.r)
    (hγ : M'.γ = M.γ) {δ : ℝ} (hδ : ∀ s a, ∑ s', |M.P s a s' - M'.P s a s'| ≤ δ) :
    dist M.optQ (M.qValue (StationaryPolicy.ofDet M'.optPolicy)) ≤
      2 * ((M.γ : ℝ) * δ * ‖M.r‖ / (1 - (M.γ : ℝ)) ^ 2) :=
  M.dist_optQ_qValue_le_of_dist_le M'
    (by rw [dist_eq_norm]; exact M.norm_qValue_sub_le_of_l1_le M' hr hγ hδ _)
    (by rw [dist_eq_norm]; exact M.norm_qValue_sub_le_of_l1_le M' hr hγ hδ _)

end FiniteMDP

/-! End of step 1, part 1. -/

/-! ## Step 1, part 2. The l1 norm through sign patterns (R1) -/

/-- The sign pattern of a boolean vector, `1` where it is true and `-1` where it is false. -/
def signOf {X : Type*} (b : X → Bool) (x : X) : ℝ := if b x then 1 else -1

/-- **R1.** Every sign pattern gives at most the l1 norm. -/
theorem sum_signOf_mul_le_sum_abs {X : Type*} [Fintype X] (b : X → Bool) (v : X → ℝ) :
    ∑ x, signOf b x * v x ≤ ∑ x, |v x| := by
  refine Finset.sum_le_sum fun x _ => ?_
  unfold signOf
  split_ifs
  · simpa using le_abs_self (v x)
  · simpa using neg_le_abs (v x)

/-- **R1.** Some sign pattern gives exactly the l1 norm. -/
theorem exists_sum_signOf_mul_eq_sum_abs {X : Type*} [Fintype X] (v : X → ℝ) :
    ∃ b : X → Bool, ∑ x, signOf b x * v x = ∑ x, |v x| := by
  refine ⟨fun x => decide (0 ≤ v x), Finset.sum_congr rfl fun x _ => ?_⟩
  by_cases h : 0 ≤ v x
  · simp [signOf, h, abs_of_nonneg h]
  · simp [signOf, h, abs_of_neg (not_le.mp h)]

/-- **R1.** There are `2 ^ |X|` sign patterns. -/
theorem card_signPatterns {X : Type*} [Fintype X] [DecidableEq X] :
    Fintype.card (X → Bool) = 2 ^ Fintype.card X := by
  simp

/-! End of step 1. -/

end Learning
