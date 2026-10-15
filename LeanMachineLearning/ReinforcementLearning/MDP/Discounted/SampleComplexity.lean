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

/-! ## Step 2. Samples and the empirical model (G1, G2, G3) -/

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace S] [DecidableEq S]

namespace FiniteMDP

/-- **G1.** `N` samples for every state-action pair, drawn from the transitions of `M`.
Book Section 2.1. Only the samples of one pair are assumed independent, which is all the
proof uses. -/
structure GenerativeSamples (M : FiniteMDP S A) (N : ℕ) (μ : Measure Ω) where
  /-- `X s a i ω` is the `i`-th next state sampled from the pair `(s, a)`. -/
  X : S → A → Fin N → Ω → S
  measurable : ∀ s a i, Measurable (X s a i)
  /-- The samples of one pair are independent. -/
  indep : ∀ s a, ProbabilityTheory.iIndepFun (fun i => X s a i) μ
  /-- Each sample lands on `s'` with probability `P s a s'`. -/
  law : ∀ s a i s', μ.real {ω | X s a i ω = s'} = M.P s a s'

namespace GenerativeSamples

variable {M : FiniteMDP S A} {N : ℕ} {μ : Measure Ω} (G : M.GenerativeSamples N μ)

/-- **G2.** The empirical transition model, the share of the samples of `(s, a)` that
land on `s'`. Book Section 2.1. -/
def empiricalP (ω : Ω) (s : S) (a : A) (s' : S) : ℝ :=
  ((Finset.univ.filter fun i => G.X s a i ω = s').card : ℝ) / N

/-- **G3.** The empirical probabilities are nonnegative. -/
theorem empiricalP_nonneg (ω : Ω) (s : S) (a : A) (s' : S) : 0 ≤ G.empiricalP ω s a s' := by
  unfold empiricalP
  positivity

/-- **G3.** With at least one sample, the empirical probabilities of each pair add up to one. -/
theorem sum_empiricalP (hN : 0 < N) (ω : Ω) (s : S) (a : A) :
    ∑ s', G.empiricalP ω s a s' = 1 := by
  have hcount : ∑ s', ((Finset.univ.filter fun i => G.X s a i ω = s').card : ℝ) = N := by
    simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
    rw [Finset.sum_comm]
    simp
  have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  unfold empiricalP
  rw [← Finset.sum_div, hcount, div_self hN']

/-- **G3.** With at least one sample, the empirical model is a finite discounted MDP with the
rewards and the discount of `M`. -/
def empiricalMDP (hN : 0 < N) (ω : Ω) : FiniteMDP S A where
  P := G.empiricalP ω
  P_nonneg s a s' := G.empiricalP_nonneg ω s a s'
  P_sum s a := G.sum_empiricalP hN ω s a
  r := M.r
  γ := M.γ
  γ_lt_one := M.γ_lt_one

theorem empiricalMDP_P (hN : 0 < N) (ω : Ω) : (G.empiricalMDP hN ω).P = G.empiricalP ω := rfl

theorem empiricalMDP_r (hN : 0 < N) (ω : Ω) : (G.empiricalMDP hN ω).r = M.r := rfl

theorem empiricalMDP_γ (hN : 0 < N) (ω : Ω) : (G.empiricalMDP hN ω).γ = M.γ := rfl

end GenerativeSamples

end FiniteMDP

/-! End of step 2. -/

end Learning
