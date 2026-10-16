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

/-! ## Step 3, part 1. What Hoeffding needs to know about one sample (C1) -/

variable [MeasurableSingletonClass S]

/-- Every function out of a finite type with measurable singletons is measurable. -/
theorem measurable_of_finiteType {X β : Type*} [Fintype X] [MeasurableSpace X]
    [MeasurableSingletonClass X] [MeasurableSpace β] (f : X → β) : Measurable f :=
  fun t _ => (Set.toFinite (f ⁻¹' t)).measurableSet

/-- A sign pattern takes values in `[-1, 1]`. -/
theorem signOf_mem_Icc {X : Type*} (b : X → Bool) (x : X) :
    signOf b x ∈ Set.Icc (-1 : ℝ) 1 := by
  unfold signOf
  split_ifs <;> norm_num

/-- A variable with values in an interval of length two has sub-Gaussian constant one. -/
theorem hoeffdingParam_eq_one (m : ℝ) : (‖(1 - m) - (-1 - m)‖₊ / 2) ^ 2 = (1 : ℝ≥0) := by
  rw [show (1 - m) - (-1 - m) = (1 : ℝ) - -1 by ring]
  norm_num

namespace FiniteMDP

namespace GenerativeSamples

variable {M : FiniteMDP S A} {N : ℕ} {μ : Measure Ω} (G : M.GenerativeSamples N μ)

theorem measurableSet_X_eq (s : S) (a : A) (i : Fin N) (s' : S) :
    MeasurableSet {ω | G.X s a i ω = s'} :=
  G.measurable s a i (measurableSet_singleton s')

/-- The empirical model averages a sign pattern exactly as the samples do. -/
theorem sum_signOf_mul_empiricalP (b : S → Bool) (ω : Ω) (s : S) (a : A) :
    ∑ s', signOf b s' * G.empiricalP ω s a s' = (∑ i, signOf b (G.X s a i ω)) / N := by
  unfold empiricalP
  simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero,
    mul_div_assoc', Finset.mul_sum, mul_ite, mul_one, mul_zero]
  rw [← Finset.sum_div, Finset.sum_comm]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  simp

/-- The sign of one sample, written with the indicators of the events `X = s'`. -/
theorem signOf_X_eq_sum (b : S → Bool) (s : S) (a : A) (i : Fin N) (ω : Ω) :
    signOf b (G.X s a i ω) = ∑ s', signOf b s' * {ω' | G.X s a i ω' = s'}.indicator 1 ω := by
  simp [Set.indicator_apply, mul_ite]

theorem integrable_signOf_X [IsFiniteMeasure μ] (b : S → Bool) (s : S) (a : A) (i : Fin N) :
    Integrable (fun ω => signOf b (G.X s a i ω)) μ := by
  simp_rw [G.signOf_X_eq_sum b s a i]
  exact integrable_finset_sum _ fun s' _ =>
    ((integrable_const (1 : ℝ)).indicator (G.measurableSet_X_eq s a i s')).const_mul _

/-- The mean of the sign of one sample is the average of the sign pattern under `P`. -/
theorem integral_signOf_X [IsFiniteMeasure μ] (b : S → Bool) (s : S) (a : A) (i : Fin N) :
    ∫ ω, signOf b (G.X s a i ω) ∂μ = ∑ s', signOf b s' * M.P s a s' := by
  simp_rw [G.signOf_X_eq_sum b s a i]
  rw [integral_finset_sum]
  · refine Finset.sum_congr rfl fun s' _ => ?_
    rw [integral_const_mul, integral_indicator_one (G.measurableSet_X_eq s a i s')]
    congr 1
    exact G.law s a i s'
  · intro s' _
    exact ((integrable_const (1 : ℝ)).indicator (G.measurableSet_X_eq s a i s')).const_mul _

end GenerativeSamples

end FiniteMDP

/-! End of step 3, part 1. -/

/-! ## Step 3, part 2. Hoeffding for one pair and one sign pattern (C1) -/

namespace FiniteMDP

namespace GenerativeSamples

variable {M : FiniteMDP S A} {N : ℕ} {μ : Measure Ω} (G : M.GenerativeSamples N μ)

/-- **C1.** Hoeffding for one state-action pair and one sign pattern. With `N` samples, the
empirical model overshoots `P` along the sign pattern `b` by at least `t` with probability at
most `exp (-N t² / 2)`. Book Proposition 2.1, first step of the proof. -/
theorem measureReal_sum_signOf_ge_le [IsProbabilityMeasure μ] (hN : 0 < N) (b : S → Bool)
    (s : S) (a : A) {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | t ≤ ∑ s', signOf b s' * (G.empiricalP ω s a s' - M.P s a s')} ≤
      Real.exp (-((N : ℝ) * t ^ 2) / 2) := by
  generalize hm : ∑ s', signOf b s' * M.P s a s' = m
  have hNne : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  -- The centred samples are independent.
  have hindep : ProbabilityTheory.iIndepFun (fun i ω => signOf b (G.X s a i ω) - m) μ :=
    (G.indep s a).comp (fun _ x => signOf b x - m) fun _ => measurable_of_finiteType _
  -- Each centred sample is sub-Gaussian with constant one, by Hoeffding's lemma.
  have hsubG : ∀ i ∈ (Finset.univ : Finset (Fin N)), ProbabilityTheory.HasSubgaussianMGF
      (fun ω => signOf b (G.X s a i ω) - m) ((fun _ => (1 : ℝ≥0)) i) μ := by
    intro i _
    have hmeas : AEMeasurable (fun ω => signOf b (G.X s a i ω) - m) μ :=
      ((measurable_of_finiteType fun x : S => signOf b x - m).comp
        (G.measurable s a i)).aemeasurable
    have hbdd : ∀ᵐ ω ∂μ, signOf b (G.X s a i ω) - m ∈ Set.Icc (-1 - m) (1 - m) := by
      refine ae_of_all μ fun ω => ?_
      have h := Set.mem_Icc.mp (signOf_mem_Icc b (G.X s a i ω))
      exact Set.mem_Icc.mpr ⟨by linarith [h.1], by linarith [h.2]⟩
    have hmean : ∫ ω, signOf b (G.X s a i ω) - m ∂μ = 0 := by
      rw [integral_sub (G.integrable_signOf_X b s a i) (integrable_const m),
        G.integral_signOf_X b s a i, hm, integral_const]
      simp
    have h :=
      ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero hmeas hbdd hmean
    rw [hoeffdingParam_eq_one m] at h
    exact h
  -- Multiplied by `N`, the event is about the sum of the centred samples.
  have hsub : {ω | t ≤ ∑ s', signOf b s' * (G.empiricalP ω s a s' - M.P s a s')} ⊆
      {ω | (N : ℝ) * t ≤ ∑ i, (signOf b (G.X s a i ω) - m)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have h1 := G.sum_signOf_mul_empiricalP b ω s a
    rw [eq_div_iff hNne] at h1
    have hkey : (N : ℝ) * ∑ s', signOf b s' * (G.empiricalP ω s a s' - M.P s a s') =
        ∑ i, (signOf b (G.X s a i ω) - m) := by
      simp only [mul_sub, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
      rw [← h1, ← hm]
      ring
    rw [← hkey]
    exact mul_le_mul_of_nonneg_left hω (Nat.cast_nonneg N)
  -- Hoeffding's inequality for the sum, then simplify the exponent.
  have hH := ProbabilityTheory.HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindep hsubG
    (mul_nonneg (Nat.cast_nonneg N) ht)
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans (hH.trans_eq ?_)
  congr 1
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one,
    NNReal.coe_natCast]
  rw [div_eq_div_iff (mul_ne_zero two_ne_zero hNne) two_ne_zero]
  ring

end GenerativeSamples

end FiniteMDP

/-! End of step 3. -/

end Learning
