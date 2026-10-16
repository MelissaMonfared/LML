/-
Copyright (c) 2026. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Melissa Monfared
-/
import LeanMachineLearning.ReinforcementLearning.MDP.Discounted.Planning

/-!
# Tabular policy gradient

Priority 3 of the RL plan. This file grows one step at a time, like the other two.

Step 1 defines the discounted state occupancy `d^π_μ` through values of indicator rewards and
proves what the policy gradient theorem needs from it, the book's identity (1.12), that
`d^π_μ` is a probability vector, and the lower bound (10.1).

Book references are to Agarwal, Brantley, Jiang, Kakade and Sun, *Reinforcement Learning
Theory and Algorithms*, draft of 27 June 2026.
-/

set_option linter.unusedSectionVars false

open scoped NNReal

noncomputable section

namespace Learning

variable {S A : Type*} [Fintype S] [Fintype A] [DecidableEq A] [Nonempty A] [DecidableEq S]

/-- The reward that pays one in state `s` and nothing elsewhere, whatever the action. -/
def stateIndicator (s : S) : S → A → ℝ := fun s' _ => if s' = s then 1 else 0

/-- Averaging the indicator reward of `s` under any policy gives the indicator of `s`. -/
theorem avg_stateIndicator (π : StationaryPolicy S A) (s s₀ : S) :
    π.avg (stateIndicator s) s₀ = if s₀ = s then 1 else 0 := by
  simp only [StationaryPolicy.avg, stateIndicator]
  split_ifs
  · simp [π.sum_eq_one s₀]
  · simp

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- The value from a start distribution `μ`, written `V^π(μ)` in the book. -/
def startValue (μ : S → ℝ) (π : StationaryPolicy S A) : ℝ := ∑ s, μ s * M.value π s

/-- **O2.** The discounted state occupancy `d^π_μ` of book Chapters 9 and 10. It is `1 - γ`
times the expected discounted number of visits to `s`, which is the value of the indicator
reward of `s`. -/
def occupancy (μ : S → ℝ) (π : StationaryPolicy S A) (s : S) : ℝ :=
  (1 - (M.γ : ℝ)) * ∑ s₀, μ s₀ * (M.withReward (stateIndicator s)).value π s₀

/-- Values are linear in the reward. The value of any reward is the sum over states `s` of
the averaged reward at `s` times the value of the indicator reward of `s`. -/
theorem value_withReward_eq_sum (ρ : S → A → ℝ) (π : StationaryPolicy S A) :
    (M.withReward ρ).value π =
      fun t => ∑ s, π.avg ρ s * (M.withReward (stateIndicator s)).value π t := by
  symm
  apply (M.withReward ρ).eq_value_of_evalOp_eq
  funext t
  change π.avg ρ t + (M.γ : ℝ) * ∑ t', M.transPi π t t' *
      (∑ s, π.avg ρ s * (M.withReward (stateIndicator s)).value π t') =
    ∑ s, π.avg ρ s * (M.withReward (stateIndicator s)).value π t
  -- each indicator value satisfies its own Bellman equation
  have hs : ∀ s, (M.withReward (stateIndicator s)).value π t =
      (if t = s then 1 else 0) + (M.γ : ℝ) *
        ∑ t', M.transPi π t t' * (M.withReward (stateIndicator s)).value π t' := by
    intro s
    rw [← avg_stateIndicator π s t]
    exact (congrFun ((M.withReward (stateIndicator s)).evalOp_value π) t).symm
  have h1 : ∑ s, π.avg ρ s * (if t = s then (1 : ℝ) else 0) = π.avg ρ t := by
    simp [mul_ite]
  have h2 : ∑ s, π.avg ρ s * ((M.γ : ℝ) *
        ∑ t', M.transPi π t t' * (M.withReward (stateIndicator s)).value π t') =
      (M.γ : ℝ) * ∑ t', M.transPi π t t' *
        ∑ s, π.avg ρ s * (M.withReward (stateIndicator s)).value π t' := by
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun t' _ => Finset.sum_congr rfl fun s _ => ?_
    ring
  rw [← h1, ← h2, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [hs s, mul_add]

/-- **O3.** Book identity (1.12). The value from `μ` is the occupancy average of the reward,
divided by `1 - γ`. -/
theorem startValue_eq (μ : S → ℝ) (π : StationaryPolicy S A) :
    M.startValue μ π = (∑ s, M.occupancy μ π s * M.rewardPi π s) / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  have hdec := M.value_withReward_eq_sum M.r π
  have hM : M.withReward M.r = M := rfl
  rw [hM] at hdec
  rw [eq_div_iff hpos.ne']
  unfold startValue occupancy rewardPi
  rw [hdec]
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun s₀ _ => ?_
  ring

/-- A constant reward `c` has value `c / (1 - γ)` from every state. -/
theorem value_withReward_const (c : ℝ) (π : StationaryPolicy S A) :
    (M.withReward (fun _ _ => c)).value π = fun _ => c / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  symm
  apply (M.withReward (fun _ _ => c)).eq_value_of_evalOp_eq
  funext t
  change π.avg (fun _ _ => c) t + (M.γ : ℝ) * ∑ t', M.transPi π t t' * (c / (1 - (M.γ : ℝ))) =
    c / (1 - (M.γ : ℝ))
  have havg : π.avg (fun _ _ => c) t = c := by
    simp only [StationaryPolicy.avg]
    rw [← Finset.sum_mul, π.sum_eq_one t, one_mul]
  rw [havg, ← Finset.sum_mul, M.sum_transPi π t, one_mul]
  have hc : c / (1 - (M.γ : ℝ)) * (1 - (M.γ : ℝ)) = c := div_mul_cancel₀ c hpos.ne'
  linarith

/-- **O2.** The occupancy is a probability vector when `μ` is. Book, the remark after (1.9). -/
theorem sum_occupancy {μ : S → ℝ} (hμ1 : ∑ s, μ s = 1) (π : StationaryPolicy S A) :
    ∑ s, M.occupancy μ π s = 1 := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  -- the indicator rewards add up to the constant reward one
  have hsum : ∀ t, ∑ s, (M.withReward (stateIndicator s)).value π t = 1 / (1 - (M.γ : ℝ)) := by
    intro t
    have h := congrFun (M.value_withReward_eq_sum (fun _ _ => 1) π) t
    rw [M.value_withReward_const 1 π] at h
    have havg : ∀ s, π.avg (fun _ _ => (1 : ℝ)) s = 1 := by
      intro s
      simp only [StationaryPolicy.avg, mul_one]
      exact π.sum_eq_one s
    simp only [havg, one_mul] at h
    exact h.symm
  calc ∑ s, M.occupancy μ π s
      = (1 - (M.γ : ℝ)) * ∑ s₀, μ s₀ * ∑ s, (M.withReward (stateIndicator s)).value π s₀ := by
        unfold occupancy
        rw [← Finset.mul_sum]
        congr 1
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun s₀ _ => ?_
        rw [Finset.mul_sum]
    _ = (1 - (M.γ : ℝ)) * ∑ s₀, μ s₀ * (1 / (1 - (M.γ : ℝ))) := by
        simp only [hsum]
    _ = 1 := by
        rw [← Finset.sum_mul, hμ1, one_mul, mul_one_div_cancel hpos.ne']

/-- **O2.** Book inequality (10.1), `(1 - γ) μ(s) ≤ d^π_μ(s)`. -/
theorem occupancy_ge {μ : S → ℝ} (hμ : ∀ s, 0 ≤ μ s) (π : StationaryPolicy S A) (s : S) :
    (1 - (M.γ : ℝ)) * μ s ≤ M.occupancy μ π s := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  obtain ⟨V, hV⟩ : ∃ V : S → ℝ, V = (M.withReward (stateIndicator s)).value π := ⟨_, rfl⟩
  -- the indicator value is nonnegative, by the comparison principle L6
  have hnn : 0 ≤ V := by
    have h0 : (0 : S → ℝ) ≤ (M.withReward (stateIndicator s)).evalOp π 0 := by
      intro t
      change (0 : ℝ) ≤ π.avg (stateIndicator s) t + (M.γ : ℝ) * ∑ t', M.transPi π t t' * 0
      rw [avg_stateIndicator]
      simp only [mul_zero, Finset.sum_const_zero, add_zero]
      split_ifs <;> norm_num
    rw [hV]
    exact le_fixedPoint_of_le_map ((M.withReward (stateIndicator s)).contractingWith_evalOp π)
      ((M.withReward (stateIndicator s)).monotone_evalOp π) h0
  -- at `s` itself the value is at least one
  have hself : 1 ≤ V s := by
    have h : V s = π.avg (stateIndicator s) s +
        (M.γ : ℝ) * ∑ t', M.transPi π s t' * V t' := by
      rw [hV]
      exact (congrFun ((M.withReward (stateIndicator s)).evalOp_value π) s).symm
    rw [avg_stateIndicator, if_pos rfl] at h
    have hrest : 0 ≤ (M.γ : ℝ) * ∑ t', M.transPi π s t' * V t' :=
      mul_nonneg M.gamma_nonneg
        (Finset.sum_nonneg fun t' _ => mul_nonneg (M.transPi_nonneg π s t') (hnn t'))
    linarith
  -- so the sum over start states is at least its `s` term
  have hsum : μ s ≤ ∑ s₀, μ s₀ * V s₀ :=
    calc μ s ≤ μ s * V s := le_mul_of_one_le_right (hμ s) hself
      _ ≤ ∑ s₀, μ s₀ * V s₀ :=
          Finset.single_le_sum (f := fun s₀ => μ s₀ * V s₀)
            (fun s₀ _ => mul_nonneg (hμ s₀) (hnn s₀)) (Finset.mem_univ s)
  unfold occupancy
  rw [← hV]
  exact mul_le_mul_of_nonneg_left hsum hpos.le

end FiniteMDP

/-! End of step 1. -/

/-! ## Step 2. Performance difference with occupancies (O3) and continuity in the policy -/

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- The occupancy does not depend on the reward. -/
theorem occupancy_withReward (ρ : S → A → ℝ) (μ : S → ℝ) (π : StationaryPolicy S A) :
    (M.withReward ρ).occupancy μ π = M.occupancy μ π := rfl

/-- **O3.** The performance difference lemma in the book's form, Lemma 1.16. The occupancy
is the one of the first policy and the advantage the one of the second. -/
theorem startValue_sub_startValue (μ : S → ℝ) (π π' : StationaryPolicy S A) :
    M.startValue μ π - M.startValue μ π' =
      (∑ s, M.occupancy μ π s * π.avg (M.advantage π') s) / (1 - (M.γ : ℝ)) := by
  have h12 := M.value_sub_value_eq π π'
  have hstart : M.startValue μ π - M.startValue μ π' =
      (M.withReward (M.advantage π')).startValue μ π := by
    unfold startValue
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun s₀ _ => ?_
    rw [← mul_sub]
    congr 1
    exact congrFun h12 s₀
  rw [hstart, (M.withReward (M.advantage π')).startValue_eq μ π, M.occupancy_withReward]
  rfl

/-- Averaging the advantage of `π` under `π'` only sees the difference of the two policies. -/
theorem avg_advantage_eq (π π' : StationaryPolicy S A) (s : S) :
    π'.avg (M.advantage π) s = ∑ a, (π'.prob s a - π.prob s a) * M.qValue π s a := by
  have hV : M.value π s = ∑ a, π.prob s a * M.qValue π s a :=
    congrFun (M.value_eq_avg_qValue π) s
  simp only [StationaryPolicy.avg, advantage, mul_sub, sub_mul, Finset.sum_sub_distrib,
    ← Finset.sum_mul, π'.sum_eq_one s, one_mul]
  rw [hV]

/-- A sharper form of L9. The value is bounded by the reward averaged by the policy. -/
theorem norm_value_le_rewardPi (π : StationaryPolicy S A) :
    ‖M.value π‖ ≤ ‖M.rewardPi π‖ / (1 - (M.γ : ℝ)) := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have key : ‖M.value π‖ ≤ ‖M.rewardPi π‖ + (M.γ : ℝ) * ‖M.value π‖ := by
    have hnn : 0 ≤ ‖M.rewardPi π‖ + (M.γ : ℝ) * ‖M.value π‖ := by positivity
    rw [pi_norm_le_iff_of_nonneg hnn]
    intro s
    have hfix := congrFun (M.evalOp_value π) s
    rw [← hfix]
    have h1 : |M.rewardPi π s| ≤ ‖M.rewardPi π‖ := by
      have h := norm_le_pi_norm (M.rewardPi π) s
      rwa [Real.norm_eq_abs] at h
    have h2 : |∑ s', M.transPi π s s' * M.value π s'| ≤ ‖M.value π‖ :=
      abs_sum_mul_le_norm (M.transPi_nonneg π s) (M.sum_transPi π s) (M.value π)
    calc ‖M.evalOp π (M.value π) s‖
        = ‖M.rewardPi π s + (M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π s'‖ := rfl
      _ ≤ ‖M.rewardPi π s‖ + ‖(M.γ : ℝ) * ∑ s', M.transPi π s s' * M.value π s'‖ :=
          norm_add_le _ _
      _ = |M.rewardPi π s| + (M.γ : ℝ) * |∑ s', M.transPi π s s' * M.value π s'| := by
          rw [norm_mul, Real.norm_of_nonneg M.gamma_nonneg, Real.norm_eq_abs (M.rewardPi π s),
            Real.norm_eq_abs (∑ s', M.transPi π s s' * M.value π s')]
      _ ≤ ‖M.rewardPi π‖ + (M.γ : ℝ) * ‖M.value π‖ :=
          add_le_add h1 (mul_le_mul_of_nonneg_left h2 M.gamma_nonneg)
  rw [le_div_iff₀ (by linarith)]
  linarith

/-- **L16.** Values move continuously with the policy. If the action distributions of `π`
and `π'` differ by at most `δ` in l1 at every state, the values differ by at most
`‖r‖ δ / (1 - γ)²`. Not in the book. It replaces the book's assumption, at the start of
Section 9.1, that derivatives can be exchanged with infinite sums. -/
theorem norm_value_sub_value_le [Nonempty S] (π π' : StationaryPolicy S A) {δ : ℝ}
    (hδ : ∀ s, ∑ a, |π'.prob s a - π.prob s a| ≤ δ) :
    ‖M.value π' - M.value π‖ ≤ ‖M.r‖ * δ / (1 - (M.γ : ℝ)) ^ 2 := by
  have hγ1 : (M.γ : ℝ) < 1 := by exact_mod_cast M.γ_lt_one
  have hpos : 0 < 1 - (M.γ : ℝ) := by linarith
  have hδ0 : 0 ≤ δ := by
    obtain ⟨s₀⟩ := ‹Nonempty S›
    exact (Finset.sum_nonneg fun a _ => abs_nonneg _).trans (hδ s₀)
  have hC : 0 ≤ δ * (‖M.r‖ / (1 - (M.γ : ℝ))) :=
    mul_nonneg hδ0 (div_nonneg (norm_nonneg _) hpos.le)
  -- the averaged advantage is small at every state
  have havg : ‖π'.avg (M.advantage π)‖ ≤ δ * (‖M.r‖ / (1 - (M.γ : ℝ))) := by
    rw [pi_norm_le_iff_of_nonneg hC]
    intro s
    rw [Real.norm_eq_abs, M.avg_advantage_eq π π' s]
    calc |∑ a, (π'.prob s a - π.prob s a) * M.qValue π s a|
        ≤ ∑ a, |π'.prob s a - π.prob s a| * |M.qValue π s a| := by
          refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [abs_mul]
      _ ≤ ∑ a, |π'.prob s a - π.prob s a| * (‖M.r‖ / (1 - (M.γ : ℝ))) := by
          refine Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
          have h := (norm_le_pi_norm (M.qValue π s) a).trans (norm_le_pi_norm (M.qValue π) s)
          rw [Real.norm_eq_abs] at h
          exact h.trans (M.norm_qValue_le π)
      _ = (∑ a, |π'.prob s a - π.prob s a|) * (‖M.r‖ / (1 - (M.γ : ℝ))) := by
          rw [Finset.sum_mul]
      _ ≤ δ * (‖M.r‖ / (1 - (M.γ : ℝ))) :=
          mul_le_mul_of_nonneg_right (hδ s) (div_nonneg (norm_nonneg _) hpos.le)
  -- by L12 the difference of values is a value with this averaged reward
  rw [M.value_sub_value_eq π' π]
  calc ‖(M.withReward (M.advantage π)).value π'‖
      ≤ ‖(M.withReward (M.advantage π)).rewardPi π'‖ / (1 - (M.γ : ℝ)) :=
        (M.withReward (M.advantage π)).norm_value_le_rewardPi π'
    _ ≤ δ * (‖M.r‖ / (1 - (M.γ : ℝ))) / (1 - (M.γ : ℝ)) :=
        div_le_div_of_nonneg_right havg hpos.le
    _ = ‖M.r‖ * δ / (1 - (M.γ : ℝ)) ^ 2 := by
        rw [mul_div_assoc', div_div, ← sq, mul_comm δ]

end FiniteMDP

/-! End of step 2. -/

end Learning
