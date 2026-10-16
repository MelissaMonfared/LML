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

/-! ## Step 3, part 1. Values and occupancies move continuously with the policy -/

open Filter Topology

/-- The l1 distance between two policies at one state is at most their total l1 distance
over all states. -/
theorem StationaryPolicy.sum_abs_sub_le_sum_sum (π π' : StationaryPolicy S A) (s : S) :
    ∑ a, |π'.prob s a - π.prob s a| ≤ ∑ s', ∑ a, |π'.prob s' a - π.prob s' a| :=
  Finset.single_le_sum (f := fun s' => ∑ a, |π'.prob s' a - π.prob s' a|)
    (fun s' _ => Finset.sum_nonneg fun a _ => abs_nonneg _) (Finset.mem_univ s)

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- Values are continuous in the policy. If the action probabilities of a family of policies
converge to those of `π₀`, the values converge to the values of `π₀`. This follows from L16,
and works for any kind of limit, sequences and `t → t₀` alike. -/
theorem tendsto_value [Nonempty S] {ι : Type*} {l : Filter ι} (π : ι → StationaryPolicy S A)
    (π₀ : StationaryPolicy S A)
    (h : ∀ s a, Tendsto (fun i => (π i).prob s a) l (𝓝 (π₀.prob s a))) (s₀ : S) :
    Tendsto (fun i => M.value (π i) s₀) l (𝓝 (M.value π₀ s₀)) := by
  -- the total l1 distance between the action distributions goes to zero
  have hδ : Tendsto (fun i => ∑ s, ∑ a, |(π i).prob s a - π₀.prob s a|) l (𝓝 0) := by
    have h1 : Tendsto (fun i => ∑ s, ∑ a, |(π i).prob s a - π₀.prob s a|) l
        (𝓝 (∑ s : S, ∑ a : A, |π₀.prob s a - π₀.prob s a|)) := by
      refine tendsto_finsetSum _ fun s _ => ?_
      refine tendsto_finsetSum _ fun a _ => ?_
      exact ((h s a).sub_const (π₀.prob s a)).abs
    have h0 : ∑ s : S, ∑ a : A, |π₀.prob s a - π₀.prob s a| = 0 := by simp
    rwa [h0] at h1
  -- so does the bound of L16
  have hC : Tendsto (fun i => ‖M.r‖ * (∑ s, ∑ a, |(π i).prob s a - π₀.prob s a|) /
      (1 - (M.γ : ℝ)) ^ 2) l (𝓝 0) := by
    have h2 := (hδ.const_mul ‖M.r‖).div_const ((1 - (M.γ : ℝ)) ^ 2)
    rwa [mul_zero, zero_div] at h2
  -- at every index, L16 bounds the distance between the values
  have hbound : ∀ i, ‖M.value (π i) s₀ - M.value π₀ s₀‖ ≤
      ‖M.r‖ * (∑ s, ∑ a, |(π i).prob s a - π₀.prob s a|) / (1 - (M.γ : ℝ)) ^ 2 := by
    intro i
    calc ‖M.value (π i) s₀ - M.value π₀ s₀‖ = ‖(M.value (π i) - M.value π₀) s₀‖ := rfl
      _ ≤ ‖M.value (π i) - M.value π₀‖ := norm_le_pi_norm _ s₀
      _ ≤ ‖M.r‖ * (∑ s, ∑ a, |(π i).prob s a - π₀.prob s a|) / (1 - (M.γ : ℝ)) ^ 2 :=
          M.norm_value_sub_value_le π₀ (π i) fun s => π₀.sum_abs_sub_le_sum_sum (π i) s
  rw [tendsto_iff_norm_sub_tendsto_zero]
  exact squeeze_zero (fun i => norm_nonneg _) hbound hC

/-- The occupancy is continuous in the policy too, by `tendsto_value` for the indicator
rewards. This is what lets T5 replace the occupancy of `π t` by the one of `π t₀` in the
limit. -/
theorem tendsto_occupancy [Nonempty S] {ι : Type*} {l : Filter ι} (μ : S → ℝ)
    (π : ι → StationaryPolicy S A) (π₀ : StationaryPolicy S A)
    (h : ∀ s a, Tendsto (fun i => (π i).prob s a) l (𝓝 (π₀.prob s a))) (s : S) :
    Tendsto (fun i => M.occupancy μ (π i) s) l (𝓝 (M.occupancy μ π₀ s)) := by
  unfold occupancy
  refine Tendsto.const_mul _ ?_
  refine tendsto_finsetSum _ fun s₀ _ => ?_
  refine Tendsto.const_mul _ ?_
  exact (M.withReward (stateIndicator s)).tendsto_value π π₀ h s₀

end FiniteMDP

/-! End of step 3, part 1. -/

/-! ## Step 3, part 2. The policy gradient theorem (T5) -/

/-- When the action probabilities are positive, the derivative of `π(a | s)` is `π(a | s)`
times the derivative of `log π(a | s)`. This turns T5 into the book's `∇ log π` form. -/
theorem prob_mul_deriv_log (π : ℝ → StationaryPolicy S A) (g : S → A → ℝ) (t₀ : ℝ)
    (hπ : ∀ s a, HasDerivAt (fun t => (π t).prob s a) (g s a) t₀)
    (hpos : ∀ s a, 0 < (π t₀).prob s a) (s : S) (a : A) :
    (π t₀).prob s a * deriv (fun t => Real.log ((π t).prob s a)) t₀ = g s a := by
  rw [((hπ s a).log (hpos s a).ne').deriv]
  exact mul_div_cancel₀ (g s a) (hpos s a).ne'

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- **T5.** The policy gradient theorem, book Theorem 9.4, `Q` form. Take a path of policies
`t ↦ π t` whose action probabilities have derivatives `g s a` at `t₀`. Then the value from
`μ` has a derivative at `t₀`, and it is the occupancy average of `∑ a, g s a * Q(s, a)`
divided by `1 - γ`. The path `θ + t eᵢ` gives each partial derivative. The proof goes through
the performance difference lemma and L16 instead of trajectories, so the book's assumption
that derivatives can be exchanged with infinite sums is not needed. -/
theorem hasDerivAt_startValue [Nonempty S] (μ : S → ℝ) (π : ℝ → StationaryPolicy S A)
    (g : S → A → ℝ) (t₀ : ℝ) (hπ : ∀ s a, HasDerivAt (fun t => (π t).prob s a) (g s a) t₀) :
    HasDerivAt (fun t => M.startValue μ (π t))
      ((∑ s, M.occupancy μ (π t₀) s * ∑ a, g s a * M.qValue (π t₀) s a) /
        (1 - (M.γ : ℝ))) t₀ := by
  -- the occupancy moves continuously with `t`, by part 1
  have hd : ∀ s, Tendsto (fun t => M.occupancy μ (π t) s) (𝓝[≠] t₀)
      (𝓝 (M.occupancy μ (π t₀) s)) := fun s =>
    (M.tendsto_occupancy μ π (π t₀) (fun s' a => (hπ s' a).continuousAt.tendsto) s).mono_left
      nhdsWithin_le_nhds
  -- the slopes of the action probabilities converge to their derivatives
  have hg : ∀ s a, Tendsto (slope (fun u => (π u).prob s a) t₀) (𝓝[≠] t₀) (𝓝 (g s a)) :=
    fun s a => hasDerivAt_iff_tendsto_slope.mp (hπ s a)
  -- by the performance difference lemma, the slope of the value is an occupancy average
  have hslope : slope (fun t => M.startValue μ (π t)) t₀ = fun t =>
      (∑ s, M.occupancy μ (π t) s *
        ∑ a, slope (fun u => (π u).prob s a) t₀ t * M.qValue (π t₀) s a) /
        (1 - (M.γ : ℝ)) := by
    funext t
    rw [slope_def_field, M.startValue_sub_startValue μ (π t) (π t₀)]
    simp only [M.avg_advantage_eq (π t₀) (π t), slope_def_field]
    rw [div_right_comm]
    congr 1
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [mul_div_assoc, Finset.sum_div]
    congr 1
    refine Finset.sum_congr rfl fun a _ => ?_
    ring
  -- now pass to the limit term by term
  refine hasDerivAt_iff_tendsto_slope.mpr ?_
  refine Tendsto.congr (fun t => (congrFun hslope t).symm) ?_
  refine Tendsto.div_const ?_ _
  refine tendsto_finsetSum _ fun s _ => ?_
  refine (hd s).mul ?_
  refine tendsto_finsetSum _ fun a _ => ?_
  exact (hg s a).mul_const _

/-- **T5.** The policy gradient theorem, book Theorem 9.4, advantage form. The derivatives of
the action probabilities at each state add up to zero, so `Q` can be replaced by the
advantage. -/
theorem hasDerivAt_startValue_advantage [Nonempty S] (μ : S → ℝ)
    (π : ℝ → StationaryPolicy S A) (g : S → A → ℝ) (t₀ : ℝ)
    (hπ : ∀ s a, HasDerivAt (fun t => (π t).prob s a) (g s a) t₀) :
    HasDerivAt (fun t => M.startValue μ (π t))
      ((∑ s, M.occupancy μ (π t₀) s * ∑ a, g s a * M.advantage (π t₀) s a) /
        (1 - (M.γ : ℝ))) t₀ := by
  -- the probabilities at each state always add up to one, so their derivatives add up to zero
  have hg0 : ∀ s, ∑ a, g s a = 0 := by
    intro s
    have h1 : HasDerivAt (fun t => ∑ a, (π t).prob s a) (∑ a, g s a) t₀ :=
      HasDerivAt.fun_sum fun a _ => hπ s a
    have h2 : HasDerivAt (fun t => ∑ a, (π t).prob s a) 0 t₀ := by
      have h3 : (fun t => ∑ a, (π t).prob s a) = fun _ => (1 : ℝ) := by
        funext t
        exact (π t).sum_eq_one s
      rw [h3]
      exact hasDerivAt_const _ _
    exact h1.unique h2
  have hsum : ∀ s, ∑ a, g s a * M.advantage (π t₀) s a = ∑ a, g s a * M.qValue (π t₀) s a := by
    intro s
    simp only [advantage, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hg0 s, zero_mul,
      sub_zero]
  simp only [hsum]
  exact M.hasDerivAt_startValue μ π g t₀ hπ

/-- **T5.** Book Theorem 9.4 in the book's own notation, with `∇ log π`, `Q` form. -/
theorem hasDerivAt_startValue_log [Nonempty S] (μ : S → ℝ) (π : ℝ → StationaryPolicy S A)
    (g : S → A → ℝ) (t₀ : ℝ) (hπ : ∀ s a, HasDerivAt (fun t => (π t).prob s a) (g s a) t₀)
    (hpos : ∀ s a, 0 < (π t₀).prob s a) :
    HasDerivAt (fun t => M.startValue μ (π t))
      ((∑ s, M.occupancy μ (π t₀) s * ∑ a, (π t₀).prob s a *
        deriv (fun t => Real.log ((π t).prob s a)) t₀ * M.qValue (π t₀) s a) /
        (1 - (M.γ : ℝ))) t₀ := by
  simp only [prob_mul_deriv_log π g t₀ hπ hpos]
  exact M.hasDerivAt_startValue μ π g t₀ hπ

/-- **T5.** Book Theorem 9.4 in the book's own notation, with `∇ log π`, advantage form. -/
theorem hasDerivAt_startValue_advantage_log [Nonempty S] (μ : S → ℝ)
    (π : ℝ → StationaryPolicy S A) (g : S → A → ℝ) (t₀ : ℝ)
    (hπ : ∀ s a, HasDerivAt (fun t => (π t).prob s a) (g s a) t₀)
    (hpos : ∀ s a, 0 < (π t₀).prob s a) :
    HasDerivAt (fun t => M.startValue μ (π t))
      ((∑ s, M.occupancy μ (π t₀) s * ∑ a, (π t₀).prob s a *
        deriv (fun t => Real.log ((π t).prob s a)) t₀ * M.advantage (π t₀) s a) /
        (1 - (M.γ : ℝ))) t₀ := by
  simp only [prob_mul_deriv_log π g t₀ hπ hpos]
  exact M.hasDerivAt_startValue_advantage μ π g t₀ hπ

end FiniteMDP

/-! End of step 3. -/

/-! ## Step 4. Softmax policies (D9, L17) -/

/-- The softmax probabilities, `exp θ(s, a) / ∑ b, exp θ(s, b)`. -/
def softmaxProb (θ : S → A → ℝ) (s : S) (a : A) : ℝ :=
  Real.exp (θ s a) / ∑ b, Real.exp (θ s b)

theorem softmaxProb_pos (θ : S → A → ℝ) (s : S) (a : A) : 0 < softmaxProb θ s a := by
  unfold softmaxProb
  exact div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)

/-- **D9.** The softmax policy of book Example 9.1. -/
def StationaryPolicy.softmax (θ : S → A → ℝ) : StationaryPolicy S A where
  prob := softmaxProb θ
  nonneg s a := (softmaxProb_pos θ s a).le
  sum_eq_one s := by
    unfold softmaxProb
    rw [← Finset.sum_div]
    exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

/-- Along the path `θ + t e`, the softmax probabilities written with exponentials. -/
theorem softmaxProb_add_smul (θ e : S → A → ℝ) (t : ℝ) (s : S) (a : A) :
    softmaxProb (θ + t • e) s a =
      Real.exp (θ s a + t * e s a) / ∑ b, Real.exp (θ s b + t * e s b) := by
  simp only [softmaxProb, Pi.add_apply, Pi.smul_apply, smul_eq_mul]

/-- **L17.** The derivative of `log π_θ(a | s)` for the softmax policy along the path
`θ + t e`, in any direction `e`. It is `e(s, a)` minus the softmax average of `e(s, ·)`. -/
theorem hasDerivAt_log_softmaxProb (θ e : S → A → ℝ) (t₀ : ℝ) (s : S) (a : A) :
    HasDerivAt (fun t : ℝ => Real.log (softmaxProb (θ + t • e) s a))
      (e s a - ∑ b, softmaxProb (θ + t₀ • e) s b * e s b) t₀ := by
  -- the exponents move linearly with `t`
  have hlin : ∀ b, HasDerivAt (fun t : ℝ => θ s b + t * e s b) (e s b) t₀ := by
    intro b
    have h := ((hasDerivAt_id' t₀).mul_const (e s b)).const_add (θ s b)
    rwa [one_mul] at h
  -- the sum of the exponentials is differentiable and positive
  have hsum : HasDerivAt (fun t : ℝ => ∑ c, Real.exp (θ s c + t * e s c))
      (∑ c, Real.exp (θ s c + t₀ * e s c) * e s c) t₀ :=
    HasDerivAt.fun_sum fun c _ => (hlin c).exp
  have hpos : ∀ t : ℝ, 0 < ∑ c, Real.exp (θ s c + t * e s c) := fun _ =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  -- the log of a softmax probability is the exponent minus the log of the sum
  have hfun : (fun t : ℝ => Real.log (softmaxProb (θ + t • e) s a)) =
      fun t : ℝ => θ s a + t * e s a - Real.log (∑ c, Real.exp (θ s c + t * e s c)) := by
    funext t
    rw [softmaxProb_add_smul, Real.log_div (Real.exp_pos _).ne' (hpos t).ne', Real.log_exp]
  -- the derivative of the log of the sum is the softmax average of `e`
  have hval : (∑ c, Real.exp (θ s c + t₀ * e s c) * e s c) /
      (∑ c, Real.exp (θ s c + t₀ * e s c)) = ∑ b, softmaxProb (θ + t₀ • e) s b * e s b := by
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [softmaxProb_add_smul]
    ring
  have h := (hlin a).fun_sub (hsum.log (hpos t₀).ne')
  rw [hval] at h
  rw [hfun]
  exact h

/-- **L17.** The derivative of the softmax probabilities along the path `θ + t e`. It is
`π_θ(a | s)` times the derivative of its log, and it is the form that T6 puts into T5. -/
theorem hasDerivAt_softmaxProb (θ e : S → A → ℝ) (t₀ : ℝ) (s : S) (a : A) :
    HasDerivAt (fun t : ℝ => softmaxProb (θ + t • e) s a)
      (softmaxProb (θ + t₀ • e) s a *
        (e s a - ∑ b, softmaxProb (θ + t₀ • e) s b * e s b)) t₀ := by
  have h := (hasDerivAt_log_softmaxProb θ e t₀ s a).exp
  rw [Real.exp_log (softmaxProb_pos _ s a)] at h
  have hfun : (fun t : ℝ => Real.exp (Real.log (softmaxProb (θ + t • e) s a))) =
      fun t : ℝ => softmaxProb (θ + t • e) s a :=
    funext fun _ => Real.exp_log (softmaxProb_pos _ s a)
  rw [hfun] at h
  exact h

/-- The indicator of the state-action pair `(s', a')`, the direction of the coordinate
`θ(s', a')`. -/
def pairIndicator (s' : S) (a' : A) : S → A → ℝ := fun s a => if s = s' ∧ a = a' then 1 else 0

/-- The softmax average of the indicator of `(s', a')` at the state `s`. -/
theorem sum_softmaxProb_mul_pairIndicator (θ : S → A → ℝ) (s' : S) (a' : A) (s : S) :
    ∑ b, softmaxProb θ s b * pairIndicator s' a' s b =
      if s = s' then softmaxProb θ s a' else 0 := by
  by_cases hs : s = s'
  · subst hs
    simp [pairIndicator, mul_ite]
  · simp [pairIndicator, hs]

/-- **L17.** The book's coordinate formula, Section 10.2. Moving only `θ(s', a')`, the
derivative of `log π_θ(a | s)` is `1[s = s'] (1[a = a'] - π_θ(a' | s))`. -/
theorem hasDerivAt_log_softmaxProb_coord (θ : S → A → ℝ) (s' : S) (a' : A) (s : S) (a : A) :
    HasDerivAt (fun t : ℝ => Real.log (softmaxProb (θ + t • pairIndicator s' a') s a))
      ((if s = s' then 1 else 0) * ((if a = a' then 1 else 0) - softmaxProb θ s a')) 0 := by
  have h := hasDerivAt_log_softmaxProb θ (pairIndicator s' a') 0 s a
  simp only [zero_smul, add_zero, sum_softmaxProb_mul_pairIndicator] at h
  have hval : pairIndicator s' a' s a - (if s = s' then softmaxProb θ s a' else 0) =
      (if s = s' then 1 else 0) * ((if a = a' then 1 else 0) - softmaxProb θ s a') := by
    by_cases hs : s = s' <;> by_cases ha : a = a' <;> simp [pairIndicator, hs, ha]
  rw [hval] at h
  exact h

/-! End of step 4. -/

/-! ## Step 5. The softmax policy gradient (T6) -/

/-- Summing against the indicator of `(s', a')` picks out the action `a'`, and only at the
state `s'`. -/
theorem sum_mul_pairIndicator (w : A → ℝ) (s' : S) (a' : A) (s : S) :
    ∑ a, w a * pairIndicator s' a' s a = if s = s' then w a' else 0 := by
  by_cases hs : s = s'
  · subst hs
    simp [pairIndicator, mul_ite]
  · simp [pairIndicator, hs]

namespace FiniteMDP

variable (M : FiniteMDP S A)

/-- The advantage of a policy averages to zero under that same policy. -/
theorem avg_advantage_self (π : StationaryPolicy S A) (s : S) :
    π.avg (M.advantage π) s = 0 := by
  rw [M.avg_advantage_eq π π s]
  simp

/-- The inner sum of T6 at one state. Only the state `s'` survives, with the term
`π_θ(a' | s') A(s', a')`. -/
theorem sum_softmax_deriv_mul_advantage (θ : S → A → ℝ) (s' : S) (a' : A) (s : S) :
    ∑ a, softmaxProb θ s a *
        (pairIndicator s' a' s a - if s = s' then softmaxProb θ s a' else 0) *
        M.advantage (StationaryPolicy.softmax θ) s a =
      if s = s' then softmaxProb θ s a' * M.advantage (StationaryPolicy.softmax θ) s a'
      else 0 := by
  -- the advantage averages to zero under the softmax policy itself
  have hzero : ∑ a, softmaxProb θ s a * M.advantage (StationaryPolicy.softmax θ) s a = 0 :=
    M.avg_advantage_self (StationaryPolicy.softmax θ) s
  -- split off the part of the derivative that does not depend on the action
  have hsplit : ∑ a, softmaxProb θ s a *
        (pairIndicator s' a' s a - if s = s' then softmaxProb θ s a' else 0) *
        M.advantage (StationaryPolicy.softmax θ) s a =
      ∑ a, softmaxProb θ s a * M.advantage (StationaryPolicy.softmax θ) s a *
          pairIndicator s' a' s a -
        (if s = s' then softmaxProb θ s a' else 0) *
          ∑ a, softmaxProb θ s a * M.advantage (StationaryPolicy.softmax θ) s a := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    ring
  rw [hsplit, hzero, mul_zero, sub_zero, sum_mul_pairIndicator]

/-- **T6.** The softmax policy gradient, book Lemma 10.2. Moving only `θ(s', a')`, the value
from `μ` has derivative `d(s') π_θ(a' | s') A(s', a') / (1 - γ)`, where `d` is the occupancy
of the softmax policy from `μ` and `A` is its advantage. -/
theorem hasDerivAt_startValue_softmax [Nonempty S] (μ : S → ℝ) (θ : S → A → ℝ) (s' : S)
    (a' : A) :
    HasDerivAt
      (fun t : ℝ => M.startValue μ (StationaryPolicy.softmax (θ + t • pairIndicator s' a')))
      (M.occupancy μ (StationaryPolicy.softmax θ) s' * softmaxProb θ s' a' *
        M.advantage (StationaryPolicy.softmax θ) s' a' / (1 - (M.γ : ℝ))) 0 := by
  -- the derivatives of the softmax probabilities in the direction of `θ(s', a')`, by L17
  have hπ : ∀ s a, HasDerivAt
      (fun t : ℝ => (StationaryPolicy.softmax (θ + t • pairIndicator s' a')).prob s a)
      (softmaxProb θ s a *
        (pairIndicator s' a' s a - if s = s' then softmaxProb θ s a' else 0)) 0 := by
    intro s a
    have h := hasDerivAt_softmaxProb θ (pairIndicator s' a') 0 s a
    simp only [zero_smul, add_zero, sum_softmaxProb_mul_pairIndicator] at h
    exact h
  -- T5 in the advantage form, at `t = 0`
  have h := M.hasDerivAt_startValue_advantage μ
    (fun t : ℝ => StationaryPolicy.softmax (θ + t • pairIndicator s' a')) _ 0 hπ
  simp only [zero_smul, add_zero] at h
  refine h.congr_deriv ?_
  -- only the pair `(s', a')` survives in the occupancy average
  simp only [M.sum_softmax_deriv_mul_advantage θ s' a']
  have hX : (if s' = s' then softmaxProb θ s' a' *
      M.advantage (StationaryPolicy.softmax θ) s' a' else 0) =
      softmaxProb θ s' a' * M.advantage (StationaryPolicy.softmax θ) s' a' := by
    simp
  rw [Finset.sum_eq_single s', hX]
  · ring
  · intro b _ hb
    simp [hb]
  · intro h'
    exact absurd (Finset.mem_univ s') h'

end FiniteMDP

/-! End of step 5. -/

end Learning
