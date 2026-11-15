# Variables and coding used by the illustrative logistic models

The following derived binary variables reproduce the definitions used in the analysis notebook. `1` indicates the condition listed; `0` indicates the alternative category unless otherwise noted.

| Derived variable | Source variable | Value coded as 1 |
|---|---|---|
| `phq_moderate_severe` | `phq_total` | Total score >= 10 |
| `gad_moderate_severe` | `gad_total` | Total score >= 10 |
| `female` | `sex` | `Female` |
| `age_35plus` | `age_group` | `35-44`, `45-54`, `55-64`, `>= 65` |
| `sleep_falling_often` | `sleep_diff_falling_asleep` | `Often`, `Almost every day` |
| `high_functional_impact` | `functional_impact` | `Very difficult`, `Extremely difficult` |
| `poor_role_functioning` | `role_functioning` | `Very difficult`, `Can not do at all` |
| `low_life_satisfaction` | `life_satisfaction` | `Dissatisfied`, `Very dissatisfied` |
| `low_social_support_talk` | `social_support_talk` | `Never`, `Sometimes` |
| `financial_stress` | `stressor_financial` | `Yes` |
| `bereavement_stress` | `stressor_bereavement` | `Yes` |
| `legal_police_stress` | `stressor_legal_police` | `Yes` |
| `very_slow_speech` | `sb_very_slow_speech` | `Yes` |

## Fixed predictor sets

PHQ-9 outcome model: `female`, `age_35plus`, `sleep_falling_often`, `high_functional_impact`, `financial_stress`, `low_social_support_talk`, `very_slow_speech`.

GAD-7 outcome model: `female`, `age_35plus`, `poor_role_functioning`, `low_life_satisfaction`, `financial_stress`, `bereavement_stress`, `legal_police_stress`, `sleep_falling_often`.

These are explanatory/illustrative associations and are not validated clinical screening models. The exact category labels above should be checked against the released codebook before each new reuse.
