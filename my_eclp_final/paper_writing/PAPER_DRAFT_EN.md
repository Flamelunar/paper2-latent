# Learning to Revise Latent Strategies from Feedback for Multi-Turn Persuasion

**Working manuscript, 7 October 2026.** This draft describes ECLP-Offline-LPO with persuasion-only public output. The method and data descriptions follow the inspected implementation. `TBD` marks evidence that must be supplied before submission. Planned studies in Section 6 have not been reported as completed experiments. Citation keys resolve to `references.bib`.

## Abstract

Multi-turn persuasion requires a policy to adjust its strategy as the other speaker reveals new concerns. Fitting reference responses provides supervision for fluent replies, but does not directly identify which latent strategies benefit from correcting an earlier expectation. We propose Evidence-Corrected Latent Persuasion (ECLP), which predicts feedback features, updates a continuous dialogue state with the observed prediction error, and conditions response generation on a latent strategy. After joint supervised training, we freeze the language executor and state modules and optimize the continuous-action policy using existing training responses. The reward combines their average negative log-likelihood with a bounded correction benefit centered on a fixed supervised reference policy. This optimization stage uses teacher-forced scoring and requires no additional dialogue rollouts or reward-judge calls. We specify an evaluation on 192 fixed scenarios with no initial dialogue history, three language-model backbones, and two distinct judge models. State and reward ablations test whether explicit error correction and its offline objective improve goal acceptance beyond the same supervised starting point. **[TBD: replace the final experimental-design statement with measured paired differences, uncertainty, and the central ablation finding once the matched runs are complete.]**

## 1. Introduction

Persuasive dialogue asks a system to help another speaker adopt a specified decision or position through a sequence of exchanges. A useful response must address what the speaker currently cares about, and these concerns often become clear only after the conversation begins. A person may initially reject a training course without explaining why. After a response about affordability, they may clarify that the real obstacle is a conflict with their work schedule. The next response should revise the system's representation of the obstacle and discuss scheduling. Repeating the affordability argument would miss the information revealed by the latest reply. This example illustrates the need to connect observed feedback to a change in strategy.

Existing persuasion methods already use dialogue context and intermediate reasoning. PersuGPT trains intent-to-strategy reasoning and uses simulated continuations to construct preference supervision [@jin-etal-2024-persuading]. MA2P coordinates agents for perception, mental-state inference, planning, execution, and memory, with a configurator that selects a meta-strategy from a structured knowledge base [@zhang-etal-2026-ma2p]. These approaches provide ways to reason about a speaker and plan a response. A complementary question is how to learn a continuous strategy update from the discrepancy between expected and observed feedback. Reading the latest reply and representing the error in a prior expectation provide different signals, whose contribution needs to be separated experimentally.

Continuous latent reasoning offers a representation in which such an update can be learned. Coconut feeds continuous hidden states back into a language model during reasoning [@hao-etal-2024-coconut]. LatentR3 combines latent sampling with a teacher-forced perplexity reward for recommendation [@zhang-etal-2026-latentr3]. These studies motivate latent computation and offline answer scoring. Multi-turn persuasion additionally requires a state that can change after each observed reply. Reference-response likelihood alone does not tell the policy whether an action uses that correction effectively.

We introduce ECLP to connect feedback prediction, state revision, and latent strategy learning. After each public persuader utterance, a transition module predicts a feature representation of the next persuadee reply. Once that reply arrives, a gated update combines its content with the prediction error. Two state tokens contextualize a continuous-action policy, and two strategy tokens condition a LoRA language executor. Joint supervised training learns the executor, state update, and policy. A subsequent offline stage freezes the executor and state and updates only the policy. Its reward includes a reference-centered estimate of how much the latest error correction helps the same action explain an existing training response.

The evaluation is designed to separate three questions: whether explicit prediction error helps beyond reading the same reply, whether the correction reward helps beyond negative log-likelihood, and whether offline policy optimization helps beyond the identical ECLP supervised checkpoint. We use fixed scenarios that begin without dialogue history, retain the newly generated exchanges, and assess both exact-goal acceptance and preference relative to reference dialogues. **[TBD: insert the final main comparison and mechanism results here. This paragraph must be revised if any hypothesis is unsupported.]**

Our contributions are a continuous state update driven by observed feedback error, an offline latent-policy objective that measures the usefulness of that correction, and an evaluation protocol that distinguishes goal acceptance from reference-dialogue preference. The empirical contribution will be stated after completing the matched comparisons and ablations.

## 2. Related Work

### 2.1 Multi-turn persuasion and strategy reasoning

DailyPersuasion provides multi-domain persuasive dialogues, and PersuGPT uses intent-to-strategy reasoning to connect a user's utterance with a response strategy [@jin-etal-2024-persuading]. Its simulation-based preference optimization evaluates candidate continuations using a learned user model. MA2P addresses complex persuasion with coordinated perception, inference, strategy execution, memory maintenance, and evaluation. Its meta-cognitive configurator uses a structured knowledge base to guide subsequent planning [@zhang-etal-2026-ma2p]. ECLP studies a learned continuous update between exchanges. The comparison targets this feedback-update mechanism rather than assuming that explicit reasoning methods ignore feedback. Our experimental MA2P baseline is a K=0 prompt adaptation with three roles, and must be distinguished from the complete published system.

ToMAP augments a persuader with possible objections and a model of the opponent's stance on counterclaims [@han-etal-2025-tomap]. It provides a closer precedent for opponent-aware training than a generic language-model baseline. ECLP represents feedback with a recurrent continuous state and uses predicted-versus-observed feature error to update that state. A comparison to ToMAP would additionally test the value of this representation against structured stance modeling.

### 2.2 Continuous latent reasoning

Coconut represents intermediate thoughts using language-model hidden states and feeds them back as continuous input embeddings [@hao-etal-2024-coconut]. LatentR3 applies reinforced latent reasoning to recommendation, using Gaussian latent exploration and a teacher-forced perplexity signal [@zhang-etal-2026-latentr3]. ECLP draws on continuous strategies and offline scoring, while adding a recurrent feedback state and an objective tied to an error-correction channel. Its inference interface maps an action to a fixed number of virtual strategy tokens. It does not implement Coconut's sequential continuous-thought chain. The local baseline name LatentRATT denotes the project's persuasion adaptation of LatentR3.

### 2.3 Policy optimization and supervision

Clipped policy optimization constrains updates through likelihood ratios [@schulman-etal-2017-ppo], and LoRA provides a parameter-efficient language executor [@hu-etal-2021-lora]. ECLP uses these tools with a continuous action distribution. The policy ratio is a joint Gaussian action-density ratio, whereas the inspected LatentR3 objective uses a target-token likelihood ratio. The offline reward evaluates how well a sampled strategy explains a training response under a frozen executor. Its relationship with acceptance in a generated dialogue is an empirical question tested separately.

LLM preference judges can exhibit position and verbosity biases [@zheng-etal-2023-judge]. This motivates reporting both candidate orders, response lengths, and human agreement. ECLP's goal-acceptance assessment uses evidence from the persuadee's current reply, while its preference assessment compares public text with a reference. These measurements capture different aspects of persuasion.

**Literature completion note:** the final related-work section should additionally cover verified work on feedback-based dialogue-state updates and recent latent-policy learning. The current citations establish the nearest inspected antecedents. Further novelty claims require a broader review of those areas.

## 3. Method

### 3.1 Task and observations

A scenario contains a background \(x\) and persuasion goal \(g\). At round \(t\), the public history is \(h_t=(y_1,u_1,\ldots,y_{t-1},u_{t-1})\), where \(y\) denotes persuader speech and \(u\) denotes persuadee feedback. The trained persuader receives \((x,g,h_t)\) and produces one public utterance \(y_t\). An evaluator later judges whether \(u_t\) accepts the exact goal. Neither future feedback nor reference test responses are available to the persuader.

The system maintains a task state \(B_t\in\mathbb R^{s\times w}\), with \(s=2\) and \(w=256\). This state summarizes evidence relevant to response selection. A frozen feature encoder supplies token features \(F(v)\) and an aggregate feature \(f(v)\in\mathbb R^{128}\) for public text. It uses the backbone with LoRA disabled, a fixed projection, and feature normalization. Initial slots attend to the background features to obtain \(B_1\).

### 3.2 Predicting and correcting feedback

After the persuader's utterance, a transition produces a prior state and predicts the next reply's aggregate feature:

\[
\bar B_{t+1}=\operatorname{LN}\left(B_t+T([B_t,\operatorname{broadcast}f(y_t)])\right),\qquad
\hat f_{t+1}=P\left(\operatorname{mean}_{slots}\bar B_{t+1}\right).
\tag{1}
\]

When feedback \(u_t\) is observed, the state reads its token features and forms a projected error:

\[
O_{t+1}=\operatorname{CrossAttn}(\bar B_{t+1},M(F(u_t))),\qquad
E_{t+1}=\operatorname{broadcast}\left(W_e[f(u_t)-\hat f_{t+1}]\right).
\tag{2}
\]

Let \(Z_{t+1}=[\bar B_{t+1},O_{t+1},E_{t+1}]\). The posterior update is

\[
B_{t+1}=\operatorname{LN}\left(\bar B_{t+1}+\sigma(G(Z_{t+1}))\odot C(Z_{t+1})\right).
\tag{3}
\]

The readout supplies the content of the actual reply, while the error supplies its discrepancy from the prior prediction. Their roles are separated by a no-error control that preserves the readout and update capacity and sets only \(E\) to zero. A no-correction control retains the prior; a stateless control reads the current reply into fresh slots.

### 3.3 Latent strategies and response execution

The state is mapped to two virtual tokens and appended to the public prompt. A frozen backbone, with LoRA disabled for this context computation, encodes the prefix. The policy reads the state-token hidden states \(c_t\), attends with learned queries, and outputs a mean action \(\mu_\phi(c_t)\in[-1,1]^{64}\) through a tanh head. A learned mapping converts an action into two strategy tokens, scaled to the backbone's embedding RMS. The LoRA executor generates speech from the prompt, state tokens, and strategy tokens.

At deployment, the system uses the mean action. Response tokens are sampled with the common decoding configuration. Latent action selection is deterministic for a fixed context, while the public utterance remains stochastic. No strategy-candidate search or intermediate public analysis is used at inference.

### 3.4 Joint supervised training

The first stage combines four losses:

\[
\mathcal L_{SFT}=\mathcal L_{public}+\mathcal L_{pred}
+0.1\mathcal L_{strategy}+0.1\mathcal L_{ISR}.
\tag{4}
\]

The public loss is the mean token NLL of the last persuader utterance in each prepared session. The prediction loss is the MSE between predicted and observed frozen feedback features, averaged over observed feedback in that session. The strategy loss aligns the mean strategy-token representation with a frozen encoding of the existing strategy annotation using cosine similarity. The ISR loss predicts the existing private reasoning annotation as an auxiliary teacher target. Public inference still produces speech alone.

The implementation maintains observation and policy state paths with shared parameters and selective gradient isolation. Public-response supervision trains the strategy and correction path while detaching the observation readout and feedback-prediction parameters along that path. Observation prediction and private auxiliary supervision use the corresponding observation path. The backbone weights are frozen and LoRA parameters are trained. The appendix must retain the exact gradient-flow specification because the state module is jointly trained, rather than frozen throughout SFT.

### 3.5 Offline correction reward

After SFT, the language executor, LoRA, state modules, and action-token mapping are frozen. Only the policy is updated. For an existing context \(c\), four actions are drawn from

\[
\pi_{old}(a\mid c)=\mathcal N(\mu_{old}(c),\Sigma),\qquad
\Sigma=\frac{r^2}{d}I,\quad r=0.5,\ d=64.
\tag{5}
\]

A fixed supervised reference supplies \(a_{ref}=\mu_{SFT}(c)\). Sampled actions are scored by teacher forcing the existing training response \(y^*\), including its EOS:

\[
\ell_f(a)=-\frac{1}{|y^*|}\sum_j\log p_{exec}(y_j^*\mid x,g,h,B_f,a,y^*_{<j}).
\tag{6}
\]

We construct an alternate state \(B_{cf}\) by replaying the same public history and setting the prediction-error channel to zero only at the latest observed persuadee reply. Its actual reply readout remains available. The identical action and target response are scored in both branches. Define \(d(a)=\ell_{cf}(a)-\ell_f(a)\). The reward is

\[
R(a)=-\ell_f(a)+\lambda\tanh\left[d(a)-d(a_{ref})\right],\qquad\lambda=0.1.
\tag{7}
\]

The first term is negative log-perplexity and rewards reference-response fit. The second measures the additional usefulness of the latest error correction relative to the fixed SFT action in the same context. Its bounded magnitude limits its contribution to the total score. The reference action has zero correction bonus. For contexts with no observed feedback, the bonus is zero. This benefit measures internal model sensitivity to the correction channel. Causal persuasion effects on people require separate interaction evidence.

### 3.6 Continuous-action policy update

For each candidate, \(\Delta_i=R(a_i)-R(a_{ref})\). The collection batch supplies a standard deviation \(s_\Delta\), and the normalized advantage is

\[
A_i=\operatorname{clip}\left(\frac{\Delta_i}{\max(s_\Delta,10^{-3})},-10,10\right).
\tag{8}
\]

Using the joint Gaussian density,

\[
q_i(\phi)=\exp\left[\log\pi_\phi(a_i\mid c)-\log\pi_{old}(a_i\mid c)\right],
\]
\[
\mathcal L_{LPO}=-\mathbb E_i\min\left(q_i A_i,\operatorname{clip}(q_i,1-\epsilon,1+\epsilon)A_i\right)
+\beta\mathbb E_c D_{KL}(\pi_\phi\Vert\pi_{SFT}),\quad \epsilon=0.2,\ \beta=0.02.
\tag{9}
\]

The soft KL reference is the fixed SFT policy. A separate hard guard bounds the maximum per-context KL to the behavior policy that collected the current batch at 0.1. Updates exceeding this bound are rolled back and retried at a reduced learning rate. The cumulative KL to the fixed SFT policy may exceed the behavior bound.

We collect 64 contexts per update, use two policy-update epochs with batch size 64, and make one pass over the 1,800 offline training contexts. The final checkpoint is the last committed checkpoint after this pass. Test outcomes do not select it. This stage invokes neither autoregressive dialogue generation, a persuadee simulator, nor a reward judge.

## 4. Experimental Setup

### 4.1 Data and fixed evaluation scenarios

The project reconstructs training and held-out partitions from the DailyPersuasion/PersuGPT material [@jin-etal-2024-persuading]. The SFT partition contains 10,000 scenarios and 59,868 prepared session targets. Feedback prediction involves 249,720 exposures during session replay. Shared source prefixes can recur across sessions. A separate 2,000-scenario partition is divided into 1,800 offline training contexts and 200 development contexts. Of the training contexts, 1,455 contain observed feedback and 345 do not. The development set has 170 and 30, respectively.

The source test partition contains 1,000 scenarios. All experiments in this study use the fixed 192 whose original target turn is one and whose original history is empty. The original indices are shared by all methods and judges. The reconstructed split is a project protocol rather than an exact reproduction of unpublished split identifiers from the source paper. **[TBD: include the completed scenario-overlap and near-duplicate audit, source version, and license details.]**

### 4.2 Backbones and implementations

The planned backbone comparison uses Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct, and DeepSeek-R1-Distill-Qwen-14B in BF16 without quantization. LoRA has rank 16, alpha 32, and dropout zero. SFT runs for one epoch at learning rate \(10^{-4}\), with gradient accumulation of eight and a target of 7,484 optimizer steps. Offline policy learning uses rate \(10^{-5}\). Each action group contains four samples and the fixed reference action. Teacher-forced scoring uses batch size one.

The SFT prompt budget is 1,536 tokens. Current family inference retains the public prompt up to 6,144 tokens and raises an error on overflow. The feature encoder retains context and reply budgets of 1,536 and 256 tokens. This training/inference difference is disclosed, and the final comparison must report each actor's actual visible-history budget. All methods use temperature 0.95, top-p 0.7, top-k 50, and at most 512 new response tokens where their interfaces permit the common configuration.

### 4.3 Baselines and controls

**Zero-shot** uses the untuned backbone with background and public history, following the user's background-only protocol. **Goal-aware untuned** is a proposed additional control that receives the same explicit goal as trained persuaders. The input distinction is necessary when attributing training gains.

**Dialogue-SFT** trains a response model on dialogue supervision. **PersuGPT** combines ISR SFT and simulation-based preference optimization. Their complete family-specific checkpoints must be identified before reporting a run. **MA2P** uses a three-role K=0 prompt adaptation with perception, world-model reasoning, and persuasion execution. It omits the published framework's retrieval and full memory machinery. **LatentRATT** adapts LatentR3 to persuasion. Its current local training protocol uses all 59,868 ISR records in both SFT and latent optimization. We disclose this budget difference and propose an additional data-matched control rather than labeling the native methods budget-matched.

**ECLP-SFT-only** is the complete ECLP architecture at exactly the supervised starting checkpoint, before policy optimization. It preserves all state and strategy modules at inference. **ECLP-Offline-LPO** applies the second training stage to that checkpoint and returns persuasion-only public speech. Dialogue-SFT cannot replace ECLP-SFT-only for attributing the effect of the offline stage.

**[TBD: populate a baseline implementation table with actual checkpoint hashes, trainable parameters, training contexts, updates, and per-round model calls. Historical conversations from a different simulator contract must remain outside this matched main table.]**

### 4.4 Dialogue generation and judging

Every scenario begins with empty public history. Each method generates five persuader–persuadee exchanges with the same locally trained Qwen3.5-9B persuadee simulator. The repaired simulator sees the background and generated public history, while the exact persuasion goal is withheld. The trained persuader sees the goal. Private analysis, original strategy annotations, reference replies, and future turns never enter the simulator's public history. Raw output is archived, and only the current speaker's public utterance is retained.

Selene-1-Mini-Llama-3.1-8B and Qwen3-14B judge the same dialogues using the same prompt rules. For acceptance, a judge first quotes evidence from the current persuadee reply and emits JSON containing `evidence`, `progress`, and a boolean `success`. Success requires acceptance of the exact goal, including all required parts. Interest or a preliminary action does not establish acceptance of a broader commitment. Schema constraints enforce output types. Human calibration assesses the semantic decisions.

### 4.5 Metrics and uncertainty

**Goal acceptance within five rounds** is the fraction of scenarios with at least one accepted round. **Mean rounds to success** uses the earliest accepted round, assigning five to failures. We additionally analyze final-round acceptance and acceptance retractions. Unresolved judgments are reported with metric-specific coverage and fixed-denominator lower and upper bounds.

**Win@1** compares the first public persuader utterance with the first reference utterance. **Win-Full** compares the complete generated conversation with all available reference exchanges. Reference dialogues can have different lengths. Candidate order is swapped to assess position bias. A win, tie, or loss contributes one, one-half, or zero, and the two order scores are averaged for the scenario. Thus the reported percentage is preference credit including ties. We also report strict win/tie/loss proportions and order-conflict rates.

Uncertainty is estimated by paired resampling of scenario identifiers, retaining all rounds and both candidate orders within each sampled scenario. The primary contrasts are fixed in advance. **[TBD: report 95% intervals, the main-model training seed count, and separate generation seeds. Swapped judge calls do not constitute independent experimental seeds.]**

## 5. Results

### 5.1 Main comparison

**[Table 1 pending.]** Report every method under each backbone and judge on the same complete 192-scenario contract. Lead with the paired Success difference between full ECLP and ECLP-SFT-only, followed by the strongest matched baseline. State the interval and whether the judges agree. Report Win@1 and Win-Full separately, including any decline. Replace this paragraph with measurements and supported interpretation once the full runs are available.

Do not copy a historical Qwen score into this table when its dialogue simulator or generation contract differs. Partial progress summaries describe execution, not final evaluation evidence.

### 5.2 State and reward attribution

**[Table 2 pending.]** Compare retrained error-on and no-error models that read the same actual feedback, using NLL-only policy optimization for both. The correction bonus is zero when the error channel is absent, so comparing a no-error model directly with full ECLP changes both the state and the effective reward. Separately compare full reward with NLL-only policy optimization from the identical error-on SFT checkpoint. ECLP-SFT-only isolates the value of the complete second stage. A matched continued-SFT or direct likelihood-tuning control on the same 1,800 contexts tests whether additional answer fitting explains the outcome.

The interpretation follows the observed contrasts. A state benefit without a reward benefit supports feedback correction but leaves the new reward unverified. Improved offline NLL without improved acceptance supports proxy optimization, rather than successful persuasion.

### 5.3 Efficiency

**[Table 5 and Figure 4 pending.]** Measure SFT and offline optimization separately on the same GPU hardware. Report elapsed time, GPU hours, peak memory, generated tokens, simulator calls, and judge calls. The offline stage has zero new rollout and judge calls by construction, but its teacher-forced forwards still have a cost. Compare total cost and task quality rather than equating reduced dialogue calls with a measured runtime improvement.

## 6. Planned Analyses

### 6.1 Adaptation across rounds

Plot cumulative acceptance for rounds one through five for full ECLP, its SFT-only checkpoint, no-error, and NLL-only reward. Confidence intervals use scenario-level paired resampling. Report final acceptance and cases in which a speaker accepts and later retracts. This distinguishes an early temporary acceptance from sustained agreement while preserving the primary any-round metric.

### 6.2 Proxy reward and observed outcomes

On fixed development contexts, record candidate NLL, correction benefit, and total reward. A separate development analysis can generate subsequent dialogue to test their association with goal acceptance. Use reward quantiles, rank correlations, and uncertainty. Keep these measurements out of test-based hyperparameter selection. Without such continuations, report only the internal reward diagnostics.

### 6.3 Feedback error and strategy revision

Record feedback-error norm, gate values, state change, and policy-mean change for observed replies. Analyze their relationship after accounting for round, response length, and domain. These measurements test whether the update is used. They do not, by themselves, establish psychological interpretation of individual latent dimensions.

### 6.4 Evaluation reliability and length effects

Compare both judges on identical public conversations and conduct blinded human calibration on a fixed representative scenario subset. Report goal and preference agreement, order conflicts, ties, and evidence-quotation validity. Analyze Win-Full by reference-dialogue length and report generated response lengths. A comparison limited to the common number of turns is a sensitivity study, while the primary full-dialogue metric remains fixed.

### 6.5 Quality and failure cases

Annotate unsupported factual claims, invented arrangements or commitments, repeated pressure, goal drift, and failure to respond to a newly revealed concern. Select both successful and unsuccessful cases using stated criteria. Show public transcript excerpts alongside feedback-error and state-change diagnostics, and distinguish a useful update from a merely large update.

## 7. Limitations and Responsible Use

The fixed no-history subset contains 192 scenarios and covers a restricted slice of the source benchmark. A simulated speaker's acceptance is a model-based outcome and requires separate human validation. Offline optimization depends on the quality of the existing training replies. Increasing their likelihood can favor style or reference-specific arguments. The learned state has no independent validation as a representation of human beliefs. Training and inference context budgets differ, and reference lengths and judge position conflicts can affect preference scores.

Persuasion quality also depends on factual accuracy and respect for the other speaker's choice. Acceptance is insufficient as a quality criterion for high-stakes advice or coercive interactions. The proposed quality audit records unsupported claims and pressure alongside task outcomes. The final manuscript must document data provenance, licensing, simulator identity, and the intended scope of deployment.

## 8. Conclusion

ECLP connects a prediction of public feedback with an error-driven latent state update and a continuous strategy policy. Its offline stage scores existing replies under a frozen executor and rewards reference-centered use of the latest correction. **[TBD: complete the conclusion with the measured scope of improvement and supported ablation findings. If the correction reward fails to improve acceptance, retain that finding and narrow the conclusion accordingly.]**

## Appendix contents to prepare

- Full generation, simulator, success, and preference prompts; constrained output schemas.
- Fixed scenario indices and hashes; split reconstruction and overlap/near-duplicate audit.
- Per-family checkpoint/module identities, exact trainable parameters, and gradient-flow specification.
- Complete hyperparameters, context budgets, decoding rules, and training/generation seeds.
- Complete dual-judge main and ablation tables, coverage, pure win/tie/loss, and order conflicts.
- Human calibration protocol and annotation examples; domain and goal-type counts.
- Per-stage timing and memory logs, plot source data, and representative failure cases.
