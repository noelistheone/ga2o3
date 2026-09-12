export const meta = {
  name: 'hybrid-ml-sandbox-design',
  description: 'Research + design-panel + adversarial critique for the optimal ML-model + physics-sandbox hybrid to reach lab-usable accuracy on 9 Ga2O3 doping properties',
  phases: [
    { title: 'Ground', detail: '5 parallel literature angles on physics-sim + ML hybrids' },
    { title: 'DesignPanel', detail: '4 independent hybrid-architecture proposals' },
    { title: 'Critique', detail: 'adversarial critique of each proposal' },
    { title: 'Synthesize', detail: 'the winning combined design + per-property plan + formula' },
  ],
}

const CONTEXT = `OUR SYSTEM. A first-principles defect-thermodynamics SANDBOX for beta-Ga2O3 doping: KROGER HSE defect DB (873 charge-states/259 defects/19 elements) + own SGTE thermochemistry -> charge-neutrality (Brouwer) solver -> quench -> transport (Caughey-Thomas mu(N_I) fed by solver, VALIDATED <=11% vs Hall) + optical (Burstein-Moss + BGR + NEW computed disorder-narrowing) + device layers -> emits 9 properties: PDR, dark current, [V_O], hall_n, hall_mu, Eg, dEg, tau, sigma. NEW capability: MACE-MPA-0 (MP-pretrained MLIP) computes doped/new-dopant STRUCTURE (validated 0.026A vs QE); new dopants beyond the 19 (Sb/Bi) via MACE->HSE->inject->verdict (Sb/Bi=deep donors, experimentally corroborated).

SANDBOX ACCURACY NOW: within-series doping response can be excellent (Si conc-series Spearman 0.99) but inconsistent (Ge -0.62); mu transport <=11%; donor absolute n <0.2 orders after calibrating ONE freeze-in-T knob; ABSOLUTES hit the exponential-sensitivity ceiling (+-0.5 eV HSE -> 10^7 in concentration); ~4 process nuisance params (effective freeze-in T, effective pO2, background Nd_bg~1e17, Seto grain E_B). dEg: BM widening +0.23 competes with computed disorder narrowing -0.23..-0.55 eV.

ML SIDE (Ga2O3-Net, classical ML on 1891-row literature corpus). FAILURES: pooled cross-paper Spearman only PDR 0.29 / dark 0.29 / hall_n 0.34 / tau 0.37 / V_O 0.42 / hall_mu 0.46 / Eg 0.47 / dEg 0.62; pooled rho does NOT transport to a new lab (between-lab confound, REML ICC 58-85%); new-dopant LOEO barely beats 1NN floor; data-saturated (20x data>methods); ONLY within-paper DELTA (dEg) and measurement-covariate models beat the confound. Available MP-pretrained models: MACE-MPA-0/CHGNet/ORB (structure), cached LLM donor/acceptor sign priors (validated 15/15), general Qwen LLMs.

GOAL: the PERFECT ML+sandbox hybrid -> lab-USABLE absolute accuracy on all 9 properties for OLD and NEW dopants, with every value GENUINELY SIMULATED by the sandbox (no table-lookup / no pure-ML memorization). Use ML failures + sandbox limits to COMPLEMENT each other.`

const ANGLES = [
  { key: 'physics-sim-plus-ml', prompt: `Research the STATE OF THE ART in combining a PHYSICS SIMULATOR (first-principles / defect-thermodynamics / process simulator) with MACHINE LEARNING to predict experimentally-measured semiconductor/materials properties. Cover: (1) Delta-learning / residual learning where ML learns (experiment - simulator); (2) Kennedy-OHagan Bayesian model-discrepancy calibration (physics model + GP discrepancy + calibrated parameters); (3) physics-informed / physics-guided neural nets using simulator outputs as features or constraints; (4) multi-fidelity ML (many cheap sim + few expensive/experimental labels); (5) any application to DEFECTS, DOPING, TCAD, or wide-gap semiconductors (Ga2O3, GaN, SiC). For each: does it improve ABSOLUTE accuracy and TRANSPORTABILITY to new conditions/labs vs ML-alone and sim-alone? Concrete papers, error reductions, URLs.` },
  { key: 'discrepancy-and-confound', prompt: `Research how to combine a BIASED-but-mechanistic physics model with SPARSE, CONFOUNDED experimental data to get TRANSPORTABLE absolute predictions WITHOUT the ML re-learning the confound. Cover: (1) Kennedy-OHagan / model-discrepancy frameworks and identifiability (separating calibration params from discrepancy); (2) hierarchical/mixed-effects models that put the between-lab confound in a RANDOM effect so the FIXED (physics) part transports; (3) domain-adaptation / invariant-risk approaches to remove batch/lab effects; (4) how to constrain the ML residual to be SMOOTH/physical (small, monotone) so it corrects sim bias without memorizing lab identity. Our ML confound is the LAB effect (REML ICC 58-85%). How do people guarantee the learned correction is physics not lab-memorization? Concrete methods + URLs.` },
  { key: 'calibrate-nuisance', prompt: `Research BAYESIAN CALIBRATION of a physics/process simulator nuisance parameters against experimental data for defect/doping/growth models. Our sandbox has ~4 nuisance params (effective freeze-in temperature, effective oxygen partial pressure, background donor density, grain-boundary barrier). Cover: (1) how to fit these per-lab vs globally from sparse data (history matching, ABC, MCMC, emulator-based calibration); (2) whether calibrating a FEW physical knobs (vs a black-box ML) gives transportable absolutes; (3) identifiability / equifinality when params trade off; (4) examples in semiconductor defect equilibria, sputter deposition, TCAD. Concrete methods, how many data points needed, URLs.` },
  { key: 'per-property-fusion', prompt: `Research property-specific fusion of physics + ML for: carrier density n, mobility mu, conductivity sigma, band gap Eg, doping gap-shift dEg, oxygen-vacancy concentration, deep-trap/PPC lifetime tau, dark current, photo-dark ratio. For EACH, the best-known route to LAB-ACCURATE prediction and where physics vs data-correction dominates? E.g. mobility via Boltzmann/AMSET + ML scattering corrections; n via defect equilibrium + calibrated activation; Eg/dEg via GW/BSE or BM+disorder + ML residual; tau via nonradiative-capture theory + ML. Which properties are physics-limited (need better DFT) vs data-limited (need calibration)? Accuracy targets that count as lab-usable per property + URLs.` },
  { key: 'uq-and-lab-usable', prompt: `Research how to QUANTIFY and CERTIFY that a hybrid physics+ML materials predictor is lab-usable. Cover: (1) calibrated uncertainty (conformal prediction, GP posterior, deep ensembles) for physics+ML hybrids; (2) per-property error bars that hold out-of-distribution / for new dopants; (3) leave-one-lab-out / leave-one-dopant-out validation protocols certifying transportability; (4) what absolute accuracy (%, orders of magnitude, eV) is the accepted usable threshold for carrier density, mobility, band gap, lifetime in Ga2O3 device work; (5) active-learning to reach the threshold with minimal new experiments. Concrete protocols, thresholds, URLs.` },
]

phase('Ground')
const ground = await parallel(ANGLES.map(a => () =>
  agent(`Materials-informatics + Bayesian-modeling research specialist. Use WebSearch + WebFetch THOROUGHLY (arXiv, npj Comput Mater, Nature, JAP, APL, Technometrics/JASA for calibration, TCAD venues). Extract concrete falsifiable claims with numbers + source URL. Prioritize 2018-2026.\n\n${CONTEXT}\n\nQUESTION:\n${a.prompt}\n\nReturn structured findings (claim, numbers, url, confidence) + a bottom_line + what is uncertain.`,
    { label: `ground:${a.key}`, phase: 'Ground', schema: { type: 'object', properties: {
      findings: { type: 'array', items: { type: 'object', properties: {
        claim: { type: 'string' }, numbers: { type: 'string' }, url: { type: 'string' },
        confidence: { type: 'string', enum: ['high', 'medium', 'low'] } }, required: ['claim', 'url', 'confidence'] } },
      bottom_line: { type: 'string' }, uncertainties: { type: 'array', items: { type: 'string' } },
    }, required: ['findings', 'bottom_line'] } }
  ).then(r => ({ angle: a.key, ...r }))
))
const groundText = ground.filter(Boolean).map(g => `### ${g.angle}\nBOTTOM LINE: ${g.bottom_line}\nKEY FINDINGS:\n${(g.findings||[]).slice(0,8).map(f => `- ${f.claim} [${f.numbers||''}] ${f.url}`).join('\n')}`).join('\n\n')
log(`Ground research done (${ground.filter(Boolean).length} angles); running design panel`)

phase('DesignPanel')
const LENSES = [
  { key: 'residual-first', angle: 'Design the hybrid as DELTA-LEARNING FIRST: the sandbox physics prediction is the mean, a constrained ML learns the (experiment - sandbox) residual. Keep the residual small/physical/transportable and the final value genuinely-simulated.' },
  { key: 'calibration-first', angle: 'Design the hybrid as BAYESIAN CALIBRATION FIRST: fit the sandbox few physical nuisance params (freeze-in T, pO2, Nd_bg, E_B) to data (global + per-lab random effects) so the sandbox itself becomes lab-accurate, with ML only for leftover discrepancy. Emphasize identifiability.' },
  { key: 'gp-mean-first', angle: 'Design the hybrid as a GAUSSIAN-PROCESS with the sandbox as the physics MEAN FUNCTION + a discrepancy GP with a lab random effect (Kennedy-OHagan). Emphasize calibrated uncertainty + transportability certification.' },
  { key: 'per-property-specialist', angle: 'Design a PER-PROPERTY specialized hybrid: for each of the 9 properties choose the combination where physics vs data-correction dominates (n = calibrated equilibrium; mu = Boltzmann+ML scattering; dEg = BM+computed-disorder+ML residual; tau = capture-theory+ML). Use each side where it is strongest.' },
]
const proposals = await parallel(LENSES.map(l => () =>
  agent(`You are a senior architect designing the PERFECT ML-model + physics-sandbox hybrid for lab-usable 9-property Ga2O3 doping prediction.\n\n${CONTEXT}\n\nGROUNDING RESEARCH:\n${groundText}\n\nYOUR ASSIGNED LENS: ${l.angle}\n\nPropose a concrete, buildable architecture from THIS lens. Specify: the exact estimator/equations, what the sandbox computes vs what ML corrects, how it stays GENUINELY-SIMULATED (no table-lookup), how it avoids re-learning the lab confound, how new dopants are handled (where ML has no data), the training/validation protocol (leave-one-lab-out, leave-one-dopant-out), and the expected accuracy gain per property. Be concrete and honest about failure modes.`,
    { label: `design:${l.key}`, phase: 'DesignPanel', effort: 'high', schema: { type: 'object', properties: {
      architecture: { type: 'string' }, equations: { type: 'string' },
      genuinely_simulated_argument: { type: 'string' }, confound_avoidance: { type: 'string' },
      new_dopant_handling: { type: 'string' }, validation_protocol: { type: 'string' },
      per_property_gain: { type: 'string' }, failure_modes: { type: 'string' },
    }, required: ['architecture', 'equations', 'genuinely_simulated_argument', 'confound_avoidance', 'failure_modes'] } }
  ).then(r => ({ lens: l.key, ...r }))
))

phase('Critique')
const valid = proposals.filter(Boolean)
const critiques = await parallel(valid.map(p => () =>
  agent(`Adversarially critique this hybrid-architecture proposal for the Ga2O3 sandbox+ML fusion. Harsh reviewer. Check: (1) does the final prediction stay GENUINELY SIMULATED or sneak in table-lookup / pure-ML memorization (forbidden)? (2) does the ML residual re-introduce the lab confound (memorize lab identity)? (3) is it identifiable / not over-parameterized given sparse data? (4) does it actually reach LAB-USABLE absolute accuracy or just re-skin the trend-only result? (5) does it handle NEW dopants where ML has zero data? (6) is the physics sound?\n\nPROPOSAL (${p.lens}):\n${JSON.stringify(p, null, 2)}\n\nReturn fatal_flaws, salvageable_strengths, verdict (adopt/adapt/reject) + reasoning.`,
    { label: `critique:${p.lens}`, phase: 'Critique', schema: { type: 'object', properties: {
      fatal_flaws: { type: 'array', items: { type: 'string' } },
      salvageable_strengths: { type: 'array', items: { type: 'string' } },
      verdict: { type: 'string', enum: ['adopt', 'adapt', 'reject'] }, reasoning: { type: 'string' },
    }, required: ['fatal_flaws', 'salvageable_strengths', 'verdict', 'reasoning'] } }
  ).then(c => ({ lens: p.lens, ...c }))
))

phase('Synthesize')
const synth = await agent(`You are the lead architect. Synthesize the FINAL hybrid ML+sandbox design for lab-usable 9-property Ga2O3 doping prediction, drawing the best from all proposals and respecting every critique.\n\n${CONTEXT}\n\nPROPOSALS:\n${JSON.stringify(valid.map(p => ({ lens: p.lens, architecture: p.architecture, equations: p.equations })), null, 2)}\n\nCRITIQUES:\n${JSON.stringify(critiques.filter(Boolean), null, 2)}\n\nDeliver a CONCRETE, BUILDABLE plan: (1) the exact hybrid estimator + equations (make explicit what the sandbox simulates vs what ML corrects, and WHY the result stays genuinely-simulated); (2) how the lab confound is quarantined (random effects / constrained residual) with an identifiability argument; (3) the per-property strategy for all 9 (physics-dominant vs data-corrected); (4) NEW-dopant handling (sandbox-only extrapolation + UQ, since ML has no data there); (5) the training + leave-one-lab-out / leave-one-dopant-out validation protocol and the lab-usable accuracy thresholds to hit; (6) a prioritized, staged BUILD plan (what to implement first, compute needed, kill criteria); (7) honest statement of which properties can realistically reach lab-usable accuracy and which stay ceiling-limited. Implementable in our codebase (Python; sandbox in src/sandbox/*; can call MACE/HSE/QE and scikit/torch).`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })

return {
  n_ground: ground.filter(Boolean).length,
  n_proposals: valid.length,
  critique_verdicts: critiques.filter(Boolean).map(c => ({ lens: c.lens, verdict: c.verdict })),
  ground_bottom_lines: ground.filter(Boolean).map(g => ({ angle: g.angle, bottom_line: g.bottom_line })),
  synthesis: synth,
}
