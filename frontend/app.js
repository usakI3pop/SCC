/**
 * Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency
 * React JS Frontend Interface
 */

const { useState, useEffect, useMemo, createElement: h } = React;

const API_BASE = "http://127.0.0.1:8000";

// Standard benchmark classification presets
const PRESETS = {
  cifar10: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
  animals: ["golden retriever dog", "gray timber wolf", "tabby domestic cat", "spotted deer in forest", "colorful tropical bird", "green tree frog", "brown galloping horse", "african lion in savanna"],
  vehicles: ["commercial passenger airplane", "modern passenger automobile", "large cargo freight truck", "ocean vessel freight ship", "supersonic military fighter jet", "two wheeled racing motorcycle", "city passenger transport bus", "yellow construction bulldozer"]
};

function App() {
  const [activeTab, setActiveTab] = useState("lab"); // "lab", "batch"
  const [health, setHealth] = useState({ status: "checking" });
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("dog");
  const [customImageBase64, setCustomImageBase64] = useState(null);
  const [candidateLabelsInput, setCandidateLabelsInput] = useState(PRESETS.cifar10.join(", "));
  const [groundTruth, setGroundTruth] = useState("dog");
  
  // Attack Hyperparameters (Paper Reference Default: eps_a = 1/255)
  const [attackEps, setAttackEps] = useState(1.0 / 255.0);
  const [attackSteps, setAttackSteps] = useState(10);
  const [attackRandomInit, setAttackRandomInit] = useState(false);

  // Defense Hyperparameters (Reference Defaults)
  const [defenseEps, setDefenseEps] = useState(4.0 / 255.0);
  const [defenseSteps, setDefenseSteps] = useState(5);
  const [warmupSteps, setWarmupSteps] = useState(5);
  const [numViews, setNumViews] = useState(2);
  const [lambdaCm, setLambdaCm] = useState(4.0);
  const [tempT, setTempT] = useState(0.5);
  const [noiseSigma, setNoiseSigma] = useState(6.0 / 255.0);

  // View mode toggles for perturbation visualization
  const [showAdvPerturbation, setShowAdvPerturbation] = useState(false);
  const [showDefPerturbation, setShowDefPerturbation] = useState(false);

  // State for Pipeline Results
  const [pipelineData, setPipelineData] = useState(null);
  const [loadingAction, setLoadingAction] = useState(null); // "pipeline", "classify", "attack", "defend", "batch"
  const [errorMessage, setErrorMessage] = useState(null);

  // Batch Evaluation State
  const [batchPreset, setBatchPreset] = useState("cifar10");
  const [batchResult, setBatchResult] = useState(null);

  // Check health and samples on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: "offline", model: "CLIP ViT-B/32" }));

    fetch(`${API_BASE}/api/samples`)
      .then(res => res.json())
      .then(data => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
        }
      })
      .catch(() => {});
  }, []);

  // Update classes when sample changes
  const handleSelectSample = (sampleId) => {
    setSelectedSampleId(sampleId);
    setCustomImageBase64(null);
    const s = samples.find(x => x.id === sampleId);
    if (s) {
      setGroundTruth(s.ground_truth);
      setCandidateLabelsInput(s.classes.join(", "));
    }
  };

  const candidateClasses = useMemo(() => {
    return candidateLabelsInput
      .split(",")
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }, [candidateLabelsInput]);

  // Handle custom image file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCustomImageBase64(reader.result);
      setSelectedSampleId(null);
    };
    reader.readAsDataURL(file);
  };

  // Run full pipeline
  const runPipeline = async () => {
    setLoadingAction("pipeline");
    setErrorMessage(null);
    try {
      const payload = {
        sample_id: customImageBase64 ? null : selectedSampleId,
        image_base64: customImageBase64,
        candidate_labels: candidateClasses,
        ground_truth: groundTruth,
        epsilon_a: attackEps,
        steps_a: attackSteps,
        random_init: attackRandomInit,
        epsilon_d: defenseEps,
        steps_d: defenseSteps,
        warmup_steps: warmupSteps,
        num_views: numViews,
        lambda_cm: lambdaCm,
        temperature: tempT,
        noise_sigma: noiseSigma
      };

      const res = await fetch(`${API_BASE}/api/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Pipeline failed");
      }
      const data = await res.json();
      setPipelineData(data);
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setLoadingAction(null);
    }
  };

  // Run batch evaluation
  const runBatchBenchmark = async () => {
    setLoadingAction("batch");
    setErrorMessage(null);
    try {
      const form = new URLSearchParams();
      form.append("dataset_preset", batchPreset);
      form.append("epsilon_a", attackEps.toString());
      form.append("steps_a", attackSteps.toString());
      form.append("epsilon_d", defenseEps.toString());
      form.append("steps_d", defenseSteps.toString());
      form.append("warmup_steps", warmupSteps.toString());
      form.append("num_views", numViews.toString());

      const res = await fetch(`${API_BASE}/api/evaluate/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString()
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Batch evaluation failed");
      }
      const data = await res.json();
      setBatchResult(data);
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setLoadingAction(null);
    }
  };

  // Helper for Top-5 Probability Bars
  const renderProbabilityBars = (top5, groundTruthLabel, isAttacked = false) => {
    return h("div", { className: "space-y-2 mt-3" },
      top5.map((item, idx) => {
        const isGT = item.label.toLowerCase() === groundTruthLabel.toLowerCase();
        const barColor = isGT 
          ? "bg-emerald-500" 
          : (idx === 0 && isAttacked ? "bg-rose-500" : "bg-cyan-500/80");

        return h("div", { key: item.label, className: "text-xs" },
          h("div", { className: "flex justify-between items-center mb-1 text-[11px]" },
            h("div", { className: "flex items-center gap-1.5 truncate max-w-[180px]" },
              h("span", { className: "font-mono text-slate-500 w-3.5" }, `#${item.rank}`),
              h("span", { className: `font-medium truncate ${isGT ? "text-emerald-400 font-semibold" : "text-slate-300"}` },
                item.label
              ),
              isGT && h("span", { className: "text-[9px] px-1 py-0.2 bg-emerald-950 text-emerald-300 border border-emerald-700/60 rounded" }, "GT")
            ),
            h("div", { className: "font-mono text-slate-400 flex items-center gap-2" },
              h("span", { className: "text-slate-500 text-[10px]" }, `z=${item.logit}`),
              h("span", { className: `font-semibold ${isGT ? "text-emerald-300" : "text-slate-200"}` },
                `${(item.probability * 100).toFixed(1)}%`
              )
            )
          ),
          h("div", { className: "h-2 w-full bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/40" },
            h("div", {
              className: `h-full rounded-full transition-all duration-500 ${barColor}`,
              style: { width: `${Math.min(100, item.probability * 100)}%` }
            })
          )
        );
      })
    );
  };

  // Helper for Margin Meter
  const renderMarginMeter = (margin) => {
    const isPositive = margin >= 0;
    return h("div", { className: "flex items-center justify-between p-2 rounded bg-slate-900/90 border border-slate-800 text-xs" },
      h("span", { className: "text-slate-400 font-mono text-[11px]" }, "Semantic Margin m(x):"),
      h("div", { className: "flex items-center gap-1.5 font-mono" },
        h("span", { className: `font-bold text-sm ${isPositive ? "text-emerald-400" : "text-rose-400"}` },
          `${isPositive ? "+" : ""}${margin.toFixed(4)}`
        ),
        h("span", { className: `text-[10px] px-1.5 py-0.5 rounded ${isPositive ? "bg-emerald-950 text-emerald-300 border border-emerald-800/60" : "bg-rose-950 text-rose-300 border border-rose-800/60"}` },
          isPositive ? "Aligned" : "Inverted"
        )
      )
    );
  };

  return h("div", { className: "min-h-screen flex flex-col bg-[#0b1020] text-[#e7edf8]" },
    // -------------------------------------------------------------
    // TOP NAVIGATION & STATUS BAR
    // -------------------------------------------------------------
    h("header", { className: "sticky top-0 z-30 bg-[#0d1424]/90 backdrop-blur-md border-b border-[#222f46] px-6 py-3.5" },
      h("div", { className: "max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4" },
        h("div", { className: "flex items-center gap-3" },
          h("div", { className: "w-9 h-9 rounded-lg bg-emerald-950/80 border border-emerald-600/40 flex items-center justify-center text-emerald-400 font-bold shadow-lg shadow-emerald-950/50" },
            "VLM"
          ),
          h("div", null,
            h("h1", { className: "font-heading font-bold text-base tracking-tight text-white flex items-center gap-2" },
              "Adversarial Robustness using PGD & Self-Calibrated Consistency (SCC)"
            ),
            h("p", { className: "text-xs text-slate-400" },
              "Test-time consistency defense against white-box adversarial perturbations on CLIP ViT-B/32"
            )
          )
        ),
        // System status chip & Tab Switcher (Only Comparison Lab and Batch Benchmark)
        h("div", { className: "flex items-center gap-3" },
          h("div", { className: "flex items-center gap-2 text-xs font-mono bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-md" },
            h("span", { className: `w-2 h-2 rounded-full ${health.status === "ready" ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-amber-400"}` }),
            h("span", { className: "text-slate-300" }, health.model || "CLIP ViT-B/32"),
            h("span", { className: "text-slate-600" }, "|"),
            h("span", { className: "text-slate-400" }, (health.device || "cpu").toUpperCase())
          ),
          h("div", { className: "flex bg-slate-900/80 p-1 rounded-lg border border-slate-800 text-xs font-medium" },
            h("button", {
              onClick: () => setActiveTab("lab"),
              className: `px-4 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${activeTab === "lab" ? "bg-emerald-600 text-white shadow font-semibold" : "text-slate-400 hover:text-white"}`
            }, "🔬 Comparison Lab"),
            h("button", {
              onClick: () => setActiveTab("batch"),
              className: `px-4 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${activeTab === "batch" ? "bg-emerald-600 text-white shadow font-semibold" : "text-slate-400 hover:text-white"}`
            }, "📊 Batch Benchmark")
          )
        )
      )
    ),

    // Error Alert Banner
    errorMessage && h("div", { className: "bg-rose-950/80 border-b border-rose-800 px-6 py-2.5 text-xs text-rose-200 flex items-center justify-between" },
      h("span", null, `⚠️ Error: ${errorMessage}`),
      h("button", { onClick: () => setErrorMessage(null), className: "text-rose-400 hover:text-white font-bold ml-4" }, "✕")
    ),

    // -------------------------------------------------------------
    // MAIN CONTENT
    // -------------------------------------------------------------
    h("main", { className: "flex-1 max-w-7xl w-full mx-auto p-6 space-y-6" },

      // ===========================================================
      // TAB 1: COMPARISON LAB
      // ===========================================================
      activeTab === "lab" && h("div", { className: "space-y-6" },
        // Top Control Grid
        h("div", { className: "grid grid-cols-1 lg:grid-cols-12 gap-6" },
          // Left Column: Image Selection & Classes (5 cols)
          h("div", { className: "lg:col-span-5 bg-[#10182a] border border-[#222f46] rounded-xl p-5 space-y-4 shadow-xl" },
            h("div", { className: "flex items-center justify-between border-b border-slate-800 pb-3" },
              h("h2", { className: "font-heading font-semibold text-sm text-slate-100 flex items-center gap-2" },
                "1. Input Image & Taxonomy"
              ),
              h("span", { className: "text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/40" },
                "CLIP ViT-B/32"
              )
            ),

            // Sample picker buttons
            h("div", null,
              h("label", { className: "block text-xs font-medium text-slate-400 mb-2" }, "Curated Benchmark Samples:"),
              h("div", { className: "grid grid-cols-3 gap-2" },
                (samples.length > 0 ? samples : [
                  { id: "dog", name: "Dog" },
                  { id: "wolf", name: "Wolf" },
                  { id: "airplane", name: "Airplane" },
                  { id: "automobile", name: "Car" },
                  { id: "cat", name: "Cat" },
                  { id: "ship", name: "Ship" }
                ]).map(s => h("button", {
                  key: s.id,
                  onClick: () => handleSelectSample(s.id),
                  className: `px-2.5 py-2 text-xs rounded-lg border font-medium transition-all text-center truncate ${selectedSampleId === s.id && !customImageBase64 ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold" : "bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700"}`
                }, s.name))
              )
            ),

            // Custom Upload
            h("div", null,
              h("label", { className: "block text-xs font-medium text-slate-400 mb-1.5" }, "Or Upload Custom Image:"),
              h("input", {
                type: "file",
                accept: "image/*",
                onChange: handleFileUpload,
                className: "block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer bg-slate-900/50 p-1.5 rounded-lg border border-slate-800"
              })
            ),

            // Candidate Classes
            h("div", null,
              h("div", { className: "flex justify-between items-center mb-1.5" },
                h("label", { className: "text-xs font-medium text-slate-400" }, "Candidate Classes:"),
                h("div", { className: "flex gap-1 text-[10px]" },
                  h("button", {
                    onClick: () => { setCandidateLabelsInput(PRESETS.cifar10.join(", ")); setGroundTruth("dog"); },
                    className: "px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                  }, "CIFAR-10"),
                  h("button", {
                    onClick: () => { setCandidateLabelsInput(PRESETS.animals.join(", ")); setGroundTruth("golden retriever dog"); },
                    className: "px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                  }, "Animals"),
                  h("button", {
                    onClick: () => { setCandidateLabelsInput(PRESETS.vehicles.join(", ")); setGroundTruth("commercial passenger airplane"); },
                    className: "px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                  }, "Vehicles")
                )
              ),
              h("textarea", {
                rows: 2,
                value: candidateLabelsInput,
                onChange: e => setCandidateLabelsInput(e.target.value),
                className: "w-full text-xs bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-slate-200 font-mono focus:border-emerald-500 focus:outline-none"
              })
            ),

            // Ground Truth
            h("div", { className: "flex items-center justify-between" },
              h("label", { className: "text-xs font-medium text-slate-400" }, "Ground Truth Label:"),
              h("select", {
                value: groundTruth,
                onChange: e => setGroundTruth(e.target.value),
                className: "text-xs bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-emerald-400 font-mono font-semibold focus:border-emerald-500 focus:outline-none max-w-[200px]"
              },
                candidateClasses.map(c => h("option", { key: c, value: c }, c))
              )
            )
          ),

          // Right Column: Hyperparameters & Execution Trigger (7 cols)
          h("div", { className: "lg:col-span-7 bg-[#10182a] border border-[#222f46] rounded-xl p-5 space-y-4 shadow-xl flex flex-col justify-between" },
            h("div", { className: "space-y-4" },
              h("div", { className: "flex items-center justify-between border-b border-slate-800 pb-3" },
                h("h2", { className: "font-heading font-semibold text-sm text-slate-100 flex items-center gap-2" },
                  "2. Attack & Defense Hyperparameters"
                ),
                h("span", { className: "text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40" },
                  "Reference Configuration"
                )
              ),

              // Attack & Defense Grid
              h("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4 text-xs" },
                // Attack parameters box
                h("div", { className: "p-3.5 rounded-lg bg-slate-900/80 border border-rose-950/60 space-y-3" },
                  h("div", { className: "flex items-center justify-between text-rose-400 font-semibold border-b border-slate-800 pb-1.5" },
                    h("span", null, "⚔️ PGD Adversarial Attack"),
                    h("span", { className: "font-mono text-[10px]" }, "White-Box L∞")
                  ),
                  h("div", null,
                    h("div", { className: "flex justify-between text-slate-400 mb-1" },
                      h("span", null, "Attack Budget (εₐ):"),
                      h("span", { className: "font-mono font-semibold text-rose-300" }, `${(attackEps * 255).toFixed(1)}/255 (${attackEps.toFixed(4)})`)
                    ),
                    h("input", {
                      type: "range",
                      min: 1 / 255,
                      max: 16 / 255,
                      step: 1 / 255,
                      value: attackEps,
                      onChange: e => setAttackEps(parseFloat(e.target.value)),
                      className: "w-full accent-rose-500 cursor-pointer"
                    }),
                    h("div", { className: "flex justify-between text-[10px] text-slate-500 mt-0.5" },
                      h("span", null, "1/255"),
                      h("span", null, "4/255 (Standard)"),
                      h("span", null, "16/255")
                    )
                  ),
                  h("div", { className: "flex justify-between items-center" },
                    h("span", { className: "text-slate-400" }, "PGD Steps:"),
                    h("div", { className: "flex items-center gap-2" },
                      h("input", {
                        type: "number",
                        min: 1,
                        max: 20,
                        value: attackSteps,
                        onChange: e => setAttackSteps(parseInt(e.target.value) || 1),
                        className: "w-14 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-center font-mono text-slate-200"
                      }),
                      h("span", { className: "text-slate-500 text-[11px]" }, "iters")
                    )
                  ),
                  h("div", { className: "flex items-center justify-between" },
                    h("span", { className: "text-slate-400" }, "Random Init (L∞ Ball):"),
                    h("input", {
                      type: "checkbox",
                      checked: attackRandomInit,
                      onChange: e => setAttackRandomInit(e.target.checked),
                      className: "accent-rose-500 h-4 w-4 rounded"
                    })
                  )
                ),

                // Defense parameters box
                h("div", { className: "p-3.5 rounded-lg bg-slate-900/80 border border-emerald-950/60 space-y-3" },
                  h("div", { className: "flex items-center justify-between text-emerald-400 font-semibold border-b border-slate-800 pb-1.5" },
                    h("span", null, "🛡️ SCC Test-Time Defense"),
                    h("span", { className: "font-mono text-[10px]" }, "Self-Calibrated")
                  ),
                  h("div", null,
                    h("div", { className: "flex justify-between text-slate-400 mb-1" },
                      h("span", null, "Defense Budget (ε_d):"),
                      h("span", { className: "font-mono font-semibold text-emerald-300" }, `${(defenseEps * 255).toFixed(1)}/255 (${defenseEps.toFixed(4)})`)
                    ),
                    h("input", {
                      type: "range",
                      min: 1 / 255,
                      max: 8 / 255,
                      step: 1 / 255,
                      value: defenseEps,
                      onChange: e => setDefenseEps(parseFloat(e.target.value)),
                      className: "w-full accent-emerald-500 cursor-pointer"
                    })
                  ),
                  h("div", { className: "grid grid-cols-2 gap-2 text-[11px]" },
                    h("div", null,
                      h("span", { className: "text-slate-400 block" }, "SCC Steps (S):"),
                      h("input", {
                        type: "number",
                        min: 1,
                        max: 10,
                        value: defenseSteps,
                        onChange: e => setDefenseSteps(parseInt(e.target.value) || 1),
                        className: "w-full mt-1 bg-slate-950 border border-slate-700 rounded px-2 py-1 font-mono text-center"
                      })
                    ),
                    h("div", null,
                      h("span", { className: "text-slate-400 block" }, "Warmup (S_w):"),
                      h("input", {
                        type: "number",
                        min: 0,
                        max: 10,
                        value: warmupSteps,
                        onChange: e => setWarmupSteps(parseInt(e.target.value) || 0),
                        className: "w-full mt-1 bg-slate-950 border border-slate-700 rounded px-2 py-1 font-mono text-center"
                      })
                    )
                  ),
                  h("div", { className: "grid grid-cols-3 gap-2 text-[10px] text-slate-400 border-t border-slate-800 pt-2" },
                    h("div", null,
                      h("span", null, "λ_cm = "),
                      h("span", { className: "text-slate-200 font-mono" }, lambdaCm.toFixed(1))
                    ),
                    h("div", null,
                      h("span", null, "Temp T = "),
                      h("span", { className: "text-slate-200 font-mono" }, tempT.toFixed(1))
                    ),
                    h("div", null,
                      h("span", null, "Views V = "),
                      h("span", { className: "text-slate-200 font-mono" }, numViews)
                    )
                  )
                )
              )
            ),

            // Execution Buttons
            h("div", { className: "pt-2 flex items-center gap-3" },
              h("button", {
                onClick: runPipeline,
                disabled: loadingAction !== null,
                className: "flex-1 py-3 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-lg text-sm shadow-lg shadow-emerald-950/60 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              },
                loadingAction === "pipeline" 
                  ? [h("span", { key: "spin", className: "animate-spin" }, "⚙️"), " Computing Gradients (PGD & SCC)..."]
                  : [h("span", { key: "bolt" }, "⚡"), " Run Full Research Pipeline (Clean → PGD → SCC)"]
              )
            )
          )
        ),

        // -----------------------------------------------------------
        // EXECUTIVE SUMMARY METRICS BANNER
        // -----------------------------------------------------------
        pipelineData && h("div", { className: "bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl space-y-3" },
          h("div", { className: "flex flex-wrap items-center justify-between gap-3 text-xs" },
            h("div", { className: "flex items-center gap-3" },
              h("span", { className: "font-heading font-bold text-slate-300 uppercase tracking-wider text-[11px]" },
                "Comparison Summary:"
              ),
              h("div", { className: "flex items-center gap-1.5" },
                h("span", { className: "text-slate-400" }, "Ground Truth:"),
                h("span", { className: "px-2 py-0.5 bg-slate-800 text-emerald-400 font-mono font-bold rounded" },
                  pipelineData.ground_truth
                )
              )
            ),
            h("div", { className: "flex items-center gap-3 font-mono" },
              h("div", { className: `px-2.5 py-1 rounded-md text-xs font-bold border ${pipelineData.attack_success ? "bg-rose-950/80 border-rose-800 text-rose-300" : "bg-slate-800 border-slate-700 text-slate-300"}` },
                pipelineData.attack_success ? "💥 PGD FOOLED MODEL" : "🛡️ ATTACK FAILED"
              ),
              h("div", { className: `px-2.5 py-1 rounded-md text-xs font-bold border ${pipelineData.defense_recovered ? "bg-emerald-950/80 border-emerald-800 text-emerald-300" : "bg-slate-800 border-slate-700 text-slate-300"}` },
                pipelineData.defense_recovered ? "🌟 SCC RECOVERED PREDICTION" : "⚠️ NOT RECOVERED"
              ),
              h("div", { className: "text-slate-400 text-xs font-mono" },
                `Total Pipeline: ${pipelineData.total_time_ms} ms`
              )
            )
          ),

          // Semantic Margin Trajectory Tracker
          h("div", { className: "grid grid-cols-1 md:grid-cols-4 gap-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 text-xs font-mono" },
            h("div", { className: "border-r border-slate-800 pr-3" },
              h("span", { className: "text-slate-500 text-[10px] block" }, "1. CLEAN MARGIN"),
              h("div", { className: "text-emerald-400 font-bold text-base mt-0.5" },
                `+${pipelineData.clean.semantic_margin.toFixed(4)}`
              ),
              h("span", { className: "text-slate-400 text-[11px]" }, `Pred: ${pipelineData.clean.predicted_label}`)
            ),
            h("div", { className: "border-r border-slate-800 pr-3" },
              h("span", { className: "text-slate-500 text-[10px] block" }, "2. ATTACKED MARGIN"),
              h("div", { className: `font-bold text-base mt-0.5 ${pipelineData.attacked.semantic_margin < 0 ? "text-rose-400" : "text-amber-400"}` },
                `${pipelineData.attacked.semantic_margin >= 0 ? "+" : ""}${pipelineData.attacked.semantic_margin.toFixed(4)}`
              ),
              h("span", { className: "text-slate-400 text-[11px]" }, `Shifted: ${pipelineData.attacked.predicted_label}`)
            ),
            h("div", { className: "border-r border-slate-800 pr-3" },
              h("span", { className: "text-slate-500 text-[10px] block" }, "3. DEFENDED MARGIN"),
              h("div", { className: `font-bold text-base mt-0.5 ${pipelineData.defended.semantic_margin >= 0 ? "text-emerald-400" : "text-amber-400"}` },
                `${pipelineData.defended.semantic_margin >= 0 ? "+" : ""}${pipelineData.defended.semantic_margin.toFixed(4)}`
              ),
              h("span", { className: "text-slate-400 text-[11px]" }, `Restored: ${pipelineData.defended.predicted_label}`)
            ),
            h("div", null,
              h("span", { className: "text-slate-500 text-[10px] block" }, "RECOVERY MARGIN GAIN (Δm)"),
              h("div", { className: `font-bold text-base mt-0.5 ${pipelineData.summary.margin_gain > 0 ? "text-cyan-400" : "text-slate-400"}` },
                `${pipelineData.summary.margin_gain > 0 ? "+" : ""}${pipelineData.summary.margin_gain.toFixed(4)}`
              ),
              h("span", { className: "text-slate-400 text-[11px]" }, "SCC Repulsion Effect")
            )
          )
        ),

        // -----------------------------------------------------------
        // 3-STAGE SIDE-BY-SIDE CARDS
        // -----------------------------------------------------------
        pipelineData && h("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-6" },
          // STAGE 1: CLEAN CLIP BASELINE
          h("div", { className: "bg-[#10182a] border border-[#222f46] rounded-xl overflow-hidden shadow-xl flex flex-col" },
            h("div", { className: "px-4 py-3 bg-slate-900/80 border-b border-slate-800 flex justify-between items-center" },
              h("div", null,
                h("h3", { className: "font-heading font-bold text-sm text-slate-200" }, "Stage 1: Clean Baseline"),
                h("span", { className: "text-[10px] text-slate-400 font-mono" }, "Zero-shot CLIP ViT-B/32")
              ),
              h("span", { className: "text-[11px] font-mono text-slate-400" }, `${pipelineData.clean.execution_time_ms} ms`)
            ),
            h("div", { className: "p-4 space-y-3 flex-1 flex flex-col justify-between" },
              h("div", { className: "space-y-3" },
                // Image container
                h("div", { className: "relative aspect-square w-full rounded-lg overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center" },
                  h("img", {
                    src: pipelineData.clean.image_base64,
                    alt: "Clean Image",
                    className: "w-full h-full object-contain"
                  }),
                  h("div", { className: "absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] font-mono text-slate-300" },
                    "Clean Input x"
                  )
                ),
                // Prediction pill
                h("div", { className: "p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex justify-between items-center text-xs" },
                  h("div", null,
                    h("span", { className: "text-slate-500 text-[10px] block" }, "Predicted Class:"),
                    h("span", { className: "font-heading font-bold text-sm text-white" }, pipelineData.clean.predicted_label)
                  ),
                  h("div", { className: "text-right" },
                    h("span", { className: "text-slate-500 text-[10px] block" }, "Confidence:"),
                    h("span", { className: "font-mono font-bold text-emerald-400" }, `${(pipelineData.clean.confidence * 100).toFixed(1)}%`)
                  )
                ),
                renderMarginMeter(pipelineData.clean.semantic_margin)
              ),
              // Top-5 distribution
              h("div", { className: "border-t border-slate-800 pt-3" },
                h("span", { className: "text-[11px] font-mono text-slate-400 uppercase tracking-wider block" }, "Top-5 Probabilities:"),
                renderProbabilityBars(pipelineData.clean.top5_predictions, pipelineData.ground_truth, false)
              )
            )
          ),

          // STAGE 2: PGD ADVERSARIAL ATTACK
          h("div", { className: "bg-[#10182a] border border-rose-950/80 rounded-xl overflow-hidden shadow-xl flex flex-col" },
            h("div", { className: "px-4 py-3 bg-rose-950/40 border-b border-rose-900/50 flex justify-between items-center" },
              h("div", null,
                h("h3", { className: "font-heading font-bold text-sm text-rose-300 flex items-center gap-1.5" },
                  "Stage 2: PGD Attack",
                  h("span", { className: "text-[10px] px-1.5 py-0.2 bg-rose-900/60 text-rose-200 border border-rose-700/60 rounded" }, "Untargeted")
                ),
                h("span", { className: "text-[10px] text-slate-400 font-mono" }, `εₐ=${pipelineData.attacked.linf_pixel}/255`)
              ),
              h("span", { className: "text-[11px] font-mono text-slate-400" }, `${pipelineData.attacked.execution_time_ms} ms`)
            ),
            h("div", { className: "p-4 space-y-3 flex-1 flex flex-col justify-between" },
              h("div", { className: "space-y-3" },
                // Image container with toggle
                h("div", { className: "relative aspect-square w-full rounded-lg overflow-hidden border border-rose-900/40 bg-slate-950 flex items-center justify-center" },
                  h("img", {
                    src: showAdvPerturbation ? pipelineData.attacked.perturbation_base64 : pipelineData.attacked.image_base64,
                    alt: "Adversarial Image",
                    className: "w-full h-full object-contain"
                  }),
                  h("button", {
                    onClick: () => setShowAdvPerturbation(!showAdvPerturbation),
                    className: "absolute top-2 right-2 px-2 py-1 rounded bg-black/80 hover:bg-black text-[10px] font-mono text-slate-300 border border-slate-700 shadow"
                  }, showAdvPerturbation ? "Show Image" : "Show Perturbation (15x)")
                ),
                // Prediction pill
                h("div", { className: "p-2.5 rounded-lg bg-slate-900 border border-rose-950/60 flex justify-between items-center text-xs" },
                  h("div", null,
                    h("span", { className: "text-slate-500 text-[10px] block" }, "Adversarial Prediction:"),
                    h("span", { className: "font-heading font-bold text-sm text-rose-400" }, pipelineData.attacked.predicted_label)
                  ),
                  h("div", { className: "text-right" },
                    h("span", { className: "text-slate-500 text-[10px] block" }, "L∞ / L2 Norm:"),
                    h("span", { className: "font-mono font-bold text-rose-300 text-[11px]" }, `${pipelineData.attacked.linf_pixel}/255 | ${pipelineData.attacked.l2_norm.toFixed(2)}`)
                  )
                ),
                renderMarginMeter(pipelineData.attacked.semantic_margin)
              ),
              // Top-5 distribution
              h("div", { className: "border-t border-slate-800 pt-3" },
                h("span", { className: "text-[11px] font-mono text-slate-400 uppercase tracking-wider block" }, "Top-5 Probabilities (Attacked):"),
                renderProbabilityBars(pipelineData.attacked.top5_predictions, pipelineData.ground_truth, true)
              )
            )
          ),

          // STAGE 3: SCC TEST-TIME DEFENSE
          h("div", { className: "bg-[#10182a] border border-emerald-950/80 rounded-xl overflow-hidden shadow-xl flex flex-col" },
            h("div", { className: "px-4 py-3 bg-emerald-950/40 border-b border-emerald-900/50 flex justify-between items-center" },
              h("div", null,
                h("h3", { className: "font-heading font-bold text-sm text-emerald-300 flex items-center gap-1.5" },
                  "Stage 3: SCC Defense",
                  h("span", { className: "text-[10px] px-1.5 py-0.2 bg-emerald-900/60 text-emerald-200 border border-emerald-700/60 rounded" }, "Test-Time")
                ),
                h("span", { className: "text-[10px] text-slate-400 font-mono" }, `ε_d=${pipelineData.defended.linf_pixel}/255`)
              ),
              h("span", { className: "text-[11px] font-mono text-slate-400" }, `${pipelineData.defended.execution_time_ms} ms`)
            ),
            h("div", { className: "p-4 space-y-3 flex-1 flex flex-col justify-between" },
              h("div", { className: "space-y-3" },
                // Image container with toggle
                h("div", { className: "relative aspect-square w-full rounded-lg overflow-hidden border border-emerald-900/40 bg-slate-950 flex items-center justify-center" },
                  h("img", {
                    src: showDefPerturbation ? pipelineData.defended.perturbation_base64 : pipelineData.defended.image_base64,
                    alt: "Defended Image",
                    className: "w-full h-full object-contain"
                  }),
                  h("button", {
                    onClick: () => setShowDefPerturbation(!showDefPerturbation),
                    className: "absolute top-2 right-2 px-2 py-1 rounded bg-black/80 hover:bg-black text-[10px] font-mono text-slate-300 border border-slate-700 shadow"
                  }, showDefPerturbation ? "Show Image" : "Show Corrective (15x)")
                ),
                // Prediction pill
                h("div", { className: "p-2.5 rounded-lg bg-slate-900 border border-emerald-950/60 flex justify-between items-center text-xs" },
                  h("div", null,
                    h("span", { className: "text-slate-500 text-[10px] block" }, "Defended Prediction:"),
                    h("span", { className: "font-heading font-bold text-sm text-emerald-300" }, pipelineData.defended.predicted_label)
                  ),
                  h("div", { className: "text-right" },
                    h("span", { className: "text-slate-500 text-[10px] block" }, "Status:"),
                    h("span", { className: `font-mono font-bold text-xs ${pipelineData.defense_recovered ? "text-emerald-400" : "text-amber-400"}` },
                      pipelineData.defense_recovered ? "RECOVERED ✓" : "STABILIZED"
                    )
                  )
                ),
                renderMarginMeter(pipelineData.defended.semantic_margin)
              ),
              // Top-5 distribution
              h("div", { className: "border-t border-slate-800 pt-3" },
                h("span", { className: "text-[11px] font-mono text-slate-400 uppercase tracking-wider block" }, "Top-5 Probabilities (Defended):"),
                renderProbabilityBars(pipelineData.defended.top5_predictions, pipelineData.ground_truth, false)
              )
            )
          )
        )
      ),

      // ===========================================================
      // TAB 2: BATCH EVALUATION BENCHMARK
      // ===========================================================
      activeTab === "batch" && h("div", { className: "space-y-6" },
        h("div", { className: "bg-[#10182a] border border-[#222f46] rounded-xl p-5 shadow-xl space-y-4" },
          h("div", { className: "flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4" },
            h("div", null,
              h("h2", { className: "font-heading font-bold text-base text-white" }, "Zero-Shot Adversarial Robustness Benchmarking"),
              h("p", { className: "text-xs text-slate-400" }, "Evaluate Clean Accuracy, Robust Accuracy (under PGD), Defended Accuracy (under SCC), Attack Success Rate (ASR), and Defense Recovery Rate (DRR).")
            ),
            h("div", { className: "flex items-center gap-3" },
              h("select", {
                value: batchPreset,
                onChange: e => setBatchPreset(e.target.value),
                className: "text-xs bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-mono"
              },
                h("option", { value: "cifar10" }, "CIFAR-10 Subset (Objects)"),
                h("option", { value: "animals" }, "Fine-Grained Animals"),
                h("option", { value: "all" }, "All Benchmark Samples")
              ),
              h("button", {
                onClick: runBatchBenchmark,
                disabled: loadingAction === "batch",
                className: "py-2 px-5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-lg shadow-lg flex items-center gap-2 disabled:opacity-50"
              },
                loadingAction === "batch" ? "⏳ Running Benchmark..." : "🚀 Run Batch Evaluation"
              )
            )
          ),

          // KPI Cards
          batchResult && h("div", { className: "grid grid-cols-2 md:grid-cols-5 gap-4 pt-2" },
            h("div", { className: "bg-slate-900/90 border border-slate-800 p-4 rounded-xl text-center" },
              h("span", { className: "text-xs text-slate-400 block font-mono" }, "Clean Accuracy"),
              h("div", { className: "font-heading font-extrabold text-2xl text-emerald-400 mt-1" }, `${batchResult.clean_accuracy}%`),
              h("span", { className: "text-[10px] text-slate-500" }, "Zero-Shot CLIP Baseline")
            ),
            h("div", { className: "bg-slate-900/90 border border-rose-950/60 p-4 rounded-xl text-center" },
              h("span", { className: "text-xs text-rose-400 block font-mono" }, "Robust Accuracy"),
              h("div", { className: "font-heading font-extrabold text-2xl text-rose-400 mt-1" }, `${batchResult.robust_accuracy}%`),
              h("span", { className: "text-[10px] text-slate-500" }, "Under PGD Attack (No Defense)")
            ),
            h("div", { className: "bg-slate-900/90 border border-cyan-950/60 p-4 rounded-xl text-center" },
              h("span", { className: "text-xs text-cyan-400 block font-mono" }, "Defended Accuracy"),
              h("div", { className: "font-heading font-extrabold text-2xl text-cyan-400 mt-1" }, `${batchResult.defended_accuracy}%`),
              h("span", { className: "text-[10px] text-slate-500" }, "With SCC Test-Time Defense")
            ),
            h("div", { className: "bg-slate-900/90 border border-rose-950/60 p-4 rounded-xl text-center" },
              h("span", { className: "text-xs text-rose-300 block font-mono" }, "Attack Success Rate (ASR)"),
              h("div", { className: "font-heading font-extrabold text-2xl text-rose-300 mt-1" }, `${batchResult.attack_success_rate}%`),
              h("span", { className: "text-[10px] text-slate-500" }, "% Clean Correct Fooled")
            ),
            h("div", { className: "bg-slate-900/90 border border-emerald-950/60 p-4 rounded-xl text-center" },
              h("span", { className: "text-xs text-emerald-300 block font-mono" }, "Defense Recovery (DRR)"),
              h("div", { className: "font-heading font-extrabold text-2xl text-emerald-300 mt-1" }, `${batchResult.defense_recovery_rate}%`),
              h("span", { className: "text-[10px] text-slate-500" }, "% Fooled Recovered by SCC")
            )
          ),

          // Per-sample table
          batchResult && h("div", { className: "border border-slate-800 rounded-lg overflow-x-auto mt-4" },
            h("table", { className: "w-full text-left text-xs font-mono" },
              h("thead", { className: "bg-slate-950 text-slate-400 border-b border-slate-800" },
                h("tr", null,
                  h("th", { className: "p-3" }, "Sample"),
                  h("th", { className: "p-3" }, "Ground Truth"),
                  h("th", { className: "p-3" }, "Clean Pred"),
                  h("th", { className: "p-3" }, "Clean Margin"),
                  h("th", { className: "p-3" }, "Attacked Pred"),
                  h("th", { className: "p-3" }, "Attacked Margin"),
                  h("th", { className: "p-3" }, "Defended Pred"),
                  h("th", { className: "p-3" }, "Defended Margin"),
                  h("th", { className: "p-3" }, "Outcome")
                )
              ),
              h("tbody", { className: "divide-y divide-slate-800/60 bg-slate-900/40" },
                batchResult.samples.map(s => {
                  return h("tr", { key: s.id, className: "hover:bg-slate-800/30" },
                    h("td", { className: "p-3 font-semibold text-slate-200" }, s.name),
                    h("td", { className: "p-3 text-emerald-400 font-bold" }, s.ground_truth),
                    h("td", { className: "p-3 text-slate-300" }, s.clean_pred),
                    h("td", { className: "p-3 text-slate-400" }, `+${s.clean_margin.toFixed(4)}`),
                    h("td", { className: `p-3 font-medium ${s.adv_correct ? "text-slate-300" : "text-rose-400 font-bold"}` }, s.adv_pred),
                    h("td", { className: "p-3 font-mono text-slate-400" }, `${s.adv_margin >= 0 ? "+" : ""}${s.adv_margin.toFixed(4)}`),
                    h("td", { className: `p-3 font-medium ${s.def_correct ? "text-emerald-400 font-bold" : "text-slate-400"}` }, s.def_pred),
                    h("td", { className: "p-3 font-mono text-slate-400" }, `${s.def_margin >= 0 ? "+" : ""}${s.def_margin.toFixed(4)}`),
                    h("td", { className: "p-3" },
                      s.defense_recovered 
                        ? h("span", { className: "px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 text-[10px]" }, "RECOVERED")
                        : (s.clean_correct && !s.adv_correct ? h("span", { className: "px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800 text-[10px]" }, "FOOLED") : h("span", { className: "px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]" }, "PASS"))
                    )
                  );
                })
              )
            )
          )
        )
      )
    ),

    // -------------------------------------------------------------
    // FOOTER
    // -------------------------------------------------------------
    h("footer", { className: "border-t border-[#222f46] px-6 py-4 bg-[#0d1424] text-slate-500 text-xs font-mono flex flex-wrap justify-between items-center gap-3" },
      h("span", null, "Adversarial Robustness Research Platform · CLIP ViT-B/32 · Frozen Weights · Differentiable Gradient Ascent"),
      h("span", { className: "text-slate-400" }, "FastAPI :8000 · React JS & Tailwind CSS :4173")
    )
  );
}

// Render into DOM
const rootElement = document.getElementById("root");
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(h(App));
}
