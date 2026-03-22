# Pitch Q&A — Databricks ML Accelerator

Questions and answers from the perspective of the three people most likely to be in the room.

---

## CTO Perspective

*Focused on: ROI, risk, security, vendor dependency, and whether this replaces a hire.*

---

**Q: We already have a Databricks contract. What does this cost on top of that?**

$500–$1,500 per workspace per month, billed through Databricks Marketplace — same invoice, no new vendor relationship. The conservative ROI scenario: one churn model reduces annual churn by 0.5%. For a $10M ARR company, that's $50K recovered in year one. The annual subscription cost is $6K–$18K. The math is straightforward.

---

**Q: What's the security model? Our data is sensitive.**

Your data never leaves your workspace. The agent reads only Unity Catalog metadata — table names, column names, data types, row counts. It never queries raw data. LLM calls are routed through your own Databricks External Model endpoint, not to an external API. The product runs as a Databricks App inside your workspace — same security boundary as everything else you run there. Your IT and security policies don't change.

---

**Q: What does it actually generate? I've seen "AI code generation" tools produce garbage.**

Three production-ready Databricks notebooks: feature engineering pipeline, training job, and batch inference job. Plus three Databricks Asset Bundle job YAML files ready to deploy with `databricks bundle deploy`. The code uses your actual table names, your actual column types, and follows Databricks governance patterns — correct GRANT statements, UC model registry, MLflow tracking, job-safe parameterization with dbutils.widgets. Every generated artifact is reviewed before anything runs — you see a risk scorecard with specific pass/warn/fail checks before approval.

---

**Q: Does this replace my data engineering team? Will they feel threatened?**

No, and you should be direct about that with your team. This handles the ML setup and scaffolding that nobody on a data engineering team enjoys doing — the boilerplate feature engineering, the MLflow plumbing, the governance grants, the job YAML configuration. Your engineers still own the data pipelines that feed it, the domain knowledge that validates the output, and the business logic that the agent can't know. The pitch to your team is: stop writing the same MLflow boilerplate for the fifth time.

---

**Q: What happens when the generated code is wrong or the model underperforms?**

Every run has a three-checkpoint approval flow. Before code is generated, you approve a dry run plan in plain English. After code is generated, you review it with a risk scorecard that flags the most common failure modes. You — or your engineer — approve before anything deploys. The agent doesn't self-deploy. It generates and explains; humans decide. For model performance, the generated training notebook includes a baseline comparison (mean predictor / last-value) so you're never deploying a model without knowing it's better than trivial.

---

**Q: What if we want to customise the generated code?**

The notebooks are written to `bundles/src/{use_case}/` as plain Python files. Your engineers can edit them like any other code. The Databricks Asset Bundle can be checked into your existing Git repository and deployed through your existing CI/CD pipeline. There's no lock-in — the output is standard Databricks code.

---

**Q: Why is this better than just asking ChatGPT or GitHub Copilot to write the notebooks?**

Three reasons. First, it knows your actual schema — table names, column names, data types, relationships — because it reads your Unity Catalog directly. Generic LLMs write generic code against placeholder table names. Second, it understands Databricks governance — OBO patterns, Unity Catalog model registry, GRANT statements for the inference identity, DAB job configuration. Copilot writes Pandas code; this writes production Databricks code. Third, the risk scorecard catches the problems that generic tools don't even know to look for: temporal splits on time-series data, target leakage from status columns, missing grants for the job service principal.

---

**Q: We're talking to three vendors this quarter. Why should we pilot this one first?**

Because your data is already in Unity Catalog and this is the only tool that starts there. Every other tool will ask you to export data, upload CSVs, configure connectors, or use their proprietary data layer. This starts with a catalog browse dropdown. Your team can have a recommendation in 45 seconds and generated code in 5 minutes from their first login.

---

## Senior ML Engineer Perspective

*Focused on: Technical correctness, governance depth, what they'll have to fix, will it create more work.*

---

**Q: What ML framework does it use? Can I change it?**

XGBoost for tabular classification and regression by default — the right choice for structured data in 95% of enterprise ML use cases. The generated notebooks are plain Python in Databricks notebooks. You can swap the model class, add hyperparameter tuning, change the preprocessing steps. The scaffold handles the boilerplate; you control the modelling decisions.

---

**Q: Does it handle temporal splits correctly? This is my biggest complaint with generated ML code.**

Yes, and specifically so. The risk scorecard includes an explicit check: if the target table has a timestamp or date column, it flags random splits as a warning and describes the correct time-based split. The generated training notebook uses time-based splitting by default when a date column is detected in the feature plan. This is one of the most common reasons models look good in testing and fail silently in production, and it's checked every time.

---

**Q: What about target leakage? How would it detect that?**

The feature planner prompts the LLM to flag columns that are semantically suspicious — status columns (e.g. `is_churned`, `order_status`), columns with names containing "result", "outcome", "final", post-event timestamps — and either excludes them or adds a warning comment in the code. The risk scorecard checks whether the generated code includes any of these column patterns. It's heuristic, not perfect — an ML engineer still needs to review the feature list. But it catches the obvious cases automatically and documents the reasoning.

---

**Q: How does it handle Unity Catalog model registry? Does it use the right MLflow URI?**

Yes. The generated training notebook sets `mlflow.set_registry_uri("databricks-uc")` and uses the three-level namespace `catalog.schema.model_name` for registration. It sets the Champion alias on the best run rather than using the deprecated `Production` stage. These are the current Databricks best practices — the exact patterns that break silently when engineers copy older tutorial notebooks that still use workspace model registry.

---

**Q: Does the inference job actually read from the feature table or does it re-compute features?**

It reads from the feature table written by the feature pipeline. Training-serving skew — where the feature transformation in the training notebook differs from inference — is the number one cause of silent model degradation in production. The generated architecture separates feature computation from training from inference, with the feature table as the shared artifact. The inference notebook reads the same feature table, not raw data, so the transformation is identical.

---

**Q: What about GRANT statements? The inference job service principal always breaks in production.**

This is explicitly handled. The risk scorecard has a check specifically for inference identity grants: it verifies that the generated code includes `GRANT SELECT ON TABLE` for the feature table to the service principal or user identity that the inference job will run as. If it's missing, it's flagged as a fail — not a warning. Missing grants on the inference identity is one of the most common reasons jobs pass in development and fail at deployment, and the scorecard blocks approval until it's addressed.

---

**Q: How does it handle class imbalance? Did you just ignore it?**

The risk scorecard checks the estimated positive class rate. If the rate is below 10%, it flags a warning with a specific recommendation — SMOTE, class weights in XGBoost, or adjusted decision thresholds — and adds a comment to the training notebook explaining the issue. It doesn't silently train on an imbalanced dataset and report 98% accuracy as if that means something.

---

**Q: Will the generated notebooks actually run on my cluster without modifications?**

The generated code uses `dbutils.widgets` for all parameters — job-safe, works both interactively and as a scheduled job. It targets the cluster ID configured in the workspace context, uses the correct UC catalog and schema names from the approved opportunity, and sets library requirements consistent with DBR 14+. In practice you should always run the feature pipeline notebook interactively first to verify the data shapes before deploying as a job — same as you'd do with any notebook.

---

## Product Director Perspective

*Focused on: User experience, adoption, what problem it solves for the team, how to measure success.*

---

**Q: Who actually uses this day-to-day — the CTO, the engineer, or someone else?**

The primary user is the senior data engineer or analytics engineer who owns the Databricks environment. They run the discovery, review the opportunities, confirm the dry run plan, and approve the generated code. The CTO's interaction is with the executive summary that's generated after each bundle is written — a single-page plain-English document with the business case, what was built, estimated impact, and the approval trail. The tool is designed so that the engineer drives and the CTO reviews, not the other way around.

---

**Q: What does the user experience actually look like? Walk me through it.**

Five steps in a Streamlit UI inside Databricks:
1. Connect and browse — dropdown selects catalog and schema, no free-text typos
2. Discover — one button, 45–60 seconds, returns top 3 opportunities each with a financial impact estimate and confidence level
3. Dry run confirmation — a plain-English panel showing exactly what will happen before any code runs; estimated cost and runtime in the header
4. Code review — generated notebooks with a risk scorecard showing every automated check as pass/warn/fail before approval
5. Bundle written — tabbed view with the executive summary, generated file list, and deploy commands

The whole flow from first login to approved bundle is under 10 minutes for a clear-cut use case.

---

**Q: How do we measure whether this is working? What does success look like?**

Primary: time from data-available to model-in-production. Benchmark it on your first use case — count the days. Secondary: number of ML use cases deployed per quarter. Tertiary: engineer time spent on ML boilerplate versus domain-specific work. For a team that previously shipped zero ML models per year due to bottleneck, shipping two or three with the same headcount is the proof point.

---

**Q: What if the team tries it once, the generated code needs significant edits, and they lose confidence in it?**

The risk scorecard and dry run design are specifically to prevent this. If the scorecard returns warnings, the team knows exactly what to fix before they touch the code. If it returns a fail, the approval is blocked — the tool tells you it's not ready. The goal is that engineers don't discover problems after approval; they discover them at the scorecard stage with an actionable fix. The dry run also means the team sees the full plan before investing time reviewing notebooks — if the plan looks wrong, they stop there and nothing has been wasted.

---

**Q: What's the adoption risk? What if engineers don't trust AI-generated code and just rewrite it?**

That's the right question. The product is designed for engineers who are already productive — it's not trying to replace their judgment. The risk scorecard positions the tool as a peer reviewer, not an authority. The generated code is readable, well-commented, and follows patterns the engineers already know. In practice, the adoption path is: engineer uses it on a low-stakes internal use case first, sees that the generated code is reasonable and the risk scorecard catches real issues, builds confidence, then uses it on higher-stakes use cases. The dry run panel also reduces anxiety — you see the full plan in English before committing to anything.

---

**Q: Is there a scenario where this creates more work than it saves?**

Yes, for the wrong customer. If the data in Unity Catalog is poorly structured — no meaningful column names, no timestamp columns, mixed data types, no business context in table or column names — the quality of the recommendations and generated code drops significantly. The agent works well when the data estate is reasonably well-governed. If you're in the middle of a catalog cleanup project, finish that first. The agent surfaces what's there; it can't invent structure that doesn't exist.

---

**Q: What's on the roadmap that would make this more valuable six months from now?**

Three things in priority order. First, deployment execution — the agent actually runs `databricks bundle deploy` rather than generating the commands for you to run. Second, drift detection — a nightly job that checks model performance against a baseline and alerts when drift is detected, with a guided retraining flow. Third, inline code editing in the UI — the ability to modify a specific notebook in the review panel and regenerate only that artifact rather than the full set. All three are in the build plan. The current phase is complete and tested against a real production catalog.

---

*Reference [pitch.md](pitch.md) for the full product pitch. Reference [CLAUDE.md](../CLAUDE.md) for all architectural decisions.*
