# 📊 A/B Test Analyzer

A Streamlit web app for running statistically rigorous A/B test analysis — built for data analysts who are tired of redoing the same significance math in Excel or Python notebooks every time a new experiment finishes.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Conversion rate testing** — two-proportion z-test for binary outcomes (clicked / converted / signed up)
- **Continuous metric testing** — Welch's t-test for numeric metrics (revenue, session time, order value), which doesn't assume equal variance between groups
- **Bayesian mode** — Beta-Binomial conjugate model for conversion tests and Bayesian bootstrap for continuous metrics, giving direct probability statements ("87% chance the variant is better") instead of p-values
- **Sample size calculator** — plan your experiment duration before you launch it
- Manual data entry **or** CSV upload
- Confidence/credible intervals, p-values or posterior probabilities, effect sizes (Cohen's d), and statistical power — not just a yes/no
- Plain-language verdict and recommendation for each test
- Interactive charts (Plotly), including posterior distribution plots in Bayesian mode

## Demo

```
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ab-test-analyzer.git
cd ab-test-analyzer

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

This repo is pre-configured for one-click deployment:

1. Push this repo to GitHub (see Quick Start above, or the git-included zip you already have).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `master`, and set the main file path to `app.py`.
4. Click **Deploy**.

That's it — `requirements.txt`, `runtime.txt` (pins Python 3.11), and `.streamlit/config.toml` (theme + server settings) are already in place, so no extra configuration is needed. The app will be live at a `*.streamlit.app` URL within a minute or two.

**Updating the deployed app:** any push to the connected branch redeploys automatically.

## Landing Page

`docs/index.html` is a standalone, no-build-step landing page that explains the project, compares
Frequentist vs Bayesian mode, and walks through setup — useful as a repo homepage or for sharing
with non-technical stakeholders. Open it directly in a browser, or enable **GitHub Pages → Deploy
from branch → `/docs`** in your repo settings to publish it at `https://YOUR_USERNAME.github.io/ab-test-analyzer/`.

Before publishing, update the `REPO_URL` constant near the bottom of `docs/index.html` (search for
`YOUR_USERNAME`) so the "View on GitHub" button points at your actual repo.

## Project Structure

```
ab-test-analyzer/
├── app.py                       # Streamlit UI (Frequentist + Bayesian modes)
├── ab_stats.py                  # Frequentist statistical engine
├── bayesian_stats.py            # Bayesian statistical engine
├── requirements.txt
├── runtime.txt                  # Python version pin for Streamlit Cloud
├── .streamlit/
│   └── config.toml              # Theme + server config for deployment
├── docs/
│   └── index.html               # Standalone landing/explainer page (GitHub Pages-ready)
├── README.md
├── LICENSE
├── .gitignore
├── sample_data/
│   ├── conversion_sample.csv    # Example data for the conversion-rate test
│   └── continuous_sample.csv    # Example data for the continuous-metric test
└── tests/
    ├── test_ab_stats.py         # Unit tests for the frequentist engine
    └── test_bayesian_stats.py   # Unit tests for the Bayesian engine
```

## CSV Format

**Conversion rate test** — one row per user/session:

| group   | converted |
|---------|-----------|
| control | 0         |
| control | 1         |
| variant | 1         |

**Continuous metric test** — one row per user/session:

| group   | value |
|---------|-------|
| control | 24.50 |
| variant | 27.10 |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## The Stats, Briefly

**Frequentist mode:**
- **Conversion tests** use a two-proportion z-test with a pooled standard error under the null hypothesis, plus an unpooled-SE confidence interval on the difference in rates — the standard approach for this kind of test.
- **Continuous tests** use Welch's t-test rather than Student's t-test, since it doesn't assume the two groups have equal variance — a safer default for real experiment data where variant traffic often behaves differently than control.
- **Power** is reported post-hoc for conversion tests, so a "no significant difference" result can be interpreted correctly (truly no effect, vs. just not enough data yet).
- **Cohen's d** is reported for continuous tests as a sample-size-independent effect size, since statistical significance alone can be misleading with large samples.

**Bayesian mode:**
- **Conversion tests** use the Beta-Binomial conjugate model — each group's conversion rate gets a Beta posterior distribution, updated directly from observed conversions. This has a closed form, so results are exact and instant (no MCMC needed).
- **Continuous tests** use the Bayesian bootstrap (Rubin, 1981) — a resampling technique that estimates the posterior over each group's mean without assuming a Normal distribution.
- Results are reported as **P(variant beats control)** — a direct probability statement that's often easier to communicate to stakeholders than a p-value, since it answers the actual business question ("how likely is B better than A?") instead of a counterfactual one ("how surprising would this data be if there were no difference?").
- A credible interval on the relative uplift is also reported, which (unlike a frequentist confidence interval) can be interpreted directly as "there's a 95% probability the true uplift falls in this range."

This tool is meant to support decision-making, not replace statistical judgment — always sanity-check results against business context (seasonality, novelty effects, sample ratio mismatch, etc.) before shipping a change.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. This was built as a practical portfolio/utility project — feel free to fork and adapt it for your own stack (e.g. swap Streamlit for Flask/Dash, or add Bayesian A/B testing as an alternative mode).
