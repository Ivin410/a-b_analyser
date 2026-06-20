"""
app.py
A/B Test Analyzer — Streamlit web app.

Run locally:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ab_stats import analyze_conversion, analyze_continuous, required_sample_size
from bayesian_stats import bayesian_conversion_analysis, bayesian_continuous_analysis

st.set_page_config(page_title="A/B Test Analyzer", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — mode & global settings
# ---------------------------------------------------------------------------
st.sidebar.title("📊 A/B Test Analyzer")
st.sidebar.caption("Built for data analysts who are tired of redoing this math by hand.")

approach = st.sidebar.radio(
    "Approach",
    ["Frequentist (p-values)", "Bayesian (probabilities)"],
    help=(
        "Frequentist gives you a p-value: 'how surprising is this data if there's no real effect?' "
        "Bayesian gives you a direct probability: 'how likely is it that the variant is actually better?' "
        "Both are valid — Bayesian results are often more intuitive to communicate to stakeholders."
    ),
)

if approach == "Frequentist (p-values)":
    mode = st.sidebar.radio(
        "What are you testing?",
        ["Conversion Rate (binary)", "Continuous Metric (revenue, time, etc.)", "Sample Size Calculator"],
    )
    alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01)
    st.sidebar.caption(f"Confidence level: {(1 - alpha) * 100:.0f}%")
else:
    mode = st.sidebar.radio(
        "What are you testing?",
        ["Conversion Rate (binary)", "Continuous Metric (revenue, time, etc.)"],
    )
    alpha = 0.05  # used only for CI width in frequentist mode; harmless default here
    credible_level = st.sidebar.slider("Credible interval width", 0.80, 0.99, 0.95, 0.01)
    st.sidebar.caption("Wider intervals = more conservative uncertainty range.")

st.sidebar.divider()
st.sidebar.markdown(
    "**How to use**\n\n"
    "1. Pick a test type above\n"
    "2. Enter your data manually or upload a CSV\n"
    "3. Read the verdict — ship it or wait\n"
)

st.title("A/B Test Analyzer")
st.caption("Statistical significance testing for product, marketing, and growth experiments.")


# ---------------------------------------------------------------------------
# Helper: results chart
# ---------------------------------------------------------------------------
def plot_conversion_bars(result):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Variant"],
        y=[result.control_rate * 100, result.variant_rate * 100],
        text=[f"{result.control_rate:.2%}", f"{result.variant_rate:.2%}"],
        textposition="outside",
        marker_color=["#94a3b8", "#6366f1"],
    ))
    fig.update_layout(
        yaxis_title="Conversion Rate (%)",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=20),
    )
    return fig


def plot_continuous_bars(result):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Variant"],
        y=[result.control_mean, result.variant_mean],
        error_y=dict(
            type="data",
            array=[result.control_std, result.variant_std],
            visible=True,
        ),
        text=[f"{result.control_mean:.2f}", f"{result.variant_mean:.2f}"],
        textposition="outside",
        marker_color=["#94a3b8", "#6366f1"],
    ))
    fig.update_layout(
        yaxis_title="Mean value",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=20),
    )
    return fig


def metric_row(labels_values):
    cols = st.columns(len(labels_values))
    for col, (label, value) in zip(cols, labels_values):
        col.metric(label, value)


def plot_posterior_distributions(control_samples, variant_samples, x_label, as_percent=False):
    """Overlaid histograms of the posterior samples for control vs variant."""
    fig = go.Figure()
    c_vals = control_samples * 100 if as_percent else control_samples
    v_vals = variant_samples * 100 if as_percent else variant_samples

    fig.add_trace(go.Histogram(
        x=c_vals, name="Control", opacity=0.6, marker_color="#94a3b8",
        histnorm="probability density", nbinsx=60,
    ))
    fig.add_trace(go.Histogram(
        x=v_vals, name="Variant", opacity=0.6, marker_color="#6366f1",
        histnorm="probability density", nbinsx=60,
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title=x_label,
        yaxis_title="Posterior density",
        height=380,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_uplift_distribution(control_samples, variant_samples):
    """Distribution of (variant - control) relative uplift %, with a zero reference line."""
    uplift = (variant_samples - control_samples) / np.abs(control_samples) * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=uplift, marker_color="#22c55e", opacity=0.75,
        histnorm="probability density", nbinsx=70, name="Relative uplift",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ef4444",
                   annotation_text="No difference", annotation_position="top")
    fig.update_layout(
        xaxis_title="Relative uplift (%)",
        yaxis_title="Posterior density",
        height=320,
        margin=dict(t=20, b=20),
        showlegend=False,
    )
    return fig


if approach == "Frequentist (p-values)":
    # ---------------------------------------------------------------------------
    # MODE 1: Conversion rate test
    # ---------------------------------------------------------------------------
    if mode == "Conversion Rate (binary)":
        st.subheader("Conversion Rate Test")
        st.caption("Use this when your metric is binary — clicked / didn't, purchased / didn't, signed up / didn't.")

        input_method = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)

        control_n = control_conv = variant_n = variant_conv = None

        if input_method == "Manual entry":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Control group**")
                control_n = st.number_input("Visitors / users (control)", min_value=1, value=1000, key="cn")
                control_conv = st.number_input("Conversions (control)", min_value=0, value=120, key="cc")
            with c2:
                st.markdown("**Variant group**")
                variant_n = st.number_input("Visitors / users (variant)", min_value=1, value=1000, key="vn")
                variant_conv = st.number_input("Conversions (variant)", min_value=0, value=145, key="vc")

        else:
            st.markdown(
                "Upload a CSV with columns: **group** (control/variant), **converted** (0 or 1). "
                "[See `sample_data/conversion_sample.csv` for the expected format.]"
            )
            file = st.file_uploader("Upload CSV", type=["csv"], key="conv_csv")
            if file:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if not {"group", "converted"}.issubset(df.columns):
                    st.error("CSV must contain 'group' and 'converted' columns.")
                else:
                    grp = df.groupby(df["group"].str.lower())["converted"].agg(["count", "sum"])
                    try:
                        control_n, control_conv = int(grp.loc["control", "count"]), int(grp.loc["control", "sum"])
                        variant_n, variant_conv = int(grp.loc["variant", "count"]), int(grp.loc["variant", "sum"])
                        st.success(f"Loaded {len(df)} rows.")
                        st.dataframe(grp.rename(columns={"count": "n", "sum": "conversions"}))
                    except KeyError:
                        st.error("Group column must contain values 'control' and 'variant' (case-insensitive).")

        if st.button("Run Analysis", type="primary", key="run_conv"):
            if None in (control_n, control_conv, variant_n, variant_conv):
                st.warning("Please provide valid data for both groups first.")
            else:
                try:
                    result = analyze_conversion(control_n, control_conv, variant_n, variant_conv, alpha)

                    st.divider()
                    metric_row([
                        ("Control rate", f"{result.control_rate:.2%}"),
                        ("Variant rate", f"{result.variant_rate:.2%}"),
                        ("Relative uplift", f"{result.relative_uplift_pct:+.2f}%"),
                        ("p-value", f"{result.p_value:.4f}"),
                    ])

                    left, right = st.columns([1, 1])
                    with left:
                        st.plotly_chart(plot_conversion_bars(result), use_container_width=True)
                    with right:
                        st.markdown("#### Verdict")
                        if result.significant:
                            st.success(result.recommendation)
                        else:
                            st.info(result.recommendation)

                        st.markdown(
                            f"""
**Details**
- Z-score: `{result.z_score:.3f}`
- {(1 - alpha) * 100:.0f}% CI on absolute difference: `[{result.ci_low:+.2%}, {result.ci_high:+.2%}]`
- Statistical power: `{result.power:.1%}`
- Significant at α={alpha}: **{"Yes" if result.significant else "No"}**
                            """
                        )

                    with st.expander("What does this mean?"):
                        st.write(
                            "We ran a two-proportion z-test comparing conversion rates between control and variant. "
                            "The p-value tells you the probability of seeing a difference this large (or larger) if "
                            "there were truly no difference between the groups. If p < α, we call the result "
                            "statistically significant. The confidence interval shows the plausible range for the "
                            "true difference in conversion rate. Power tells you how likely this test was to detect "
                            "a real effect, if one existed — low power means a 'no difference' result is less trustworthy."
                        )
                except ValueError as e:
                    st.error(str(e))

    # ---------------------------------------------------------------------------
    # MODE 2: Continuous metric test
    # ---------------------------------------------------------------------------
    elif mode == "Continuous Metric (revenue, time, etc.)":
        st.subheader("Continuous Metric Test")
        st.caption("Use this when your metric is a number per user — revenue, session length, order value, etc.")

        input_method = st.radio("Input method", ["Paste values", "Upload CSV"], horizontal=True)

        control_values = variant_values = None

        if input_method == "Paste values":
            c1, c2 = st.columns(2)
            with c1:
                control_text = st.text_area(
                    "Control values (comma or newline separated)",
                    value="12.5, 14.0, 9.8, 15.2, 11.0, 13.4, 10.1, 16.0, 12.8, 14.9",
                    height=160,
                )
            with c2:
                variant_text = st.text_area(
                    "Variant values (comma or newline separated)",
                    value="15.0, 16.2, 13.5, 17.8, 14.0, 15.9, 13.0, 18.1, 14.7, 16.5",
                    height=160,
                )
            try:
                control_values = np.array(
                    [float(x) for x in control_text.replace("\n", ",").split(",") if x.strip()]
                )
                variant_values = np.array(
                    [float(x) for x in variant_text.replace("\n", ",").split(",") if x.strip()]
                )
            except ValueError:
                st.error("Could not parse one or more values — make sure they're all numbers.")

        else:
            st.markdown(
                "Upload a CSV with columns: **group** (control/variant), **value** (numeric metric). "
                "[See `sample_data/continuous_sample.csv` for the expected format.]"
            )
            file = st.file_uploader("Upload CSV", type=["csv"], key="cont_csv")
            if file:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if not {"group", "value"}.issubset(df.columns):
                    st.error("CSV must contain 'group' and 'value' columns.")
                else:
                    df["group"] = df["group"].str.lower()
                    control_values = df.loc[df["group"] == "control", "value"].to_numpy()
                    variant_values = df.loc[df["group"] == "variant", "value"].to_numpy()
                    st.success(f"Loaded {len(df)} rows ({len(control_values)} control, {len(variant_values)} variant).")

        if st.button("Run Analysis", type="primary", key="run_cont"):
            if control_values is None or variant_values is None or len(control_values) < 2 or len(variant_values) < 2:
                st.warning("Please provide at least 2 valid values per group.")
            else:
                try:
                    result = analyze_continuous(control_values, variant_values, alpha)

                    st.divider()
                    metric_row([
                        ("Control mean", f"{result.control_mean:.3f}"),
                        ("Variant mean", f"{result.variant_mean:.3f}"),
                        ("Relative uplift", f"{result.relative_uplift_pct:+.2f}%"),
                        ("p-value", f"{result.p_value:.4f}"),
                    ])

                    left, right = st.columns([1, 1])
                    with left:
                        st.plotly_chart(plot_continuous_bars(result), use_container_width=True)
                    with right:
                        st.markdown("#### Verdict")
                        if result.significant:
                            st.success(result.recommendation)
                        else:
                            st.info(result.recommendation)

                        effect_label = (
                            "negligible" if abs(result.cohens_d) < 0.2 else
                            "small" if abs(result.cohens_d) < 0.5 else
                            "medium" if abs(result.cohens_d) < 0.8 else
                            "large"
                        )

                        st.markdown(
                            f"""
**Details**
- t-score: `{result.t_score:.3f}`
- {(1 - alpha) * 100:.0f}% CI on mean difference: `[{result.ci_low:+.3f}, {result.ci_high:+.3f}]`
- Effect size (Cohen's d): `{result.cohens_d:.3f}` ({effect_label})
- Significant at α={alpha}: **{"Yes" if result.significant else "No"}**
                            """
                        )

                    with st.expander("What does this mean?"):
                        st.write(
                            "We ran Welch's t-test, which compares means between two groups without assuming "
                            "equal variances — a safer default than the standard t-test for real-world experiment data. "
                            "Cohen's d measures how large the difference is in standardized terms, independent of "
                            "sample size, which is useful because a tiny effect can still be 'significant' with enough data."
                        )
                except ValueError as e:
                    st.error(str(e))

    # ---------------------------------------------------------------------------
    # MODE 3: Sample size calculator
    # ---------------------------------------------------------------------------
    else:
        st.subheader("Sample Size Calculator")
        st.caption("Plan your experiment before you run it — how many users do you need per group?")

        c1, c2, c3 = st.columns(3)
        with c1:
            baseline = st.number_input("Baseline conversion rate (%)", min_value=0.1, max_value=99.0, value=10.0) / 100
        with c2:
            mde = st.number_input("Minimum detectable effect (relative %)", min_value=1.0, max_value=200.0, value=10.0)
        with c3:
            power = st.slider("Desired power", 0.5, 0.99, 0.80, 0.01)

        if st.button("Calculate", type="primary"):
            n = required_sample_size(baseline, mde, alpha, power)
            st.success(f"You need approximately **{n:,} users per group** ({n*2:,} total) to detect a "
                       f"{mde:.0f}% relative lift from a {baseline:.1%} baseline, at α={alpha} and {power:.0%} power.")
            st.caption(
                "Rule of thumb: smaller effects, tighter significance levels, and higher power all require "
                "more data. Halving the MDE roughly quadruples the required sample size."
            )

else:
    # -----------------------------------------------------------------------
    # BAYESIAN MODE
    # -----------------------------------------------------------------------
    if mode == "Conversion Rate (binary)":
        st.subheader("Bayesian Conversion Rate Test")
        st.caption(
            "Use this when your metric is binary — clicked / didn't, purchased / didn't, signed up / didn't. "
            "Results are expressed as direct probabilities rather than p-values."
        )

        input_method = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True, key="bayes_conv_input")

        control_n = control_conv = variant_n = variant_conv = None

        if input_method == "Manual entry":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Control group**")
                control_n = st.number_input("Visitors / users (control)", min_value=1, value=1000, key="bcn")
                control_conv = st.number_input("Conversions (control)", min_value=0, value=120, key="bcc")
            with c2:
                st.markdown("**Variant group**")
                variant_n = st.number_input("Visitors / users (variant)", min_value=1, value=1000, key="bvn")
                variant_conv = st.number_input("Conversions (variant)", min_value=0, value=145, key="bvc")
        else:
            st.markdown(
                "Upload a CSV with columns: **group** (control/variant), **converted** (0 or 1)."
            )
            file = st.file_uploader("Upload CSV", type=["csv"], key="bayes_conv_csv")
            if file:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if not {"group", "converted"}.issubset(df.columns):
                    st.error("CSV must contain 'group' and 'converted' columns.")
                else:
                    grp = df.groupby(df["group"].str.lower())["converted"].agg(["count", "sum"])
                    try:
                        control_n, control_conv = int(grp.loc["control", "count"]), int(grp.loc["control", "sum"])
                        variant_n, variant_conv = int(grp.loc["variant", "count"]), int(grp.loc["variant", "sum"])
                        st.success(f"Loaded {len(df)} rows.")
                    except KeyError:
                        st.error("Group column must contain values 'control' and 'variant'.")

        with st.expander("Prior settings (optional)"):
            st.caption(
                "Default is a flat/uninformative prior (Beta(1,1)) — it lets the data speak for itself. "
                "Only change this if you have genuine prior knowledge about expected conversion rates."
            )
            pc1, pc2 = st.columns(2)
            with pc1:
                prior_alpha = st.number_input("Prior alpha", min_value=0.1, value=1.0, step=0.5)
            with pc2:
                prior_beta = st.number_input("Prior beta", min_value=0.1, value=1.0, step=0.5)

        if st.button("Run Bayesian Analysis", type="primary", key="run_bayes_conv"):
            if None in (control_n, control_conv, variant_n, variant_conv):
                st.warning("Please provide valid data for both groups first.")
            else:
                try:
                    result = bayesian_conversion_analysis(
                        control_n, control_conv, variant_n, variant_conv,
                        prior_alpha=prior_alpha, prior_beta=prior_beta,
                        credible_level=credible_level,
                    )

                    st.divider()
                    metric_row([
                        ("Control rate", f"{result.control_rate:.2%}"),
                        ("Variant rate", f"{result.variant_rate:.2%}"),
                        ("P(variant > control)", f"{result.prob_variant_better:.1%}"),
                        ("Expected uplift", f"{result.expected_uplift_pct:+.1f}%"),
                    ])

                    left, right = st.columns([1, 1])
                    with left:
                        st.plotly_chart(
                            plot_posterior_distributions(
                                result.control_samples, result.variant_samples,
                                "Conversion rate (%)", as_percent=True,
                            ),
                            use_container_width=True,
                        )
                    with right:
                        st.markdown("#### Verdict")
                        if result.prob_variant_better >= 0.95 or result.prob_variant_better <= 0.05:
                            st.success(result.recommendation)
                        else:
                            st.info(result.recommendation)

                        st.markdown(
                            f"""
**Details**
- P(variant beats control): `{result.prob_variant_better:.1%}`
- Expected relative uplift: `{result.expected_uplift_pct:+.2f}%`
- {credible_level:.0%} credible interval on uplift: `[{result.credible_interval_uplift[0]:+.1f}%, {result.credible_interval_uplift[1]:+.1f}%]`
                            """
                        )

                    st.plotly_chart(plot_uplift_distribution(result.control_samples, result.variant_samples), use_container_width=True)

                    with st.expander("What does this mean?"):
                        st.write(
                            "We modeled each group's conversion rate as a Beta distribution (the conjugate "
                            "prior for binomial data) and updated it with your observed conversions. The "
                            "result is a full posterior distribution for each rate, shown above. P(variant > "
                            "control) is the direct probability — across all plausible values — that the "
                            "variant truly converts better than control. This is often easier to act on than "
                            "a p-value, since it answers the actual business question directly."
                        )
                except ValueError as e:
                    st.error(str(e))

    else:
        st.subheader("Bayesian Continuous Metric Test")
        st.caption(
            "Use this when your metric is a number per user — revenue, session length, order value, etc. "
            "Uses the Bayesian bootstrap, so it doesn't assume a Normal distribution."
        )

        input_method = st.radio("Input method", ["Paste values", "Upload CSV"], horizontal=True, key="bayes_cont_input")

        control_values = variant_values = None

        if input_method == "Paste values":
            c1, c2 = st.columns(2)
            with c1:
                control_text = st.text_area(
                    "Control values (comma or newline separated)",
                    value="12.5, 14.0, 9.8, 15.2, 11.0, 13.4, 10.1, 16.0, 12.8, 14.9",
                    height=160, key="bayes_control_text",
                )
            with c2:
                variant_text = st.text_area(
                    "Variant values (comma or newline separated)",
                    value="15.0, 16.2, 13.5, 17.8, 14.0, 15.9, 13.0, 18.1, 14.7, 16.5",
                    height=160, key="bayes_variant_text",
                )
            try:
                control_values = np.array(
                    [float(x) for x in control_text.replace("\n", ",").split(",") if x.strip()]
                )
                variant_values = np.array(
                    [float(x) for x in variant_text.replace("\n", ",").split(",") if x.strip()]
                )
            except ValueError:
                st.error("Could not parse one or more values — make sure they're all numbers.")
        else:
            st.markdown("Upload a CSV with columns: **group** (control/variant), **value** (numeric metric).")
            file = st.file_uploader("Upload CSV", type=["csv"], key="bayes_cont_csv")
            if file:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]
                if not {"group", "value"}.issubset(df.columns):
                    st.error("CSV must contain 'group' and 'value' columns.")
                else:
                    df["group"] = df["group"].str.lower()
                    control_values = df.loc[df["group"] == "control", "value"].to_numpy()
                    variant_values = df.loc[df["group"] == "variant", "value"].to_numpy()
                    st.success(f"Loaded {len(df)} rows.")

        if st.button("Run Bayesian Analysis", type="primary", key="run_bayes_cont"):
            if control_values is None or variant_values is None or len(control_values) < 2 or len(variant_values) < 2:
                st.warning("Please provide at least 2 valid values per group.")
            else:
                try:
                    result = bayesian_continuous_analysis(
                        control_values, variant_values, credible_level=credible_level,
                    )

                    st.divider()
                    metric_row([
                        ("Control mean", f"{result.control_mean:.3f}"),
                        ("Variant mean", f"{result.variant_mean:.3f}"),
                        ("P(variant > control)", f"{result.prob_variant_better:.1%}"),
                        ("Expected uplift", f"{result.expected_uplift_pct:+.1f}%"),
                    ])

                    left, right = st.columns([1, 1])
                    with left:
                        st.plotly_chart(
                            plot_posterior_distributions(result.control_samples, result.variant_samples, "Mean value"),
                            use_container_width=True,
                        )
                    with right:
                        st.markdown("#### Verdict")
                        if result.prob_variant_better >= 0.95 or result.prob_variant_better <= 0.05:
                            st.success(result.recommendation)
                        else:
                            st.info(result.recommendation)

                        st.markdown(
                            f"""
**Details**
- P(variant beats control): `{result.prob_variant_better:.1%}`
- Expected relative uplift: `{result.expected_uplift_pct:+.2f}%`
- {credible_level:.0%} credible interval on uplift: `[{result.credible_interval_uplift[0]:+.1f}%, {result.credible_interval_uplift[1]:+.1f}%]`
                            """
                        )

                    st.plotly_chart(plot_uplift_distribution(result.control_samples, result.variant_samples), use_container_width=True)

                    with st.expander("What does this mean?"):
                        st.write(
                            "We used the Bayesian bootstrap to estimate the posterior distribution of each "
                            "group's mean, without assuming the underlying data is Normally distributed. "
                            "P(variant > control) is the direct probability that the variant's true mean "
                            "exceeds control's, based on your observed data."
                        )
                except ValueError as e:
                    st.error(str(e))

st.divider()
st.caption("Built with Streamlit · scipy · pandas · plotly — see README.md for setup instructions.")
