"""Render publication tables, diagnostics and the model validation PDF from artifacts."""

import csv
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

matplotlib.use("Agg")

PALETTE = ("#133c57", "#168789", "#c47632", "#725282", "#405f42", "#8e3652")
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "font.size": 9,
        "figure.dpi": 140,
    }
)


def _csv(root, name):
    with (root / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row, key):
    return float(row[key])


def _save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_all(root):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    hedge = _csv(root, "hedge")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    groups = [
        ("BS", "true_delta", "BS control"),
        ("no_jump", "true_mv", "Heston / stock"),
        ("no_jump", "true_two_uncapped", "Heston / two tools"),
        ("base", "sticky_strike", "Bates / BS delta"),
        ("base", "delta_vega", "Bates / BS delta-vega"),
        ("base", "full_two", "Bates / full information"),
    ]
    for (world, strategy, label), color in zip(groups, PALETTE):
        rows = [
            r
            for r in hedge
            if (r["world"], r["strategy"], r["stock_cost"], r["product"])
            == (world, strategy, "0.0", "vanilla")
        ]
        rows.sort(key=lambda r: _f(r, "dt"))
        ax.plot(
            [_f(r, "dt") * 252 for r in rows], [_f(r, "sd") for r in rows], "o-", label=label, color=color
        )
    ax.set(
        xlabel="Trading days between rebalances",
        ylabel="Terminal P&L standard deviation",
        xscale="log",
        yscale="log",
    )
    ax.legend(frameon=False, ncol=2)
    _save(fig, figures, "hedge_frequency")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for ax, product in zip(axes, ("vanilla", "barrier")):
        for label, col in zip(("sticky_strike", "delta_vega", "calibrated_two", "no_trade_band"), PALETTE):
            rows = [
                r
                for r in hedge
                if r["world"] == "base"
                and r["strategy"] == label
                and r["product"] == product
                and _f(r, "stock_cost") > 0
            ]
            ax.scatter(
                [_f(r, "mean_cost") for r in rows], [_f(r, "sd") for r in rows], label=label, color=col, s=16
            )
            best = []
            for a in rows:
                if not any(
                    _f(b, "mean_cost") <= _f(a, "mean_cost")
                    and _f(b, "sd") <= _f(a, "sd")
                    and (_f(b, "mean_cost") < _f(a, "mean_cost") or _f(b, "sd") < _f(a, "sd"))
                    for b in rows
                ):
                    best.append(a)
            best.sort(key=lambda r: _f(r, "mean_cost"))
            ax.plot([_f(r, "mean_cost") for r in best], [_f(r, "sd") for r in best], color=col, lw=0.8)
        ax.set(title=product.capitalize(), xlabel="Mean execution cost", ylabel="P&L standard deviation")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, figures, "frontier")

    rr = _csv(root, "reconciliation")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, level in enumerate(sorted({row["grid_refinement"] for row in rr})):
        points = []
        for step in sorted({_f(row, "trade_dt") for row in rr}):
            a = [
                _f(row, "residual")
                for row in rr
                if row["grid_refinement"] == level and _f(row, "trade_dt") == step
            ]
            points.append((252 * step, np.sqrt(np.mean(np.square(a)))))
        ax.loglog(*np.array(points).T, "o-", color=PALETTE[i], label=f"Simulation refinement x{level}")
    ax.set(xlabel="Trading days between rebalances", ylabel="Pathwise reconciliation RMSE")
    ax.legend(frameon=False)
    _save(fig, figures, "reconciliation")

    calib = _csv(root, "calibration")
    params = ("v0", "kappa", "theta", "xi", "rho")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    for ax, key in zip(axes.ravel(), params):
        for i, world in enumerate(("Heston", "Bates")):
            clean = [_f(r, key) for r in calib if r["world"] == world and r["experiment"] == "clean"][0]
            groups = [
                [
                    _f(r, key)
                    for r in calib
                    if r["world"] == world and r["experiment"] == "noisy" and _f(r, "noise") == noise
                ]
                for noise in (0.5, 1, 2)
            ]
            pos = np.arange(3) + 1 + (i - 0.5) * 0.22
            bp = ax.boxplot(groups, positions=pos, widths=0.20, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(PALETTE[i])
                patch.set_alpha(0.50)
            ax.axhline(clean, color=PALETTE[i], lw=0.9)
        ax.set(title=key, xticks=[1, 2, 3], xticklabels=["0.5", "1", "2"])
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.03, 0.75, "Heston-on-Heston", color=PALETTE[0])
    axes.ravel()[-1].text(0.03, 0.55, "Heston-on-Bates", color=PALETTE[1])
    axes.ravel()[-1].text(0.03, 0.35, "Horizontal lines: clean fits", color="#454545")
    _save(fig, figures, "calibration")

    residual = _csv(root, "calibration_residuals")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, world in zip(axes, ("Heston", "Bates")):
        rows = [row for row in residual if row["world"] == world and row["experiment"] == "clean"]
        times = sorted({_f(row, "tenor") for row in rows})
        log_k = sorted({_f(row, "log_moneyness") for row in rows})
        matrix = np.array([_f(row, "residual_iv_bp") for row in rows]).reshape(len(times), len(log_k))
        lim = max(np.nanmax(abs(matrix)), 0.1)
        img = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
        ax.set(
            title=world + " clean",
            xlabel="Forward log moneyness",
            ylabel="Tenor (years)",
            xticks=range(len(log_k)),
            xticklabels=[f"{k:.2f}" for k in log_k],
            yticks=range(len(times)),
            yticklabels=[f"{t:g}" for t in times],
        )
        ax.tick_params(axis="x", labelrotation=60, labelsize=6)
        fig.colorbar(img, ax=ax, label="IV bp")
    _save(fig, figures, "calibration_residuals")

    drift = [
        row
        for row in _csv(root, "daily_calibration")
        if (row["product"], row["world"], row["path"]) == ("vanilla", "base", "0")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, key in zip(axes, ("v0", "xi", "rho")):
        for prefix, col in zip(("clean", "full", "weekly"), PALETTE):
            ax.plot(
                [_f(r, "day") for r in drift],
                [_f(r, prefix + "_" + key) for r in drift],
                ".-",
                lw=0.9,
                color=col,
                label=prefix,
            )
        ax.set(title=key, xlabel="Trading day")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, figures, "drift")

    greek = _csv(root, "barrier_greeks")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for column, slice_name in enumerate(("one_day", "near_monitor")):
        for row, key in enumerate(("delta", "gamma")):
            ax = axes[row, column]
            for method, c in zip(
                ("pathwise", "likelihood_ratio", "crn_fd_0.35", "one_step_survival", "survival_lr"), PALETTE
            ):
                values = [
                    r
                    for r in greek
                    if r["monitor_slice"] == slice_name and r["greek"] == key and r["algorithm"] == method
                ]
                if not values:
                    continue
                x = np.array([_f(r, "spot") for r in values])
                y = np.array([_f(r, "estimate") for r in values])
                err = np.array([1.96 * _f(r, "standard_error") for r in values])
                ax.plot(x, y, label=method, color=c, lw=0.85)
                ax.fill_between(x, y - err, y + err, color=c, alpha=0.10)
            values = [
                r
                for r in greek
                if r["monitor_slice"] == slice_name
                and r["greek"] == key
                and r["algorithm"] == "one_step_survival"
            ]
            ax.plot(
                [_f(r, "spot") for r in values],
                [_f(r, "reference") for r in values],
                "k--",
                label="CN Richardson",
                lw=1,
            )
            ax.axvline(125, color="0.45", lw=0.7)
            ax.set(title=slice_name.replace("_", " "), ylabel=key, xlabel="Spot before fixing")
    axes[0, 0].legend(frameon=False, fontsize=6)
    _save(fig, figures, "barrier_greeks")

    explain = _csv(root, "pnl_explain")
    residual = np.array(
        [[_f(r, key) for key in ("first_residual", "second_residual", "market_model_gap")] for r in explain]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    row = max((r for r in explain if r["path"] == "0"), key=lambda r: abs(_f(r, "hpl")))
    components = ("delta", "vega", "theta", "gamma", "vanna", "volga", "second_residual", "market_model_gap")
    vals = np.array([_f(row, key) for key in components])
    axes[0].bar(
        components,
        vals,
        bottom=np.r_[0, np.cumsum(vals)[:-1]],
        color=np.where(vals >= 0, PALETTE[1], PALETTE[2]),
    )
    axes[0].tick_params(axis="x", labelrotation=63, labelsize=6)
    axes[0].set(title=f"Frozen book, path 0 day {row['day']}", ylabel="Incremental P&L")
    lim = max(np.quantile(np.abs(residual), 0.99), 1e-8)
    for i, name in enumerate(("Taylor first", "Taylor second", "Market/model gap")):
        axes[1].hist(
            residual[:, i], bins=np.linspace(-lim, lim, 32), alpha=0.45, label=name, color=PALETTE[i]
        )
    axes[1].set(title="Residuals, central 98% pooled range", xlabel="Daily P&L", ylabel="Observations")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, figures, "pnl")

    data = _csv(root, "spread_prices")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, col in zip((0.0, 5.0, 12.0, 24.0, 40.0), PALETTE):
        rows = [r for r in data if _f(r, "strike") == k]
        x = [_f(r, "rho") for r in rows]
        axes[0].plot(x, [_f(r, "quadrature_price") for r in rows], ".-", color=col, label=f"K={k:g}")
        axes[1].plot(x, [_f(r, "kirk_error") for r in rows], ".-", color=col)
        axes[1].plot(x, [_f(r, "bjerksund_error") for r in rows], ":", color=col)
    axes[0].set(xlabel="Correlation", ylabel="Reference price")
    axes[1].set(
        xlabel="Correlation", ylabel="Price error", title="Kirk (solid), Bjerksund–Stensland (dotted)"
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    _save(fig, figures, "commodity")
    return figures


def render(root, config):
    figures = plot_all(root)
    pricing = _csv(root, "pricing")
    calibration = _csv(root, "calibration")
    quote = _csv(root, "quotes")
    variance = _csv(root, "hedge_variance_fit")
    explain = _csv(root, "pnl_residual_summary")
    drift = _csv(root, "daily_calibration")
    spreads = _csv(root, "spread_hedge")
    pla = json.loads((root / "pla_zone_probability.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "calibration_diagnostics.json").read_text(encoding="utf-8"))
    style = getSampleStyleSheet()
    style.add(
        ParagraphStyle(
            name="BodyLab",
            parent=style["BodyText"],
            fontSize=10,
            leading=15,
            spaceAfter=11,
            textColor=colors.HexColor("#263647"),
            alignment=TA_LEFT,
        )
    )
    style.add(
        ParagraphStyle(
            name="Kicker",
            parent=style["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=18,
            textColor=colors.HexColor("#168789"),
        )
    )
    style["Title"].fontSize = 30
    style["Title"].leading = 34
    style["Title"].alignment = TA_LEFT
    style["Heading1"].fontSize = 23
    style["Heading1"].leading = 27
    story = []

    def paragraph(value, kind="BodyLab"):
        story.append(Paragraph(value, style[kind]))

    def page(title, section):
        if story:
            story.append(PageBreak())
        paragraph(section.upper(), "Kicker")
        paragraph(title, "Heading1")

    def table(headers, rows, width=None):
        t = Table([headers, *rows], colWidths=width, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163c55")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eff4f6"), colors.white]),
                ]
            )
        )
        story.extend((t, Spacer(1, 12)))

    def image(name, height=3.4):
        img = Image(str(figures / f"{name}.png"))
        ratio = img.imageWidth / img.imageHeight
        img.drawWidth = min(6.95 * inch, height * inch * ratio)
        img.drawHeight = img.drawWidth / ratio
        story.append(img)

    def selected(seq, **criteria):
        return next(r for r in seq if all(r[k] == v for k, v in criteria.items()))

    bs = selected(variance, product="vanilla", world="BS", strategy="true_delta")
    delta = selected(variance, product="vanilla", world="base", strategy="sticky_strike")
    vega = selected(variance, product="vanilla", world="base", strategy="delta_vega")
    all_explain = selected(explain, sample="all")
    fft = max(abs(_f(r, "error_bp")) for r in pricing if r["method"].startswith("FFT"))
    bad = [r for r in drift if r["world"] == "base" and r["product"] == "vanilla"]
    failures = np.mean([_f(r, "full_fallback") for r in bad])
    findings = [
        f"The BS control log-log standard-deviation slope was {_f(bs, 'log_sd_slope'):.3f}, "
        "against the complete-market small-step benchmark of 0.5.",
        f"The Bates vanilla variance-fit intercepts were {_f(delta, 'residual_variance_intercept'):.4f} "
        f"for sticky-strike delta and {_f(vega, 'residual_variance_intercept'):.4f} for BS delta-vega.",
        f"The largest FFT versus COS difference in the benchmark grid was {fft:.4f} implied-volatility bp.",
        f"First- and second-order Taylor residual RMSE were {_f(all_explain, 'first_rmse_over_bs'):.1%} "
        f"and {_f(all_explain, 'second_rmse_over_bs'):.1%} of full BS P&amp;L RMSE.",
        f"Daily full recalibration fell back on {failures:.1%} of base vanilla path-days.",
    ]
    page("Model validation report", "Bates hedging laboratory")
    paragraph("Using Black–Scholes to hedge a stochastic-volatility market with jumps", "Title")
    paragraph(
        f"Configuration <b>{config['name']}</b> &nbsp;&nbsp; Seed <b>{config['seed']}</b> &nbsp;&nbsp; "
        f"Vanilla calibrated hedges <b>{config['hedging']['paths']} paths</b>"
    )
    paragraph(
        "Scope covers European and discretely monitored up-and-out calls, with American puts for "
        "numerical benchmarking. The core validation object is the BS hedging model used by a trader "
        "who sees closes and noisy option quotes."
    )
    for finding in findings:
        paragraph(finding)
    if not config["greeks"]["aad"]:
        paragraph(
            "This configuration uses a small engineering sample and disables the JAX benchmark. "
            "The reported numbers are diagnostic and carry Monte Carlo sampling uncertainty."
        )
    page("Model use and acceptance criteria", "01 / purpose")
    paragraph(
        "The trader holds a short option and rebalances stock and, in selected variants, a rolling option. "
        "The analyst observes the latent variance and uses it to build controls and to reconcile P&amp;L. "
        "The cash account earns the configured constant rate."
    )
    table(
        ["Evidence", "Acceptance criterion"],
        [
            ["Pricing", "Fourier, simulation and numerical PDE results agree within declared tolerances"],
            ["Hedging controls", "BS discretization and true Heston two-tool cases are reported explicitly"],
            ["Information", "Trader Greeks depend on observed returns and option quotes"],
            ["Attribution", "Frozen holdings, a 250-day sample and separate Taylor error measurement"],
            ["Repeatability", "Seed, exact configuration, outputs and dependencies recorded"],
        ],
        [103, 379],
    )
    paragraph(
        "The report follows the conceptual soundness, ongoing monitoring and outcomes analysis "
        "structure from SR 11-7. The scope is a controlled synthetic validation of a hedge model."
    )
    page("Dynamics and observable data", "02 / conceptual soundness")
    paragraph(
        "Bates adds a compound Poisson process with Gaussian log jumps to Heston. "
        "Pricing and simulation share the risk-neutral parameter set. A heavy-tail control reduces v0 and "
        "theta by 0.003009 so expected annualized quadratic variation matches the base at every tenor."
    )
    paragraph(
        "Quotes add Gaussian noise in implied-volatility units and use 20% shared expiry variance. "
        "The moneyness multiplier and correlation share are modelling assumptions. The reported spread "
        "coverage uses the fixed nominal half-spread calibrated at the unit-noise scenario."
    )
    table(
        ["Noise multiplier", "Mean deletion", "Mean bid/ask coverage"],
        [
            [
                f"{s:g}x",
                f"{np.mean([_f(r, 'deletion_rate') for r in quote if _f(r, 'noise') == s]):.1%}",
                f"{np.mean([_f(r, 'spread_coverage') for r in quote if _f(r, 'noise') == s]):.1%}",
            ]
            for s in (0.5, 1.0, 2.0)
        ],
    )
    paragraph(
        "An ATM-outward deterministic filter checks cross-strike monotonicity and convexity and common "
        "forward log-moneyness calendar ordering. Reapplying the retention mask to clean data measures "
        "the effect of missing nodes separately."
    )
    page("Pricing cross-validation", "03 / implementation tests")
    selected_methods = {
        "FFT_alpha_1.25",
        "MC_qe",
        "ADI_hv",
        "CN_Rannacher_400",
        "HV_ADI",
        "IMEX_ADI",
        "CN_daily",
        "ADI_IT",
        "CN_PSOR",
        "cross_fitted_LSMC",
    }
    data = [r for r in pricing if r["method"] in selected_methods]
    table(
        ["Model", "Product", "Method", "Price", "Error (bp)", "Seconds"],
        [
            [
                r["model"],
                r["product"],
                r["method"],
                f"{_f(r, 'price'):.5f}",
                f"{_f(r, 'error_bp'):.3f}",
                f"{_f(r, 'seconds'):.3f}",
            ]
            for r in data
        ],
    )
    paragraph(
        "European errors use BS implied-volatility bp. American puts use an American binomial BS quote "
        "mapping. Barrier differences use bp of the initial spot notional. The CSV includes absolute "
        "price errors, sensitivity to COS width and Fourier damping, and Monte Carlo standard errors."
    )
    page("Calibration and identifiability", "04 / outcomes")
    image("calibration", 3.75)
    paragraph(
        "Clean lines are optimization baselines. The boxplots compare parameter dispersion under "
        "three levels of noise. Separate matched-clean fits use the noisy quote retention masks."
    )
    table(
        ["World", "Clean fitting error (IV bp)", "Scaled Hessian condition"],
        [
            [
                w,
                f"{_f(selected(calibration, world=w, experiment='clean'), 'rmse_iv_bp'):.4f}",
                f"{diagnostics['hessian'][w]['condition']:.3g}",
            ]
            for w in ("Heston", "Bates")
        ],
    )
    page("Residual structure and daily monitoring", "05 / monitoring")
    image("calibration_residuals", 2.6)
    image("drift", 2.4)
    paragraph(
        "Daily drift is measured relative to a same-day noise-free pseudo-true fit. The state estimate "
        "can change daily. Daily full updates and weekly structural updates with daily state fitting "
        "record solver fallbacks independently. The CSVs retain raw and scaled Hessians, eigenvectors, "
        "profile fits, parameter correlations and Feller margins."
    )
    page("Hedge frequency and tool completeness", "06 / outcomes")
    image("hedge_frequency")
    paragraph(
        "Regressions use zero costs and common random paths. The fitted intercept describes residual "
        "variance for the strategy, information and instrument set. It is fitted on rebalancing intervals "
        "of at most one trading day, where the linear small-step expansion applies. The log SD slope uses "
        "every interval. The option hedge is an at-the-money call with a 126-day tenor at issue, longer "
        "than the hedged claim. A roll re-hedges both legs. Jump risk is rare and heavy tailed, so "
        "realized rankings between two-instrument hedges need many jump paths. The full-information Bates hedge "
        "uses diffusion and jump instantaneous covariation. The finite-grid comparison contains "
        "discrete rebalancing error."
    )
    table(
        ["Strategy", "Fitted intercept", "Slope of log SD"],
        [
            ["BS control", f"{_f(bs, 'residual_variance_intercept'):.4f}", f"{_f(bs, 'log_sd_slope'):.3f}"],
            [
                "Bates / BS delta",
                f"{_f(delta, 'residual_variance_intercept'):.4f}",
                f"{_f(delta, 'log_sd_slope'):.3f}",
            ],
            [
                "Bates / BS delta-vega",
                f"{_f(vega, 'residual_variance_intercept'):.4f}",
                f"{_f(vega, 'log_sd_slope'):.3f}",
            ],
        ],
    )
    page("Hedging costs and pathwise reconciliation", "07 / outcomes")
    image("frontier", 2.65)
    image("reconciliation", 2.55)
    paragraph(
        "The cost frontier leaves the trade-off to the desk's risk preference. The reconciliation tracks "
        "premium convention, discounted gamma exposure, and an explicit jump-by-jump correction. "
        "Two path discretizations share the Brownian tape and jump draws."
    )
    page("Barrier risk and Greek estimation", "08 / outcomes")
    image("barrier_greeks", 4.9)
    paragraph(
        "Fine CN with repeated Rannacher start-up is the discrete BS reference. At the next fixing the "
        "outcome changes discontinuously at the barrier. Pathwise differentiation omits that boundary "
        "term. Survival conditioning and the likelihood ratio address it with different variances."
    )
    page("P&amp;L explain and attribution testing", "09 / outcomes")
    image("pnl", 3.25)
    table(
        ["ATM estimator", "Green", "Amber", "Red", "Undefined"],
        [
            [method, *[f"{p[z]:.1%}" for z in ("green", "amber", "red", "undefined")]]
            for method, p in pla.items()
        ],
    )
    paragraph(
        "P&amp;L explain keeps the Taylor approximation error separate from the BS-versus-Bates mark gap. "
        "PLA revalues the previous close's frozen holdings on 250 consecutive business days per path. "
        "The RTPL uses only spot and ATM tenor nodes, excluding the target option."
    )
    page("Commodity spread and correlation risk", "10 / extension")
    image("commodity", 2.8)
    table(
        ["Correlation driver", "Hedge", "P&L SD", "Local residual variance"],
        [
            [
                r["driver"],
                r["strategy"],
                f"{_f(r, 'sd'):.4f}",
                f"{_f(r, 'continuous_residual_variance_estimate'):.4f}",
            ]
            for r in spreads
            if r["strategy"] in ("dynamic_delta", "kirk_implied", "dynamic_local_risk")
        ],
    )
    paragraph(
        "Dynamic conditional repricing is compared with exact constant correlation, Kirk with average "
        "correlation, and Kirk with the reference option's implied correlation. Orthogonality makes "
        "dynamic-model deltas and the covariance projection coincide."
    )
    page("Limitations and monitoring actions", "11 / conclusions")
    paragraph(
        "Sizing uncertainty depends on the path count, especially for barriers, jumps, empirical CVaR, "
        "and variance intercepts. Check the paired confidence intervals before interpreting a model "
        "ranking. Convergence studies distinguish simulation-grid error from the rebalancing effect."
    )
    paragraph(
        "The market is simulated with a constant rate, daily quote updates and stylized bid/ask spreads. "
        "Rounding, surface cleaning, numerical barrier interpolation, and the finite QE and Jacobi "
        "steps affect short-maturity and tail results."
    )
    paragraph(
        "Monitor quote deletion, calibration fallback, wing errors and the pathwise distribution of "
        "unexplained P&amp;L. Revalidate these controls when the strike coverage, monitoring schedule, "
        "available hedge instruments, or market assumptions change."
    )
    paragraph(
        "References: Federal Reserve SR 11-7 (2011), Basel Framework MAR32.35–42, Andersen QE, "
        "Fang–Oosterlee COS, Carr–Madan FFT, and Bjerksund–Stensland spread valuation. "
        "Raw artifacts are listed in the project README."
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526574"))
        canvas.drawString(doc.leftMargin, 0.43 * inch, "Bates Hedging Laboratory  /  Model Validation")
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.43 * inch, str(doc.page))

    doc = SimpleDocTemplate(
        str(root / "model_validation.pdf"),
        leftMargin=40,
        rightMargin=40,
        topMargin=38,
        bottomMargin=36,
        title="Bates Hedging Laboratory | Model Validation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (root / "findings.md").write_text(
        "# Quantitative findings\n\n" + "\n".join(f"- {x.replace('&amp;', '&')}" for x in findings) + "\n",
        encoding="utf-8",
    )
