"""Render the rates validation chapter from saved, reproducible results."""

import csv
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

matplotlib.use("Agg")

INK = "#17364a"
TEAL = "#13877b"
ORANGE = "#ba6a31"


def read(path, name):
    with (path / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _figures(path, config):
    target = path / "figures"
    target.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "figure.dpi": 145,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(target / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    for method, col in (("log_linear", INK), ("monotone_convex", TEAL)):
        x = read(path, f"forward_{method}")
        ax.plot(
            [float(v["time"]) for v in x],
            [100 * float(v["forward"]) for v in x],
            label=method.replace("_", " "),
            color=col,
        )
    ax.set(xlabel="Tenor in ACT/360 years", ylabel="Instantaneous forward (%)")
    ax.legend(frameon=False)
    save(fig, "ois_curve")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
    exercise = read(path, "exercise_loss")
    cal = read(path, "hw_calibration")
    for world, col in (("g2_base", INK), ("g2_second", TEAL)):
        x = [r for r in exercise if r["world"] == world]
        xy = [[float(r[c]) for c in ("a", "loss_mean_bp", "loss_se_bp")] for r in x]
        a, loss, se = np.asarray(xy).T
        ax[1].errorbar(a, loss, yerr=1.96 * se, color=col, marker=".", label=world)
        r = [r for r in cal if r["world"] == world and r["noisy"] == "0"]
        a = sorted({float(v["a"]) for v in r})
        pv = [float(next(v["hw_bermudan_bp"] for v in r if float(v["a"]) == aa)) for aa in a]
        ax[0].plot(a, pv, "o-", color=col, label=world)
        ax[0].axhline(float(r[0]["market_adi_bp"]), ls=":", color=col)
    ax[0].set(xlabel="HW mean reversion per year", ylabel="Bermudan PV (notional bp)")
    ax[1].set(xlabel="HW mean reversion per year", ylabel="Paired exercise loss (notional bp)")
    ax[0].legend(frameon=False, fontsize=8)
    save(fig, "reversion")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
    data = read(path, "hedge_summary")
    for name, col in (
        ("single_delta", INK),
        ("key_rate", TEAL),
        ("key_rate_vega", ORANGE),
        ("full_info_all", "#765285"),
    ):
        r = [
            x
            for x in data
            if x["world"] == "g2_base"
            and x["strategy"] == name
            and x["holder"] == "true"
            and float(x["cost_multiple"]) == 0
            and float(x["a"]) == config["calibration"]["mean_reversion"]
        ]
        ax[0].plot(
            [float(x["days"]) for x in r],
            [float(x["sd"]) * 1e4 for x in r],
            "o-",
            color=col,
            label=name.replace("_", " "),
        )
    ax[0].set(xlabel="Days between rebalances", ylabel="Discounted P&L SD (notional bp)")
    ax[0].legend(frameon=False, fontsize=7)
    rec = read(path, "annuity_reconciliation")
    for scale, color in zip(sorted({r["grid_multiple"] for r in rec}), [INK, TEAL, ORANGE]):
        r = [x for x in rec if x["grid_multiple"] == scale]
        ax[1].loglog(
            [int(x["rehedge_days"]) for x in r],
            [float(x["rmse"]) * 1e4 for x in r],
            "o-",
            color=color,
            label=f"Simulation mesh ×{scale}",
        )
    ax[1].legend(frameon=False, fontsize=7)
    ax[1].set(xlabel="Days between rebalances", ylabel="Annuity identity RMSE (notional bp)")
    save(fig, "hedging")
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    w = [r for r in read(path, "pnl_explain") if r["path"] == "0"]
    days = [int(r["day"]) - int(w[0]["day"]) + 1 for r in w]
    for key, col in (("hpl", INK), ("rtpl", TEAL)):
        ax.plot(days, [float(r[key]) * 1e4 for r in w], lw=0.8, color=col, label=key.upper())
    for x, r in zip(days, w):
        if r["event"] != "none":
            ax.axvline(x, color=ORANGE, ls=":")
    ax.set(xlabel="Observation in the fixed 250-day window", ylabel="Frozen-book P&L (bp)")
    ax.legend(frameon=False)
    save(fig, "pla")
    return target


def render(path, config):
    images = _figures(path, config)
    styles = getSampleStyleSheet()
    story = []

    def heading(name, title):
        if story:
            story.append(PageBreak())
        story.append(Paragraph(name, styles["Heading1"]))
        story.append(Paragraph(title, styles["Title"]))

    def text(txt):
        story.append(Paragraph(txt, styles["BodyText"]))
        story.append(Spacer(1, 11))

    def img(name):
        filename = str(images / f"{name}.png")
        width, height = ImageReader(filename).getSize()
        story.append(Image(filename, width=484, height=484 * height / width))

    def table(headers, rows, widths):
        t = Table([headers, *rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eaf2f4"), colors.white]),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    b = read(path, "bermudan")
    b2 = read(path, "bermudan_bounds")
    conv = [r for r in b if r["world"] == "g2_base" and "ADI" in r["method"]]
    benchmark = float(conv[-1]["notional_bp"])
    spread = abs(float(conv[-1]["notional_bp"]) - float(conv[0]["notional_bp"]))
    cr = read(path, "curve_repricing")
    bound = [r for r in b2 if r["world"] == "g2_base"]
    exercise = [
        r
        for r in read(path, "exercise_loss")
        if r["world"] == "g2_base" and float(r["a"]) == config["calibration"]["mean_reversion"]
    ][0]
    pla = read(path, "pla")
    stats = Counter(x["zone"] for x in pla)
    exercised = sum(int(x["exercises"]) for x in pla)
    findings = [
        f"Both OIS curves reprice the complete input strip within {max(float(r['max_par_error_bp']) for r in cr):.2g} bp.",
        f"The fine base G2++ Bermudan ADI mark is {benchmark:.2f} notional bp. Grid refinement changes it by {spread:.2f} bp.",
        f"The base paired HW exercise loss averages {float(exercise['loss_mean_bp']):.2f} notional bp "
        f"with a 95% interval [{float(exercise['loss_ci_low_bp']):.2f}, {float(exercise['loss_ci_high_bp']):.2f}].",
        f"Against true G2++ marks, the HW desk PLA classified {stats['green']} green, "
        f"{stats['amber']} amber and {stats['red']} red paths, with {exercised} actual "
        f"exercise{'' if exercised == 1 else 's'} in the windows.",
    ]
    hedge = [
        r
        for r in read(path, "hedge_summary")
        if r["holder"] == "true"
        and float(r["cost_multiple"]) == 0
        and float(r["a"]) == config["calibration"]["mean_reversion"]
    ]

    def hedge_sd(world, strategy, days):
        rows = [
            r for r in hedge if r["world"] == world and r["strategy"] == strategy and int(r["days"]) == days
        ]
        return float(rows[0]["sd"]) * 1e4 if rows else np.nan

    daily = min(int(r["days"]) for r in hedge)
    findings.append(
        f"With {daily}-day rebalancing in the base G2++ world, the three-swap key-rate hedge has a "
        f"discounted P&amp;L SD of {hedge_sd('g2_base', 'key_rate', daily):.1f} bp, against "
        f"{hedge_sd('g2_base', 'single_delta', daily):.1f} bp for one-swap delta and "
        f"{hedge_sd('g2_base', 'full_info_swaps', daily):.1f} bp for the full-information swap control."
    )
    vega = [float(r["sd"]) * 1e4 for r in hedge if r["strategy"] == "key_rate_vega"]
    if vega:
        days = sorted({int(r["days"]) for r in hedge})
        findings.append(
            f"Across both G2++ worlds and {', '.join(map(str, days))}-day rebalancing, the key-rate and "
            f"vega hedge SD stays between {min(vega):.1f} and {max(vega):.1f} bp."
        )
    if pla and "aligned_g2_zone" in pla[0]:
        for label, key in (("HW", "aligned_hw_zone"), ("G2++", "aligned_g2_zone")):
            counts = Counter(x[key] for x in pla)
            findings.append(
                f"On the shared quote snapshot against the front-office G2++, the {label} risk desk "
                f"has {counts['green']} green, {counts['amber']} amber and {counts['red']} red paths."
            )
    heading("Rates validation / Scope", "Bermudan exercise, pricing and hedging")
    text(
        "A synthetic G2++ OIS market with two parameter sets tests a co-terminal calibrated Hull–White "
        "one-factor desk model. The trade is a physically settled 10NC1 payer. A correctly specified "
        "Hull–White world supplies the one-factor hedge control."
    )
    text(
        f"Configuration {config['name']} · Seed {config['seed']} · "
        f"{config['hedging']['paths']} paths per hedge comparison · {config['pnl']['paths']} PLA paths"
    )
    for line in findings:
        text(line)
    if config["name"] == "rates-quick":
        text(
            "This smoke configuration verifies the interfaces and accounting. The reported small-sample "
            "estimate and interval are exploratory diagnostics."
        )
    heading("R1 / Conceptual soundness", "OIS curve and European benchmarks")
    img("ois_curve")
    text(
        "T+0 start, annual fixed and compounded floating payments, zero lag, modified following "
        "and ACT/360 are used throughout. OIS discounting and forecasting use one curve, with "
        "a synthetic Nelson–Siegel opening curve. Hagan–West smooths instantaneous forwards."
    )
    european_rows = read(path, "european")
    table(
        ["World", "Max MC error (normal vol bp)", "Max |z|"],
        [
            [
                name,
                f"{max(abs(float(x['error_normal_bp'])) for x in european_rows if x['world'] == name):.3f}",
                f"{max(abs(float(x['z_score'])) for x in european_rows if x['world'] == name):.2f}",
            ]
            for name in ("hw_control", "g2_base", "g2_second")
        ],
        [150, 190, 100],
    )
    text(
        "The MC uses exact forward-measure factor transitions. A separate RN simulation checks "
        "the martingale property of discounted zero-coupon bonds."
    )
    heading("R2 / Model identification", "Exercise opportunities and calibrated volatility")
    img("reversion")
    table(
        ["Base G2++ method", "PV (bp)", "95% interval (bp)"],
        [
            ["Fine HV-ADI", f"{benchmark:.2f}", "grid reference"],
            *[
                [
                    x["method"],
                    f"{float(x['mean_bp']):.2f}",
                    f"[{float(x['ci_low_bp']):.2f}, {float(x['ci_high_bp']):.2f}]",
                ]
                for x in bound
            ],
        ],
        [225, 85, 135],
    )
    text(
        "Co-terminal quotes identify exercise-date state variances conditional on a. An infeasible "
        "variance strip takes the nonnegative weighted fitting route. The quote-Greek comparison "
        "flags calibration constraints and exercise-mask switches. The nested upper and out-of-sample "
        "lower price estimates give statistical context for the estimated policy loss."
    )
    bound_width = float(bound[1]["ci_high_bp"]) - float(bound[0]["ci_low_bp"])
    loss = abs(float(exercise["loss_mean_bp"]))
    if bound_width > loss:
        text(
            f"The base price confidence bracket spans {bound_width:.2f} bp, wider than the "
            f"{loss:.2f} bp absolute point estimate of exercise loss. The experiment leaves "
            "uncertainty about the loss relative to the true optimum."
        )
    heading("R3 / Outcomes analysis", "Hedging and stochastic funding")
    img("hedging")
    text(
        "Books finance on the exact Gaussian integrated bank account. Coupon settlement and fair "
        "liquidation occur at the observed exercise event. Variance comparisons resample paired "
        "paths. Quote and factor hedge fits share the same position limits, with a separate uncapped "
        "two-swap spanning control. A positive-weight tie-break resolves redundant positions."
    )
    text(
        "Quote hedges split each OIS key-rate node linearly between the neighbouring hedge-swap "
        "maturities, which preserves the parallel DV01. The vega hedge excludes options that "
        "expire by the next rebalance and moves their vega node to the first surviving expiry."
    )
    text(
        "The European gamma-identity experiment includes the cash-funded annuity position. "
        "The two mesh refinements share the same exact Gaussian tape."
    )
    heading("R3 / Ongoing monitoring", "Frozen P&amp;L attribution")
    img("pla")
    comparisons = [
        ("True G2++ marks vs HW desk (diagnostic)", "zone", "spearman"),
        ("Front-office G2++ vs HW desk", "aligned_hw_zone", "aligned_hw_spearman"),
        ("Front-office G2++ vs G2++ risk desk", "aligned_g2_zone", "aligned_g2_spearman"),
    ]
    rows = []
    for label, zone_key, rho_key in comparisons:
        counts = Counter(x.get(zone_key, "undefined") for x in pla)
        rhos = [float(x[rho_key]) for x in pla if x.get(rho_key) not in (None, "", "nan")]
        median = f"{np.median(rhos):.2f}" if rhos else "n/a"
        rows.append([label, *(str(counts[z]) for z in ("green", "amber", "red")), median])
    table(["HPL against RTPL", "Green", "Amber", "Red", "Median Spearman"], rows, [210, 50, 50, 40, 95])
    text(
        "The diagnostic row compares noise-free true marks with a desk that sees noisy quotes, so "
        "quote noise counts against the desk. The aligned rows value the same frozen book on one "
        "daily quote snapshot and calibration set, as MAR32 data alignment permits. Mean reversions "
        "and the factor correlation are structural inputs, and both G2++ desks refit the two factor "
        "volatilities to every retained quote each day. The risk desk rounds the structural inputs and "
        "uses a coarser ADI grid. A hedged book has small daily P&amp;L, so the remaining model gap "
        "decides the zone."
    )
    labels = {
        "daily_rho": "Both desks refit rho daily",
        "risk_atm_only": "Risk desk fits ATM quotes only",
    }
    sens_file = path / "pla_sensitivity.csv"
    if sens_file.exists():
        sens = read(path, "pla_sensitivity")
        rows = []
        for variant in dict.fromkeys(r["variant"] for r in sens):
            group = [r for r in sens if r["variant"] == variant]
            counts = Counter(r["zone"] for r in group)
            rhos = [float(r["spearman"]) for r in group if r["spearman"] not in ("", "nan")]
            rows.append(
                [
                    labels.get(variant, variant),
                    *(str(counts[z]) for z in ("green", "amber", "red")),
                    f"{np.median(rhos):.2f}" if rhos else "n/a",
                ]
            )
        if rows:
            table(
                ["G2++ risk desk sensitivity", "Green", "Amber", "Red", "Median Spearman"],
                rows,
                [210, 50, 50, 40, 95],
            )
            text(
                "Each sensitivity changes one convention of the aligned G2++ comparison. A daily rho "
                "refit is weakly identified by co-terminal quotes and moves both desks' marks with the "
                "quote noise. A risk desk calibrated to a different quote set than the front office "
                "has parameter noise that does not cancel in the comparison."
            )
    cal_file = path / "pla_desk_calibration.csv"
    if cal_file.exists():
        cal_rows = read(path, "pla_desk_calibration")
        summary = []
        for name in dict.fromkeys(r["desk"] for r in cal_rows):
            group = [r for r in cal_rows if r["desk"] == name]
            rmse = [float(r["rmse_bp"]) for r in group if r["rmse_bp"] not in ("", "nan")]
            summary.append(
                [
                    name.replace("_", " "),
                    str(len(group)),
                    f"{np.median(rmse):.3f}" if rmse else "n/a",
                    f"{100 * np.mean([r['success'] != '1' for r in group]):.1f}%",
                    f"{100 * np.mean([r['fallback'] == '1' for r in group]):.1f}%",
                ]
            )
        table(
            ["G2++ desk", "Fits", "Median RMSE (vol bp)", "Not converged", "Previous-day fallback"],
            summary,
            [150, 50, 110, 80, 105],
        )
        text(
            "A desk whose daily fit does not converge keeps the previous day's parameters for that "
            "valuation and does not use the failed fit as the next starting point."
        )
    p = read(path, "pnl_explain")
    for order in ("first", "second"):
        sq = np.mean([float(r[f"residual_{order}"]) ** 2 for r in p])
        text(
            f"{order.capitalize()}-order Taylor residual RMSE on the 250-day windows is {np.sqrt(sq) * 1e4:.4f} bp."
        )
    worst = max(p, key=lambda row: abs(float(row["residual_second"])))
    text(
        f"The largest second-order residual is {abs(float(worst['residual_second'])) * 1e4:.2f} bp "
        f"on {worst['date']}. Small-sample RMSE can be dominated by days on which the fitted price "
        "responds nonlinearly to the full quote move."
    )
    text(
        "Both books use the same actual exercise or expiry event. The new ATM contract enters "
        "the P&amp;L starting on the next business day. The CSV separates Taylor error from the full "
        "HW repricing gap against G2++ marks."
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(48, 30, "Bates Hedging Lab / Rates validation")
        canvas.drawRightString(doc.pagesize[0] - 48, 30, str(doc.page))

    document = SimpleDocTemplate(
        str(path / "rates_validation.pdf"),
        leftMargin=48,
        rightMargin=48,
        topMargin=36,
        bottomMargin=45,
        title="Rates model validation",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (path / "findings.md").write_text(
        "# Rates validation findings\n\n" + "".join(f"- {s.replace('&amp;', '&')}\n" for s in findings),
        encoding="utf-8",
    )
