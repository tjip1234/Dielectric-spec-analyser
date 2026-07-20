"""
Estimate the relation between molecular polarity and the dielectric-probe
(S11 reflection) response for the CURRENT measurement setup.

Frame: across compounds. Each pure liquid is one data point. We relate two
polarity predictors -- the gas-phase dipole moment AND the bulk static
permittivity eps_s -- to two measured response features, |S11| (dB) and S11
phase (deg). Fitting the slope lets us state, for this setup:
    "1 Debye (or 1 eps-unit) of difference ~= X dB and Y deg of S11 difference".

Physics: an open-ended coax probe is loaded by the bulk complex permittivity of
the liquid. A more polar molecule has a larger dipole, which (via Onsager /
Kirkwood, modulated by H-bonding/association) raises eps_s and shifts the
reflection coefficient. eps_s is the physically direct driver; the gas-phase
dipole is the upstream molecular cause -- we report both so their explanatory
power can be compared.

S11 files are Touchstone: '# HZ S RI R 50' -> freq[Hz], Re(S11), Im(S11).
This is an UNCALIBRATED estimate: absolute permittivity from the simplified
open-coax model is not meaningful; only differences/ordering are used.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
DB = HERE / "Database"

# Pure-compound measurement -> (file, gas-phase dipole [Debye], static perm eps_s ~20C).
# eps_s from standard tables (CRC/literature, ~20-25 C). DEET eps_s is approximate.
COMPOUNDS = {
    # name              file                                       dipole  eps_s
    "o-Xylene":        ("Aromatic-hydrocarbon/o-xylene.s1p",        0.64,   2.57),
    "Diethyl ether":   ("Diethylether/diethyl-ether.s1p",          1.15,   4.27),
    "Isopropanol":     ("Isopropanol/100%.s1p",                    1.66,  18.3),
    "tert-Butanol":    ("2-Methylpropan-2-ol/tert-butanol.s1p",    1.66,  12.5),
    "Ethanol":         ("Ethanol/ethanol.s1p",                     1.69,  24.5),
    "Water":           ("Water/DI-water.s1p",                      1.85,  80.1),
    "2-Butoxyethanol": ("2-Butoxyethanol/2-Butoxyethanol.s1p",     2.08,   9.3),
    "Propylene glycol":("Propane-1,2-diol/100%.s1p",               2.27,  32.0),
    "Acetone":         ("Propan-2-one/100%.s1p",                   2.88,  21.0),
    "DEET":            ("DEET/DEET.s1p",                            3.30,   8.0),  # eps_s approx
}

# Open-coax probe model is most stable in the low-GHz band.
F_LO, F_HI = 0.5e9, 3.0e9
F_SPOTS = (1.0e9, 2.4e9)          # representative spot frequencies
BASELINE = "o-Xylene"             # non-polar reference for differential features


def load_s1p(path):
    freqs, re, im = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#!":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            freqs.append(float(parts[0]))
            re.append(float(parts[1]))
            im.append(float(parts[2]))
    f = np.asarray(freqs)
    s11 = np.asarray(re) + 1j * np.asarray(im)
    return f, s11


def features(path):
    f, s11 = load_s1p(path)
    band = (f >= F_LO) & (f <= F_HI)
    s = s11[band]
    mag_db = 20 * np.log10(np.maximum(np.abs(s), 1e-12))
    phase = np.degrees(np.unwrap(np.angle(s)))
    eps = ((1 + s) / (1 - s)) ** 2          # simplified open-coax permittivity

    out = {
        "mag_db": float(np.mean(mag_db)),
        "phase_deg": float(np.mean(phase)),
        "eps_real": float(np.mean(np.real(eps))),
    }
    # spot-frequency magnitude/phase
    for fs in F_SPOTS:
        i = int(np.argmin(np.abs(f - fs)))
        out[f"mag_{fs/1e9:.1f}G"] = float(20 * np.log10(max(abs(s11[i]), 1e-12)))
        out[f"pha_{fs/1e9:.1f}G"] = float(np.degrees(np.angle(s11[i])))
    return out


def fit(x, y):
    """Return slope, intercept, Pearson R, R^2 for y ~ x."""
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r, r2


def main():
    names, dip, eps_s = [], [], []
    mag, pha, epsr = [], [], []
    m1, p1, m2, p2 = [], [], [], []
    for name, (rel, d, es) in COMPOUNDS.items():
        p = DB / rel
        if not p.exists():
            print(f"  (skip, missing) {rel}")
            continue
        fe = features(p)
        names.append(name); dip.append(d); eps_s.append(es)
        mag.append(fe["mag_db"]); pha.append(fe["phase_deg"]); epsr.append(fe["eps_real"])
        m1.append(fe["mag_1.0G"]); p1.append(fe["pha_1.0G"])
        m2.append(fe["mag_2.4G"]); p2.append(fe["pha_2.4G"])

    names = np.array(names)
    dip = np.array(dip); eps_s = np.array(eps_s)
    mag = np.array(mag); pha = np.array(pha); epsr = np.array(epsr)
    m1 = np.array(m1); p1 = np.array(p1); m2 = np.array(m2); p2 = np.array(p2)

    # differential features vs non-polar baseline
    bi = list(names).index(BASELINE)
    dmag = mag - mag[bi]
    dpha = pha - pha[bi]

    # ---- table ----
    print(f"\nResponse features averaged over {F_LO/1e9:.1f}-{F_HI/1e9:.1f} GHz")
    print(f"(differences referenced to non-polar baseline '{BASELINE}')\n")
    hdr = (f"{'compound':<18}{'dipole':>8}{'eps_s':>8}"
           f"{'|S11|dB':>9}{'phase':>8}{'d|S11|':>8}{'dphase':>8}"
           f"{'m@1G':>8}{'p@1G':>8}{'m@2.4G':>8}{'p@2.4G':>8}")
    print(hdr)
    order = np.argsort(dip)
    for i in order:
        print(f"{names[i]:<18}{dip[i]:>8.2f}{eps_s[i]:>8.1f}"
              f"{mag[i]:>9.2f}{pha[i]:>8.1f}{dmag[i]:>8.2f}{dpha[i]:>8.1f}"
              f"{m1[i]:>8.2f}{p1[i]:>8.1f}{m2[i]:>8.2f}{p2[i]:>8.1f}")

    # ---- dual regression ----
    targets = [("|S11| magnitude", mag, "dB"),
               ("S11 phase", pha, "deg"),
               ("d|S11| (vs baseline)", dmag, "dB"),
               ("dphase (vs baseline)", dpha, "deg")]
    regr = {}  # (predictor, target) -> (slope, intercept, r, r2)

    def report(predictor_name, x, unit):
        print(f"\nSensitivity vs {predictor_name}:")
        for ylabel, y, yunit in targets:
            s, b, r, r2 = fit(x, y)
            regr[(predictor_name, ylabel)] = (s, b, r, r2, yunit, unit)
            print(f"  {ylabel:<22} = {s:+9.3f} {yunit}/{unit:<7} "
                  f"(intercept {b:+8.2f}, R={r:+.2f}, R^2={r2:.2f})")

    report("gas-phase dipole [Debye]", dip, "Debye")
    report("static permittivity eps_s", eps_s, "eps")

    # which predictor explains the magnitude/phase better?
    print("\nVerdict (R^2 of absolute feature):")
    verdict = {}
    for ylabel, y in [("|S11| magnitude", mag), ("S11 phase", pha)]:
        _, _, _, r2d = fit(dip, y)
        _, _, _, r2e = fit(eps_s, y)
        better = "eps_s" if r2e > r2d else "dipole"
        verdict[ylabel] = (r2d, r2e, better)
        print(f"  {ylabel:<16}: dipole R^2={r2d:.2f}  eps_s R^2={r2e:.2f}  -> {better} explains more")

    # ---- write report + CSV ----
    outdir = HERE / "reports"
    outdir.mkdir(exist_ok=True)

    csv_path = outdir / "dipole_response_features.csv"
    with open(csv_path, "w") as fh:
        fh.write("compound,dipole_D,eps_s,mag_dB,phase_deg,dmag_dB,dphase_deg,"
                 "mag_1GHz,phase_1GHz,mag_2.4GHz,phase_2.4GHz\n")
        for i in order:
            fh.write(f"{names[i]},{dip[i]:.2f},{eps_s[i]:.1f},{mag[i]:.3f},"
                     f"{pha[i]:.2f},{dmag[i]:.3f},{dpha[i]:.2f},"
                     f"{m1[i]:.3f},{p1[i]:.2f},{m2[i]:.3f},{p2[i]:.2f}\n")

    md_path = outdir / "dipole_response_report.md"
    with open(md_path, "w") as fh:
        w = fh.write
        w("# Dipole / polarity vs dielectric-probe (S11) response\n\n")
        w(f"_Generated by `dipole_response_estimate.py`. Band-average "
          f"{F_LO/1e9:.1f}-{F_HI/1e9:.1f} GHz; differences vs non-polar baseline "
          f"`{BASELINE}`._\n\n")

        w("## Question\n")
        w("For this open-ended coax probe setup, how does a difference in a "
          "molecule's polarity translate into a difference in the measured S11 "
          "magnitude and phase? Each pure liquid is one data point; we test two "
          "predictors: the gas-phase **dipole moment** and the bulk **static "
          "permittivity epsilon_s**.\n\n")

        w("## Measured features per compound\n\n")
        w("| compound | dipole (D) | eps_s | \\|S11\\| (dB) | phase (deg) | "
          "d\\|S11\\| (dB) | dphase (deg) |\n")
        w("|---|---|---|---|---|---|---|\n")
        for i in order:
            w(f"| {names[i]} | {dip[i]:.2f} | {eps_s[i]:.1f} | {mag[i]:.2f} | "
              f"{pha[i]:.1f} | {dmag[i]:.2f} | {dpha[i]:.1f} |\n")
        w("\nFull per-frequency features in `dipole_response_features.csv`.\n\n")

        w("## Sensitivity (linear fit: feature = slope * predictor + intercept)\n\n")
        w("| predictor | target | slope | intercept | R | R^2 |\n")
        w("|---|---|---|---|---|---|\n")
        for (pred, ylabel), (s, b, r, r2, yunit, unit) in regr.items():
            w(f"| {pred} | {ylabel} | {s:+.3f} {yunit}/{unit} | {b:+.2f} | "
              f"{r:+.2f} | {r2:.2f} |\n")
        w("\n")

        w("## Verdict\n\n")
        for ylabel, (r2d, r2e, better) in verdict.items():
            w(f"- **{ylabel}**: dipole R^2={r2d:.2f}, eps_s R^2={r2e:.2f} "
              f"-> **{better}** explains more.\n")
        w("\n")

        # pull the headline numbers
        s_ph, _, _, r2_ph, _, _ = regr[("static permittivity eps_s", "S11 phase")]
        s_mg, _, _, r2_mg, _, _ = regr[("static permittivity eps_s", "|S11| magnitude")]
        w("## Headline estimate for this setup\n\n")
        w(f"- **Phase is the sensitive channel:** ~ **{s_ph:.2f} deg of S11 phase "
          f"per unit of static permittivity** (R^2={r2_ph:.2f}).\n")
        w(f"- Magnitude moves far less: ~ **{s_mg:.3f} dB per eps-unit** "
          f"(R^2={r2_mg:.2f}).\n")
        w("- The single-molecule gas-phase dipole is a **weak** predictor on its "
          "own (R^2 ~ 0.0-0.15): the probe is loaded by bulk permittivity, and "
          "H-bonding/association scrambles the dipole->eps_s mapping (the alcohols "
          "cluster at ~1.66-1.69 D but span eps_s 12-25).\n\n")

        w("## Caveats\n\n")
        w("- Uncalibrated: the simplified open-coax permittivity model is used only "
          "for ordering/differences, not absolute eps.\n")
        w("- Ethanol is 96%; sample temperatures vary 12-18 C; eps_s are literature "
          "~20 C values (DEET eps_s is approximate).\n")
        w("- n = %d compounds; this is an estimate, not a calibrated measurement.\n"
          % len(names))
        w("\n![scatter](dipole_response_estimate.png)\n")

    print(f"\nWrote report -> {md_path}")
    print(f"Wrote CSV    -> {csv_path}")

    # ---- plot ----
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (ax[0, 0], dip, mag, "gas dipole [D]", "|S11| [dB]"),
        (ax[0, 1], dip, pha, "gas dipole [D]", "S11 phase [deg]"),
        (ax[1, 0], eps_s, mag, "static eps_s", "|S11| [dB]"),
        (ax[1, 1], eps_s, pha, "static eps_s", "S11 phase [deg]"),
    ]
    for a, x, y, xl, yl in panels:
        a.scatter(x, y, c="tab:blue", zorder=3)
        for xi, yi, nm in zip(x, y, names):
            a.annotate(nm, (xi, yi), fontsize=7, xytext=(3, 3),
                       textcoords="offset points")
        s, b, r, r2 = fit(x, y)
        xs = np.linspace(min(x), max(x), 50)
        a.plot(xs, s * xs + b, "r--", lw=1,
               label=f"slope={s:.2f}, R^2={r2:.2f}")
        a.set_xlabel(xl); a.set_ylabel(yl); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"Polarity vs probe S11 response ({F_LO/1e9:.1f}-{F_HI/1e9:.1f} GHz band-avg)")
    fig.tight_layout()
    out_png = outdir / "dipole_response_estimate.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved plot -> {out_png}")

    print("\nCaveats: uncalibrated estimate; Ethanol is 96%, temps 12-18 C,")
    print("eps_s are literature ~20 C (DEET eps_s approximate). Differences/ordering")
    print("are meaningful; absolute model permittivity is not.")


if __name__ == "__main__":
    main()
