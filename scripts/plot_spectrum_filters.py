"""Overplot filter system throughputs (from the ETC instrument configs) on a
comet spectrum. The emission-line labels and default legend are set up for the
109P/Swift-Tuttle spectrum of Cochran (personal comm.; Fink & Hicks 1996).

Usage:
    python scripts/plot_spectrum_filters.py SPECTRUM -f INSTRUMENT FILTER [LABEL] [-f ...] [-o out.png]

SPECTRUM is a text file of wavelength (nm) and relative intensity columns.
INSTRUMENT is a config name from src/etc/data (with or without .toml), FILTER is
one of the names in its filterlist, and an optional LABEL overrides the default
"<instrument name> <filter>" label, e.g.

    python scripts/plot_spectrum_filters.py 109P.txt -f LSST_VRO LSST::g -f LSST_VRO LSST::r \\
        -f ESO_VLT_FORS2 ESO::FORS::g -f ESO_VLT_FORS2 ESO::FORS::R
"""

import argparse
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import toml

from etc import data
from etc.etc import ETC

SPECTRUM_LABEL = "109P/Swift-Tuttle\n(Cochran, personal comm.,\nFink & Hicks 1996)"
LINESTYLES = ["-", "--", ":", "-."]
# (label, text (x nm, y intensity), tick end (x nm, y intensity) or None)
LINES = [
    ("OH\n(0,0)", (309, 40), None),
    ("OH\n(1,1)", (318, 9), None),
    ("NH\n(0,0)", (336, 4), None),
    ("CN\nΔv=0", (396, 12), (389, 9)),
    ("C$_3$", (403, 3.2), None),
    ("CN\nΔv=-1", (421, 6.5), (421, 1.7)),
    ("CH", (431, 2.6), (431, 1.4)),
    ("C$_2$", (438, 1.9), (437, 1.35)),
    ("C$_2$\nΔv=1", (470, 4.5), None),
    ("C$_2$\nΔv=0", (515, 6), None),
    ("C$_2$\nΔv=-1", (560, 2.0), None),
    ("NH$_2$\n(10,0) Π", (573, 4.2), (573, 1.1)),
    ("C$_2$\nΔv=-2", (600, 1.6), None),
    ("[OI]", (624, 2.4), (630, 1.3)),
    ("NH$_2$\n(8,0) Π", (637, 4.2), (633, 1.1)),
    ("C$_2$\nΔv=-3", (660, 1.4), None),
    ("NH$_2$\n(6,0) Π", (701, 4.2), (701, 1.1)),
    ("CN\n(2,0) (3,1)", (795, 1.5), None),
]


def filter_label(instrument_name, filtername):
    short = filtername.split("::")[-1]
    if len(short) == 2 and short[1] == "p":
        short = short[0] + "'"
    return f"{instrument_name} {short}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spectrum", help="wavelength (nm), relative intensity text file")
    parser.add_argument("--label", default=SPECTRUM_LABEL, help="legend label for the spectrum")
    parser.add_argument(
        "-f",
        "--filter",
        nargs="+",
        action="append",
        required=True,
        metavar="INSTRUMENT FILTER [LABEL]",
        help="instrument config, filter to plot and optional label",
    )
    parser.add_argument("-o", "--output", default="spectrum_filters.png")
    args = parser.parse_args()

    for f in args.filter:
        if len(f) not in (2, 3):
            parser.error(f"-f takes INSTRUMENT FILTER [LABEL], got {f}")
        f[0] = f[0].removesuffix(".toml")

    etcs = {}
    for config_name in (f[0] for f in args.filter):
        if config_name not in etcs:
            config = toml.loads(resources.files(data).joinpath(config_name + ".toml").read_text())
            etcs[config_name] = ETC(config)
    linestyles = {name: LINESTYLES[i % len(LINESTYLES)] for i, name in enumerate(etcs)}

    spec_wave, spec_flux = np.loadtxt(args.spectrum, unpack=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax_thru = ax.twinx()
    ax.set_zorder(ax_thru.get_zorder() + 1)
    ax.patch.set_visible(False)

    for i, (config_name, filtername, *label) in enumerate(args.filter):
        etc = etcs[config_name]
        label = label[0] if label else filter_label(etc.instrument.name, filtername)
        waves, thru = etc._efficiency_for_filter(filtername)
        waves = waves.to("nm").value
        thru = thru.value / 100.0
        colour = f"C{i}"
        ax_thru.plot(waves, thru, color=colour, ls=linestyles[config_name], alpha=0.8, lw=1.2)
        ax_thru.fill_between(waves, thru, color=colour, alpha=0.08)
        centre = np.sum(waves * thru) / np.sum(thru)
        ax_thru.text(centre, thru.max() + 0.04, label, color=colour, fontsize=14, ha="center")

    ax.plot(spec_wave, spec_flux, color="k", lw=0.8, label=args.label)
    for text, xytext, tick in LINES:
        if tick is None:
            ax.text(*xytext, text, ha="center", va="bottom", fontsize=9)
        else:
            ax.annotate(
                text,
                tick,
                xytext=xytext,
                ha="center",
                va="bottom",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=0, shrinkB=0),
            )

    ax.set_yscale("log")
    ax.set_xlim(300, 830)
    ax.set_ylim(0.5, 300)
    ax_thru.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative Intensity")
    ax_thru.set_ylabel("Throughput")
    ax.legend(loc="upper right", frameon=False)
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
