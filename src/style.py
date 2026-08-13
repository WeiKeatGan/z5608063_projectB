"""NoBearTrader figure design system: FT-inspired constants and helpers shared
across all Part B exhibits and the app. Carried over and extended from Part A."""
import matplotlib.pyplot as plt

FT_BG = "#FFF1E5"
FT_BLUE = "#0F5499"
FT_PINK = "#E6007E"
FT_GREY = "#66605C"
FT_PALETTE = [FT_BLUE, FT_PINK, "#990F3D", "#0D7680", "#262A33"]
FT_PALETTE_LONG = FT_PALETTE + [
    "#9E9E00", "#5C9E9E", "#7A3E9E", "#9E6B3E", "#3E7A9E", "#9E3E3E",
]

SAMPLE_PERIOD = "Out-of-sample: 2021 to 2023-12-29"
SOURCE = "Source: FINS3645 project dataset"


def new_ft_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FT_BG)
    ax.set_facecolor(FT_BG)
    return fig, ax


def apply_ft_style(ax, title, xlabel="", ylabel="", source=SOURCE, sample_period=SAMPLE_PERIOD):
    """Sentence-case title, clean spines, source+sample-period footer."""
    ax.set_title(title, fontsize=13, loc="left", color="#33302E")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.figure.text(0.01, 0.01, f"{source} | {sample_period}", fontsize=8, color=FT_GREY)


def savefig_close(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    plt.close(fig)
