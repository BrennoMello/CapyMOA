from capymoa.drift.detectors import CUSUM, PageHinkley, ADWIN, SeqDrift1ChangeDetector
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math

# Setup plots
df = pd.read_csv("owid-covid-data.csv")

df = df.loc[df["location"] == "Brazil"]

df["weekly_new_deaths_mean"] = df["new_deaths"].rolling(window=7).mean()
df["weekly_new_cases_mean"] = df["new_cases"].rolling(window=7).mean()

# df["weekly_new_deaths_mean"] = df["weekly_new_deaths_mean"].fillna(method="ffill")
# df["weekly_new_cases_mean"] = df["weekly_new_cases_mean"].fillna(method="ffill")

df["weekly_new_deaths_mean"] = df["weekly_new_deaths_mean"].ffill()
df["weekly_new_cases_mean"] = df["weekly_new_cases_mean"].ffill()


# df.fillna(method="ffill", inplace=True)
df.ffill(inplace=True)
df = df[::7]

fig, ax_left = plt.subplots()
ax_right = ax_left.twinx()
ax_third = ax_left.twinx()

ax_left.plot(
    df["date"],
    df["people_vaccinated"],
    color="navy",
    linestyle="-",
    linewidth=1.00,
)

ax_left.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

ax_right.plot(
    df["date"],
    df["weekly_new_deaths_mean"],
    color="red",
    linestyle="-",
    linewidth=1.00,
)

ax_third.plot(
    df["date"],
    df["weekly_new_cases_mean"],
    color="lightgreen",
    linestyle="-",
    linewidth=1.00,
)

xticks = []
# rbf = RBF(**{"sigma": 0.01, "lambda_": 0.5, "alpha": 0.5, "delta": 1.0})
detector = "SEQ_DRIFT"
# cusum = CUSUM(delta=0.001, lambda_=50)
# page_hinkley = PageHinkley()
adwin = ADWIN()
# seq_drift = SeqDrift1ChangeDetector(delta=0.01, deltaWarning=0.1, block=100)
for index, row in df.iterrows():
    date = df["date"][index]
    value = row["weekly_new_deaths_mean"]

    # rbf.add_element(value)
    # cusum.add_element(value)
    # page_hinkley.add_element(value)
    # adwin.add_element(value)
    if not math.isnan(value):
        print(f"Date: {date}, Value: {value}")
        adwin.add_element(value)

    if adwin.detected_change():
        print(f"Concept drift detected at {date}")
        ax_left.axvline(date, color="orange", ls="--", linewidth=1.00)
        xticks.append(date)

ax_left.tick_params(axis="x", labelrotation=90)
ax_right.tick_params(axis="x", labelrotation=90)

ax_left.tick_params(axis="y", colors="navy")
ax_right.tick_params(axis="y", colors="red")

ax_third.tick_params(axis="y", colors="lightgreen")
ax_third.get_yaxis().set_visible(False)

ax_right.set_xticks(xticks)

for tick in ax_right.xaxis.get_minor_ticks():
    tick.label1.set_horizontalalignment("right")

plt.gcf().subplots_adjust(bottom=0.15)

custom_legends = [
    Line2D([0], [0], color="navy", linestyle="-", linewidth=2),
    Line2D([0], [0], color="lightgreen", ls="-", linewidth=2),
    Line2D([0], [0], color="red", ls="-", linewidth=2),
    Line2D([0], [0], color="orange", ls="--", linewidth=2),
]
legend = plt.legend(
    custom_legends,
    [
        "People Vaccinated",
        "Cases",
        "Deaths",
        f"Concept Drift (Deaths)",
    ],
    ncol=2,
    borderaxespad=0,
    loc="upper left",
)
legend.get_frame().set_alpha(None)

plt.suptitle("Vaccinated / Cases / Deaths")
plt.savefig(f"plots/vaccinated_cases_deaths_plot_{detector}.svg", dpi=300, bbox_inches="tight", format="svg")
plt.show()
