import plotly.io as pio
from hmm_model import plot_hmm_plotly

fig = plot_hmm_plotly()

fig.write_json("data/chart.json")
