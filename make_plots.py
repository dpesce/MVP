############################################
# imports

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

############################################
# inputs

bands = ['Band1a','Band1b','Band3a','Band3b','Band4a']
field = 'centerfield'

segment_separation = 0.1        # in GHz

############################################
# read in and organize spectra

nu_individual = list()
S_individual = list()
for band in bands:
    tablename = './data/spectrum_'+band+'_'+field+'.txt'
    nu_here, S_here = np.loadtxt(tablename,unpack=True)
    nu_individual.append(nu_here / (1.0e9))
    S_individual.append(S_here * 1000.0)
nu = np.concatenate(nu_individual)
S = np.concatenate(S_individual)
ind = np.argsort(nu)
nu = nu[ind]
S = S[ind]

# determine segmentations
nu_segs = list()
S_segs = list()
ind = np.diff(nu) > segment_separation
if ind.sum() > 0:
    nu_segs.append(nu[:np.where(ind)[0][0]+1])
    S_segs.append(S[:np.where(ind)[0][0]+1])
    for i in range(ind.sum()-1):
        nu_segs.append(nu[np.where(ind)[0][0]+1:np.where(ind)[0][1]+1])
        S_segs.append(S[np.where(ind)[0][0]+1:np.where(ind)[0][1]+1])
    nu_segs.append(nu[np.where(ind)[0][-1]+1:])
    S_segs.append(S[np.where(ind)[0][-1]+1:])

# create dataframe
df = pd.DataFrame({
    'nu': nu,
    'S': S
    })

############################################
# generate spectrum plot

title = 'Continuum-subtracted spectrum of Circinus Galaxy'

fig = go.Figure()

for i in range(len(nu_segs)):
    fig.add_trace(go.Scatter(x=nu_segs[i], y=S_segs[i], mode='lines',
        line=dict(color='black', width=2),
        hovertemplate='ν = %{x} GHz<br />S = %{y} mJy',
        connectgaps=True,
    ))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        zeroline=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        gridwidth=1,
        gridcolor='lightgray',
        zerolinewidth=1,
        zerolinecolor='lightgray',
        griddash='dot',
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
            ),
        title=dict(text='Frequency (GHz)')
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        zeroline=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        gridwidth=1,
        gridcolor='lightgray',
        zerolinewidth=1,
        zerolinecolor='lightgray',
        griddash='dot',
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
            ),
        title=dict(text='Flux density (mJy)')
    ),
    autosize=True,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=75,r=25,b=25,t=25)
)

# output interactive html plot
fig.write_html('plots/spectrum.html')

############################################

