############################################
# imports

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

############################################
# inputs

bands = ['Band1a','Band1b','Band3a','Band3b','Band4a']
seg_names = ['Band 1','Band 3','Band 4']
seg_colors = ['red','darkorange','gold']

field = 'centerfield'

segment_separation = 0.1        # in GHz
smoothing = 0.02                # in GHz

##################################################
# boxcar averaging function

def boxcar(x,y,window):
    # x and window both specified in GHz

    hw = window / 2.0
    av = np.zeros_like(x)

    for i in range(len(y)):
        if ((x[i]-x[0]) < hw):
            ind = ((x - x[0]) < (x[i]-x[0]+hw))
            av[i] = np.nanmean(y[ind])
        elif ((x[i]-x[0]) >= hw) & ((x[i]-x[0]) < ((x[-1]-x[0])-hw)):
            ind = ((x - x[0]) > (x[i]-x[0]-hw)) & ((x - x[0]) < (x[i]-x[0]+hw))
            av[i] = np.nanmean(y[ind])
        else:
            ind = ((x - x[0]) > (x[i]-x[0]-hw))
            av[i] = np.nanmean(y[ind])

    return av

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

# atmospheric transmission
f_atm, tau_atm, _ = np.loadtxt('./data/atmospheric_opacity.txt',unpack=True)
trans_atm = np.exp(-tau_atm)

############################################
# spectral lines

speclines = {36.95: 'OH  [?]',
             48.92: 'CS',
             87.2: 'CH<sub>3</sub>OH [?]',
             88.5: 'HCN',
             89.05: 'HCO<sup>+</sup> [?]',
             90.55: 'CH<sub>3</sub>OH [?]',
             93.05: 'CH<sub>3</sub>OH [?]',
             96.60: 'CH<sub>3</sub>OH [?]',
             97.80: 'CS',
             109.6: 'C<sup>18</sup>O',
             110.0: '<sup>13</sup>CO',
             113.0: 'CH<sub>3</sub>OH [?]',
             113.3: 'CN',
             115.1: 'CO',
             135.1: 'CH<sub>3</sub>OD [?]',
             140.6: 'H<sub>2</sub>CO [?]',
             144.9: 'CH<sub>3</sub>OH [?]',
             145.4: 'H<sub>2</sub>CO [?]',
             146.75: 'CS [?]',
             150.3: 'H<sub>2</sub>CO [?]',
             150.6: 'SO<sub>2</sub> [?]'
             }

############################################
# generate spectrum plot

# Things to add:
# - sliding scale for boxcar averaging
# - can we get the toggle effects to be non-interfering?
# --> may need to use legend for this sort of thing, or check out plotly_relayout / eventdata

# fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True, 'r':-0.06}]])

# plot spectra
maxS = 0.0
for i in range(len(nu_segs)):
    if smoothing > 0:
        yhere = boxcar(nu_segs[i],S_segs[i],smoothing)
    else:
        yhere = S_segs[i]
    fig.add_trace(go.Scatter(x=nu_segs[i], y=yhere, mode='lines',
        line=dict(color='black', width=2),
        hovertemplate='ν = %{x} GHz<br />S = %{y} mJy',
        connectgaps=True,
        name=seg_names[i],
        visible=True
        ),
        secondary_y=False)
    if np.nanmax(yhere) > maxS:
        maxS = np.nanmax(yhere)

# plot atmospheric transmission
fig.add_trace(go.Scatter(x=f_atm, y=trans_atm, mode='lines', opacity=0.3,
        line=dict(color='red', width=1),
        connectgaps=True,
        name='Atmospheric transmission',
        hoverinfo='skip',
        visible=False
    ),
    secondary_y=True)

# plot individual SBs
for i in range(len(nu_individual)):
    if smoothing > 0:
        yhere = boxcar(nu_individual[i],S_individual[i],smoothing)
    else:
        yhere = S_individual[i]
    fig.add_trace(go.Scatter(x=nu_individual[i], y=yhere, mode='lines', opacity=0.5,
        line=dict(width=2),
        connectgaps=True,
        hoverinfo='skip',
        visible=False
        ),
        secondary_y=False)

# plot spectral lines
keys = list(speclines.keys())
for i, key in enumerate(keys):
    fig.add_trace(go.Scatter(x=[key,key], y=[0.97*maxS,1.03*maxS], mode='lines', opacity=1.0,
        line=dict(width=2),
        connectgaps=True,
        hovertemplate=speclines[key],
        visible=False,
        name=''
        ),
        secondary_y=False)

# axis properties
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
        title=dict(text='Frequency (GHz)'),
        autorange=False,
        range=[min(nu), max(nu)]
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
        title=dict(text='Flux density (mJy)'),
        autorange=False,
        range=[-10, 1.10*maxS]
    ),
    yaxis2=dict(
        showgrid=False,
        overlaying='y',
        side='right',
        rangemode='normal',
        range=[0, 1],
        showticklabels=False,
        title=''
    ),
    autosize=True,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=75,r=100,b=25,t=25)
)

# buttons
fig.update_layout(
    updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    active=0,
                    x=1.056,
                    y=1.00,
                    buttons=list([
                            dict(label='Reset',
                                method='update',
                                args=[{'visible': [True]*len(nu_segs) + [False] + [False]*len(nu_individual) + [False]*len(keys)},
                                      {'annotations': []}])
                            ]),
                    ),
                dict(
                    type='buttons',
                    direction='right',
                    active=0,
                    x=1.09,
                    y=0.90,
                    buttons=list([
                            dict(label='Atmospheric<br />transmission',
                                method='update',
                                args=[{'visible': [True]*len(nu_segs) + [True] + [False]*len(nu_individual) + [False]*len(keys)},
                                      {'annotations': []}])
                            ]),
                    ),
                dict(
                    type='buttons',
                    direction='right',
                    active=0,
                    x=1.0765,
                    y=0.80,
                    buttons=list([
                            dict(label='Individual<br />SBs',
                                method='update',
                                args=[{'visible': [False]*len(nu_segs) + [False] + [True]*len(nu_individual) + [False]*len(keys)},
                                      {'annotations': []}])
                            ]),
                    ),
                dict(
                    type='buttons',
                    direction='right',
                    active=0,
                    x=1.091,
                    y=0.70,
                    buttons=list([
                            dict(label='Line<br />identification',
                                method='update',
                                args=[{'visible': [True]*len(nu_segs) + [False] + [False]*len(nu_individual) + [True]*len(keys)},
                                      {'annotations': []}])
                            ]),
                    ),
                ])

# output interactive html plot
fig.write_html('plots/spectrum.html')

############################################

