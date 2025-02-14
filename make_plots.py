##################################################
# imports

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from astropy.io import fits

##################################################
# inputs

make_spectrum_figure = False
make_SED_figure = False

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

##################################################
# spectral lines

speclines = {36.94: 'OH [?]',
             48.92: 'CS',
             86.22: 'H<sup>13</sup>CN',
             86.63: 'H<sup>13</sup>CO<sup>+</sup>',
             87.19: 'CCH',
             87.28: 'CCH',
             88.50: 'HCN',
             89.06: 'HCO<sup>+</sup>',
             90.53: 'HNC',
             97.84: 'CS',
             109.62: 'C<sup>18</sup>O',
             110.04: '<sup>13</sup>CO',
             112.98: 'CN',
             113.33: 'CN',
             115.10: 'CO',
             135.10: 'H<sub>2</sub>CS [?]',
             140.64: 'H<sub>2</sub>CO',
             145.39: 'H<sub>2</sub>CO [?]',
             146.76: 'CS',
             150.28: 'H<sub>2</sub>CO [?]'
             }

##################################################
##################################################
# generate spectrum plot

# Things to add:
# - sliding scale for boxcar averaging
# - can we get the toggle effects to be non-interfering?
# --> may need to use legend for this sort of thing, or check out plotly_relayout / eventdata

if make_spectrum_figure:

    ############################################
    # inputs

    bands = ['Band1a','Band1b','Band3a','Band3b','Band4a']
    seg_names = ['Band 1','Band 3','Band 4']
    seg_colors = ['red','darkorange','gold']

    field = 'centerfield'

    segment_separation = 0.1        # in GHz
    smoothing = 0.02                # in GHz

    ############################################
    # read in and organize spectra

    nu_individual = list()
    S_individual = list()
    for band in bands:
        tablename = './data/spectra/spectrum_'+band+'_'+field+'.txt'
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
    # plot

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
            title=dict(text='Flux density (mJy/beam)'),
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
##################################################
##################################################
# generate SED plot

# continuum info
freqs = [39.40633132032,46.05395407602,97.89582649825999,107.2456922428,137.6965151531]
bws = [8.78122053592,7.657156181928,30.72496072362,17.28863737626,26.320344557520002]
bmajs = [9.5243530273452,3.537628173828,3.03612208366392,1.71604394912736,2.1782915592192]
bmins = [5.9364953041068,3.01214885711652,2.85134267806992,1.0656483173370002,1.9247877597808802]
cont_peaks = [1.356348681485666,5.44685388853636,4.990724216951112,24.024687395294844,9.721306480139791]
cont_peak_errs = [0.001897262229692023,0.012416213403675666,0.012448246141759371,0.0833008904063904,0.04010009159700606]
alphas = [-0.59252626,-0.42468208,-0.36385986,-0.35234842,-0.3182819]
alpha_errs = [0.024346706,0.038815476,0.050575197,0.03417288,0.032523334]

if make_SED_figure:

    fig = go.Figure()

    for i in range(len(freqs)):

        #######################
        # SED point + spectral index

        beamsize = np.pi*bmajs[i]*bmins[i]

        x0 = freqs[i]
        xlo = x0 - (bws[i]/2.0)
        xhi = x0 + (bws[i]/2.0)
        xpoly = np.array([xlo,x0,xhi,xhi,x0,xlo,xlo])

        ahi = alphas[i] + alpha_errs[i]
        alo = alphas[i] - alpha_errs[i]

        y0 = cont_peaks[i]
        yhi_mid = y0 + cont_peak_errs[i]
        ylo_mid = y0 - cont_peak_errs[i]
        yhi_l = yhi_mid*(xlo/x0)**alo
        yhi_r = yhi_mid*(xhi/x0)**ahi
        ylo_l = ylo_mid*(xlo/x0)**ahi
        ylo_r = ylo_mid*(xhi/x0)**alo
        ypoly = beamsize*np.array([yhi_l,yhi_mid,yhi_r,ylo_r,ylo_mid,ylo_l,yhi_l])

        #######################
        # hover text

        textstr = ''
        textstr += 'ν<sub>0</sub> = '+str(np.round(x0,2))+' GHz'
        textstr += '<br />'
        textstr += 'BW = '+str(np.round(bws[i],2))+' GHz'
        textstr += '<br />'
        textstr += 'beam FWHM = '+str(np.round(bmajs[i],2)) + '×' + str(np.round(bmins[i],2))+' arcsec<sup>2</sup>'
        textstr += '<br />'
        textstr += '<br />'
        textstr += 'S<sub>ν</sub> = '+str(np.round(y0*beamsize,2))+' ± '+str(np.round(cont_peak_errs[i]*beamsize,2))+' mJy/beam'
        textstr += '<br />'
        textstr += 'α = '+str(np.round(alphas[i],3))+' ± '+str(np.round(alpha_errs[i],3))

        #######################
        # add to plot

        fig.add_trace(go.Scatter(x=xpoly, y=ypoly,
                      fill='toself',
                      mode='lines',
                      text=textstr,
                      hoveron = 'fills',
                      hoverinfo = 'text'))

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
            autorange=True,
            type='log'
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
            title=dict(text='Flux density (mJy/beam)'),
            autorange=True,
            type='log'
        ),
        autosize=True,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=75,r=25,b=25,t=25)
    )

    # output interactive html plot
    fig.write_html('plots/SED.html')

##################################################
# line cubes
##################################################

from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import ehtplot

c = 299792.458
vrec = 434.0
SNR_cut = 5.0

write_tables = {'OH_36.94GHz': False,
                'CS_48.92GHz': False,
                'H13CN_86.22GHz': False,
                'H13CO_86.63GHz': False,
                'CCH_87.19GHz': False,
                'CCH_87.28GHz': False,
                'HCN_88.50GHz': False,
                'HCO_89.06GHz': False,
                'HNC_90.53GHz': False,
                'CS_97.84GHz': False,
                'C18O_109.62GHz': False,
                '13CO_110.04GHz': False,
                'CN_112.98GHz': False,
                'CN_113.33GHz': False,
                'CO_115.10GHz': False,
                'H2CS_135.10GHz': False,
                'H2CO_140.64GHz': False,
                'H2CO_145.39GHz': False,
                'CS_146.76GHz': False,
                'H2CO_150.28GHz': False
                }

make_plots = {'OH_36.94GHz': False,
              'CS_48.92GHz': False,
              'H13CN_86.22GHz': False,
              'H13CO_86.63GHz': False,
              'CCH_87.19GHz': False,
              'CCH_87.28GHz': False,
              'HCN_88.50GHz': False,
              'HCO_89.06GHz': False,
              'HNC_90.53GHz': False,
              'CS_97.84GHz': False,
              'C18O_109.62GHz': False,
              '13CO_110.04GHz': False,
              'CN_112.98GHz': False,
              'CN_113.33GHz': False,
              'CO_115.10GHz': False,
              'H2CS_135.10GHz': False,
              'H2CO_140.64GHz': False,
              'H2CO_145.39GHz': False,
              'CS_146.76GHz': False,
              'H2CO_150.28GHz': False
              }

nu0_dict = {'OH_36.94GHz': 36.9944102,
            'CS_48.92GHz': 48.9909549,
            'H13CN_86.22GHz': 86.3399214,
            'H13CO_86.63GHz': 86.7542884,
            'CCH_87.19GHz': 87.316925,
            'CCH_87.28GHz': 87.407165,
            'HCN_88.50GHz': 88.6316023,
            'HCO_89.06GHz': 89.1885247,
            'HNC_90.53GHz': 90.663568,
            'CS_97.84GHz': 97.9809533,
            'C18O_109.62GHz': 109.7821734,
            '13CO_110.04GHz': 110.2013543,
            'CN_112.98GHz': 113.16879389,
            'CN_113.33GHz': 113.49438589,
            'CO_115.10GHz': 115.2712018,
            'H2CS_135.10GHz': 135.297811,
            'H2CO_140.64GHz': 140.8395197167,
            'H2CO_145.39GHz': 145.602949,
            'CS_146.76GHz': 146.9690287,
            'H2CO_150.28GHz': 150.498334
            }

rms_ind_start_dict = {'OH_36.94GHz': 120,
                      'CS_48.92GHz': 200,
                      'H13CN_86.22GHz': 100,
                      'H13CO_86.63GHz': 100,
                      'CCH_87.19GHz': 100,
                      'CCH_87.28GHz': 100,
                      'HCN_88.50GHz': 100,
                      'HCO_89.06GHz': 100,
                      'HNC_90.53GHz': 100,
                      'CS_97.84GHz': 100,
                      'C18O_109.62GHz': 100,
                      '13CO_110.04GHz': 100,
                      'CN_112.98GHz': 250,
                      'CN_113.33GHz': 250,
                      'CO_115.10GHz': 128,
                      'H2CS_135.10GHz': 100,
                      'H2CO_140.64GHz': 100,
                      'H2CO_145.39GHz': 100,
                      'CS_146.76GHz': 100,
                      'H2CO_150.28GHz': 100
                      }

xmax_dict = {'OH_36.94GHz': 27,
             'CS_48.92GHz': 20,
             'H13CN_86.22GHz': 27,
             'H13CO_86.63GHz': 27,
             'CCH_87.19GHz': 20,
             'CCH_87.28GHz': 20,
             'HCN_88.50GHz': 27,
             'HCO_89.06GHz': 27,
             'HNC_90.53GHz': 27,
             'CS_97.84GHz': 20,
             'C18O_109.62GHz': 27,
             '13CO_110.04GHz': 27,
             'CN_112.98GHz': 20,
             'CN_113.33GHz': 20,
             'CO_115.10GHz': 27,
             'H2CS_135.10GHz': 10,
             'H2CO_140.64GHz': 12,
             'H2CO_145.39GHz': 12,
             'CS_146.76GHz': 20,
             'H2CO_150.28GHz': 12
             }

def gaussian(x,A,s,v0,S0):
    return S0 + (A*np.exp(-0.5*((x-v0)**2.0)/(s**2.0)))

# def write_table(v,cube,tablename,rms_ind_start=200):
def write_table(linehere,rms_ind_start=200,bounds_amp=None):

    infile = './data/line_cubes/'+linehere+'.fits'
    infile_freq = './data/line_cubes/'+linehere+'_freq.txt'
    tablename = './data/line_cubes/gaussian_fits_'+linehere+'.txt'
    nu0 = nu0_dict[linehere]

    hdul = fits.open(infile)
    cube = hdul[0].data
    nu = np.loadtxt(infile_freq)
    nu /= (1.0e9)
    v = ((nu0 - nu)/nu)*c

    # get axes in arcseconds
    head = hdul[0].header
    pixnumx = np.linspace(0.0,head['NAXIS1']-1,head['NAXIS1'])
    pixnumy = np.linspace(0.0,head['NAXIS2']-1,head['NAXIS2'])
    x = head['CDELT1']*(pixnumx - head['CRPIX1']) + head['CRVAL1']
    y = head['CDELT2']*(pixnumy - head['CRPIX2']) + head['CRVAL2']
    x0 = head['CDELT1']*(np.ceil(cube.shape[1]/2) - head['CRPIX1']) + head['CRVAL1']
    y0 = head['CDELT2']*(np.ceil(cube.shape[2]/2) - head['CRPIX2']) + head['CRVAL2']
    x_arcsec = (x - x0)*3600.0
    y_arcsec = (y - y0)*3600.0

    A = np.zeros_like(cube[0,:,:])
    s = np.zeros_like(cube[0,:,:])
    v0 = np.zeros_like(cube[0,:,:])
    S0 = np.zeros_like(cube[0,:,:])
    A_err = np.zeros_like(cube[0,:,:])
    s_err = np.zeros_like(cube[0,:,:])
    v0_err = np.zeros_like(cube[0,:,:])
    S0_err = np.zeros_like(cube[0,:,:])

    sigma = np.ones_like(v)*np.nanstd(cube[:,rms_ind_start:,rms_ind_start:])

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):

            print(i,j)

            yhere = cube[:,i,j]

            if (np.isfinite(yhere).sum() != 0) & (np.nanmax(yhere) > 0):

                if bounds_amp is None:
                    p0 = [np.nanmax(cube[:,i,j]),100.0,434.0,0.0]
                    bounds = ([0.0,10.0,100.0,0.0],[2.0*np.nanmax(cube[:,i,j]),500.0,700.0,0.05])
                else:
                    p0 = [np.mean(bounds_amp),100.0,434.0,0.0]
                    bounds = ([bounds_amp[0],10.0,100.0,0.0],[bounds_amp[1],500.0,700.0,0.05])
                popt, pcov = curve_fit(gaussian,v,yhere,sigma=sigma,p0=p0,bounds=bounds,nan_policy='omit',maxfev=100000)

                A[i,j], s[i,j], v0[i,j], S0[i,j] = popt
                A_err[i,j] = np.sqrt(pcov[0,0])
                s_err[i,j] = np.sqrt(pcov[1,1])
                v0_err[i,j] = np.sqrt(pcov[2,2])
                S0_err[i,j] = np.sqrt(pcov[3,3])

            else:

                A[i,j] = np.nan
                s[i,j] = np.nan
                v0[i,j] = np.nan
                S0[i,j] = np.nan
                A_err[i,j] = np.nan
                s_err[i,j] = np.nan
                v0_err[i,j] = np.nan
                S0_err[i,j] = np.nan

    with open(tablename,'w') as f:
        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                strhere = ''
                strhere += str(i).ljust(6)
                strhere += str(j).ljust(6)
                strhere += str(A[i,j]).ljust(24)
                strhere += str(A_err[i,j]).ljust(24)
                strhere += str(s[i,j]).ljust(24)
                strhere += str(s_err[i,j]).ljust(24)
                strhere += str(v0[i,j]).ljust(24)
                strhere += str(v0_err[i,j]).ljust(24)
                strhere += str(S0[i,j]).ljust(24)
                strhere += str(S0_err[i,j]) + '\n'
                f.write(strhere)

def plot_triplet(tablename,plotname,SNR_cut=5.0,xmax=27.0,vmin_amp=None):

    infile = './data/line_cubes/'+linehere+'.fits'
    infile_freq = './data/line_cubes/'+linehere+'_freq.txt'
    nu0 = nu0_dict[linehere]

    hdul = fits.open(infile)
    cube = hdul[0].data
    nu = np.loadtxt(infile_freq)
    nu /= (1.0e9)
    v = ((nu0 - nu)/nu)*c

    # get axes in arcseconds
    head = hdul[0].header
    pixnumx = np.linspace(0.0,head['NAXIS1']-1,head['NAXIS1'])
    pixnumy = np.linspace(0.0,head['NAXIS2']-1,head['NAXIS2'])
    x = head['CDELT1']*(pixnumx - head['CRPIX1']) + head['CRVAL1']
    y = head['CDELT2']*(pixnumy - head['CRPIX2']) + head['CRVAL2']
    x0 = head['CDELT1']*(np.ceil(cube.shape[1]/2) - head['CRPIX1']) + head['CRVAL1']
    y0 = head['CDELT2']*(np.ceil(cube.shape[2]/2) - head['CRPIX2']) + head['CRVAL2']
    x_arcsec = (x - x0)*3600.0
    y_arcsec = (y - y0)*3600.0

    i, j, A, A_err, s, s_err, v0, v0_err, S0, S0_err = np.loadtxt(tablename,unpack=True)
    A = A.reshape(cube[0, :, :].shape)
    A_err = A_err.reshape(cube[0, :, :].shape)
    s = s.reshape(cube[0, :, :].shape)
    s_err = s_err.reshape(cube[0, :, :].shape)
    v0 = v0.reshape(cube[0, :, :].shape)
    v0_err = v0_err.reshape(cube[0, :, :].shape)
    S0 = S0.reshape(cube[0, :, :].shape)
    S0_err = S0_err.reshape(cube[0, :, :].shape)

    mask = (np.abs(A/A_err) < SNR_cut)
    mask |= (np.abs(v0 - vrec) > 250.0)

    A_plot = np.copy(A)
    s_plot = np.copy(s)
    v0_plot = np.copy(v0)
    A_plot[mask] = np.nan
    s_plot[mask] = np.nan
    v0_plot[mask] = np.nan

    xx,yy = np.meshgrid(x_arcsec,y_arcsec)

    # initialize figure
    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_axes([0.05,0.05,0.3,0.9])
    ax2 = fig.add_axes([0.375,0.05,0.3,0.9])
    ax3 = fig.add_axes([0.70,0.05,0.3,0.9])
    cax1 = fig.add_axes([0.05,0.96,0.3,0.03])
    cax2 = fig.add_axes([0.375,0.96,0.3,0.03])
    cax3 = fig.add_axes([0.70,0.96,0.3,0.03])

    ax1.set_facecolor('black')

    # plot peak line intensity
    if vmin_amp is None:
        vmin = 0.0
        vmax = np.nanmax(A_plot[(np.abs(xx) < xmax) & (np.abs(yy) < xmax)])
        cmap_amp = 'afmhot_us'
    else:
        vmin = vmin_amp
        vmax = -vmin_amp
        cmap_amp = 'seismic'
    cf = ax1.contourf(x_arcsec,y_arcsec,A_plot,255,cmap=cmap_amp,vmin=vmin,vmax=vmax)
    ax1.set_xlim(xmax,-xmax)
    ax1.set_ylim(-xmax,xmax)
    ax1.set_xlabel(r'$\Delta$RA (arcsec)')
    ax1.set_ylabel(r'$\Delta$DEC (arcsec)')
    ax1.grid(color='gray',linewidth=0.5,alpha=0.3,linestyle='--')

    # colorbar
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax), cmap=cmap_amp)
    plt.colorbar(mappable,cax=cax1,orientation='horizontal')
    cax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax1.set_title(r'Peak flux density (Jy beam$^{-1}$)',pad=35)

    # plot line velocity
    vmin = -200.0
    vmax = 200.0
    cf = ax2.contourf(x_arcsec,y_arcsec,v0_plot - vrec,255,cmap='seismic',vmin=vmin,vmax=vmax)
    ax2.set_xlim(xmax,-xmax)
    ax2.set_ylim(-xmax,xmax)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(color='gray',linewidth=0.5,alpha=0.3,linestyle='--')

    # colorbar
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin+vrec,vmax=vmax+vrec), cmap='seismic')
    plt.colorbar(mappable,cax=cax2,orientation='horizontal')
    cax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.set_title(r'Line velocity (km s$^{-1}$)',pad=35)

    # plot line dispersion
    vmin = 0.0
    vmax = 100.0
    cf = ax3.contourf(x_arcsec,y_arcsec,s_plot,255,cmap='Blues_us',vmin=vmin,vmax=vmax)
    ax3.set_xlim(xmax,-xmax)
    ax3.set_ylim(-xmax,xmax)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.grid(color='gray',linewidth=0.5,alpha=0.3,linestyle='--')

    # colorbar
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax), cmap='Blues_us')
    plt.colorbar(mappable,cax=cax3,orientation='horizontal')
    cax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax3.set_title(r'Line dispersion (km s$^{-1}$)',pad=35)

    plt.savefig(plotname,dpi=300,bbox_inches='tight')
    plt.close()

def make_plot(linehere,SNR_cut=5.0,xmax=27.0,vmin_amp=None):
    plotname = './plots/Circinus_'+linehere+'.png'
    tablename = './data/line_cubes/gaussian_fits_'+linehere+'.txt'
    plot_triplet(tablename,plotname,SNR_cut=SNR_cut,xmax=xmax,vmin_amp=vmin_amp)

##################################################
# make line plots (png)

for linehere in list(make_plots.keys()):
    if write_tables[linehere]:
        if linehere in ['OH_36.94GHz']:
            bounds_amp = [-0.01,0.01]
            write_table(linehere,rms_ind_start=rms_ind_start_dict[linehere],bounds_amp=bounds_amp)
        else:
            write_table(linehere,rms_ind_start=rms_ind_start_dict[linehere])
    if make_plots[linehere]:
        if linehere in ['OH_36.94GHz']:
            make_plot(linehere,SNR_cut=SNR_cut,xmax=xmax_dict[linehere],vmin_amp=-0.01)
        else:
            make_plot(linehere,SNR_cut=SNR_cut,xmax=xmax_dict[linehere])

##################################################


