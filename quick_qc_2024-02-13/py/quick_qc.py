import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
import os
import numpy as np
import netCDF4 as nc
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.metrics import r2_score
import yaml
import argparse
import pathlib
import plotly.graph_objects as go
import glob

def read_config(): #Read in taget parameters and settings from config
    description = """\
    \033[1mExample Usage\033[0m
    (base) $ python quick_qc.py config.yaml

    \033[1mAlgorithm Description\033[0m
        1) Read in taget parameters and settings from config
        2) Load desired profiles to perform quick QC on into a Pandas DataFrame
        3) Perform quick QC on profiles
        4) Generate text based results from QC
        5) Make a summary refractivity plot of all profiles
        6) Make an interactive version of the above plot
    
    \033[1mInstructions\033[0m
        1) User identifies flights and aircraft that they want to perform quick QC on and modifies config file accordingly.
        2) User confirms nret folder contents for those profiles are reasonable.
        3) User runs quick_qc.
    
    \033[1mOutputs\033[0m
        - A text file containing a list of suspect profiles and some summary statistics
        - A static refractivity plot of all QCed profiles
        - An interactive refractivity plot, which can be opened in a web browser.
    
    \033[1mDependencies\033[0m
        - Run in (base) conda environment on claron for best results.
        - Uses 3rd party Python packages matplotlib, numPy, pandas, yaml, and plotly
    
    \033[1mSource\033[0m
        - Source path: /ags/projects/hiaper/code_noah/python/quick_qc/quick_qc_2024-02-13/py/quick_qc.py
        - Author: Noah Barton nbarton@ucsd.edu
        - Last Modified: 2024-02-14 nbarton, added documentation
        - Created:       2024-02-05 nbarton
    """
    formatter_class=argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)    
    parser.add_argument('config', nargs='?', help = 'name of configuration file to run the script')
    args = parser.parse_args()
    if args.config is None:
        print(description)
        exit()
    cfgpath = pathlib.Path(args.config)
    with open(cfgpath, 'r') as fcfg:            
        config = yaml.safe_load(fcfg)
    global fig_dir, start_dir, campaign, receiver, aircraft, A, B, r2_threshold, out_filename_prefix

    #TODO allow for only loading some flights
    #TODO allow switch for only good/all occultations
    results_dir = config['results_dir']
    start_dir = config['start_dir']
    campaign = config['campaign']
    receiver = config['receiver']
    aircraft = config['aircraft']
    if aircraft == 'G-IV':
        A = float(config['a'])
        B = float(config['b'])
        r2_threshold = float(config['r2_threshold'])
    out_filename_prefix = os.path.join(results_dir,aircraft.lower().replace('-',''))


def read_profiles(): #load desired profiles to perform quick QC on into a Pandas DataFrame
    global profiles, profiles_by_flight
    iops = sorted([d for d in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, d)) and receiver in d])
    profiles = pd.DataFrame(columns=['flight','occ','lat','lon','msl_alt','ref','log_ref', 'grad','d2n','msg'])
    profiles_by_flight = pd.DataFrame(columns=['flight','n_occs','n_ncs','n_good'])
    print('loading occultations from...')
    for iop in iops: #start by loading all profiles into Pandas df
        print(iop)
        occultations = sorted([occultation for occultation in os.listdir(os.path.join(start_dir,iop)) if '-' in occultation])
        n_occs = len(occultations)
        n_ncs = 0
        n_good = 0
        filepaths = os.listdir(os.path.join(start_dir,iop))
        for occultation in occultations:
            ncs = [filename for filename in os.listdir(os.path.join(start_dir,iop,occultation)) if '_nc' in filename]
            if len(ncs) != 1:
                continue
            n_ncs += 1
            if len(occultation) > 9:
                continue
            n_good += 1
            try:
                with nc.Dataset(os.path.join(start_dir,iop,occultation,ncs[0])) as ds:
                    lat = ds['Lat'][:] 
                    lon = ds['Lon'][:]
                    msl_alt = ds['MSL_alt'][:]
                    ref = ds['Ref'][:]
                    log_ref = np.log1p(ref)
                grad = np.diff(ref)/np.diff(msl_alt)
                d2n = np.abs(np.diff(grad))
                profiles.loc[len(profiles)] = [iop, occultation, lat, lon, msl_alt, ref, log_ref, grad, d2n, None]
                
            except:
                continue
        profiles_by_flight.loc[len(profiles_by_flight)] = [iop, n_occs, n_ncs, n_good]

a_s = np.empty(0)
b_s = np.empty(0)
r2_s = np.empty(0)

def quick_qc(profile): #defines quick QC profiles
    global a_s, b_s, r2_s, A, B
    msl_alt = profile.msl_alt
    ref = profile.ref
    log_ref = profile.log_ref
    grad = profile.grad
    d2n = profile.d2n
    idx = np.min([3, len(ref) - 1])
    msg = 'none'
    if np.any(ref < 50):
        if msg != 'none': 
            msg = 'multiple'
        else:
            msg = 'Refractivity too small (any < 50 N-Units)'
    if np.any(grad > 50):
        if msg != 'none': 
            msg = 'multiple'
        else:
            msg = 'Gradient too large (any > 50 N/km)'
    if (np.any(grad < -60)) | (np.any(grad > 10)):
        if msg != 'none': 
            msg = 'multiple'
        else:
            msg = 'Gradient out of bounds (any not in [-60, 10] N/km)'
    if np.any(d2n > 50):
        if msg != 'none': 
            msg = 'multiple'
        else:
            msg = 'Gradient has sharp spike (any 2 adjacent points differ by more than 50 N/km'
    if np.mean(grad) > -5:
        if msg != 'none': 
            msg = 'multiple'
        else:
            msg = 'Gradient is too large (mean gradient > -5 N/km)'

    if aircraft == 'G-IV': #additional, stricter criteria for G-IV
        '''
        [a, b] = np.polyfit(log_ref, msl_alt, 1)
        a_s = np.append(a_s, a)
        b_s = np.append(b_s, b)
        log_ref_predicted = A * log_ref + B
        r2 = r2_score(msl_alt, log_ref_predicted)
        r2_s = np.append(r2_s, r2)
        #ssr = (msl_alt_predicted - msl_alt) ^ 2
        if r2 < r2_threshold:
            if msg is not None: multiple_issues = True
            msg = 'Refractivity fit poor (R2 < '+str(r2_threshold)+')'
        '''
        if grad[0] < grad[idx - 1] - 10:
            if msg is not None: 
                msg = 'multiple'
            else:
                msg = 'Gradient at top too steep (top point is at least 10 N/km less than 3rd point)'
    return msg

#TODO Jennifer wants this to save prior results and just check to see if new flights exist
def run_qc(): #performs quick QC on loaded profiles and generates text results
    assert len(profiles) > 0, "no profiles were loaded, check nret for folder structure & atmPrf files"
    print('writing results')
    profiles['msg'] = profiles.apply(quick_qc, axis = 1)
    profiles_by_flight['p_good'] = profiles_by_flight.n_ncs / profiles_by_flight.n_occs * 100
    #TODO add case for when fig folder does not already exist, create
    profiles_by_flight.to_excel(out_filename_prefix+'_profiles_by_flight.xlsx')
    print(out_filename_prefix+'_results.txt')
    with open(out_filename_prefix+'_results.txt', 'w') as f:
        for index, row in profiles.iterrows():
            if index == 0:
                f.write(row.flight+'\n')
            else:
                if row.flight != profiles.iloc[index-1].flight:
                    f.write('-'*79+'\n\n')
                    f.write(row.flight+'\n')
                if row.msg != 'none':
                    f.write(row.occ + ': ' + row.msg + '\n')
        f.write('-'*79+'\n\n')
        f.write(profiles_by_flight.to_string()+'\n')
        '''
        if aircraft == 'G-IV':
            f.write('Mean fit coef A: '+str(np.mean(a_s))+'\n')
            f.write('Mean fit coef B: '+str(np.mean(b_s))+'\n')
            f.write('Mean R2 score: '+str(np.mean(r2_s))+'\n')
            f.write('\n')
        '''

def static_plot(): #make a refractivity plot of all profiles
    print('making static plot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    def plot(profile):
        ax2.plot(profile.grad, profile.msl_alt[:-1], c='r', linewidth=0.5)
        ax1.plot(profile.ref, profile.msl_alt, c='b', linewidth=0.5)
    profiles.apply(plot, axis = 1)

    if aircraft == 'G-IV':
        ax1.set_xlim([75, 350])
        ax1.set_ylim([0,15])
    if aircraft == 'C-130':
        ax1.set_xlim([50, 325])
        ax1.set_ylim([0,12])

    ax2.set_xlabel('Refractivity Gradient (N/km)')
    ax2.set_xlim([-150, 100])
    ax2.tick_params(axis='x', colors='red')
    ax2.xaxis.label.set_color('red')

    ax1.set_ylabel('MSL Altitude (km)') #Make red
    ax1.set_xlabel('Refractivity (N)')
    ax1.set_xticks(np.arange(50,350,50))
    ax1.tick_params(axis='x', colors='blue')
    ax1.set_xticks(np.arange(50,350,50))
    ax1.grid(which='major', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    ax1.set_yticks(np.arange(0,14,2))
    ax1.grid(which='major', axis='y', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    title = campaign+' multi-GNSS ARO in ' + str(profiles.flight.nunique()) + ' ' + aircraft + ' flights (G+R+E+C=' + str(profiles.shape[0]) + ')'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_filename_prefix+'_static_plot.png',dpi=500)

def interactive_plot(): #make an interactive refractivity plot of all profiles
    print('making interactive plot')
    fig = go.Figure()
    for index, profile in profiles.iterrows():
        fig.add_trace(go.Scatter(x=profile['grad'], y=profile['msl_alt'][:-1], mode='lines', line=dict(color='red', width=0.5), hoverinfo='text', hovertext=profile.flight+':'+profile['occ'], showlegend=False, xaxis='x2'))
        fig.add_trace(go.Scatter(x=profile['ref'], y=profile['msl_alt'], mode='lines', line=dict(color='blue', width=0.5), hoverinfo='text', hovertext=profile.flight+':'+profile['occ'], showlegend=False))

    if aircraft == 'G-IV':
        x_lims = [50, 350]
        y_lims = [0,15]
    if aircraft == 'C-130':
        x_lims = [75, 325]
        y_lims = [0,12]
    title = campaign+' multi-GNSS ARO in ' + str(profiles.flight.nunique()) + ' ' + aircraft + ' flights (G+R+E+C=' + str(profiles.shape[0]) + ')'

    fig.update_layout(
        title = title,
        xaxis=dict(
            title='Refractivity (N)',
            range=x_lims,
            tickvals=np.arange(x_lims[0], x_lims[1] + 25, 25),
            tickcolor='blue',
            gridcolor='gray',
            gridwidth=0.5,
            ticklen=10,
            title_font=dict(color='blue')
        ),
        yaxis=dict(
            title='MSL Altitude (km)',
            range=y_lims,
            tickvals=np.arange(y_lims[0], y_lims[1] + 2, 2),
            gridcolor='gray',
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Refractivity Gradient (N/km)',
            range=[-150, 100],
            tickcolor='red',
            overlaying='x',
            side='top',
            title_font=dict(color='red'),
            showgrid=False
        ),
        # margin=dict(l=50, r=50, t=80, b=50),
        # width=800,
        # height=600
    )
    fig.write_html(out_filename_prefix+'_interactive_plot.html')


read_config()
read_profiles()
print(profiles)
run_qc()
static_plot()
interactive_plot()