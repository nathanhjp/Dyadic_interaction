# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:23:48 2024

@author: Nathan Pieterse
"""

#%% --- Setup
import pandas as pd
import os 
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from scipy.stats import chisquare



def get_filenames_in_folder(folder_path):
    # List all files and directories in the folder
    all_items = os.listdir(folder_path)
    
    # Filter out directories, keeping only files
    filenames = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    
    return filenames

# For the deepsqueak files
folder_path_usvs = r"C:\Users\Kesselslab\Desktop\M1_happenings\combinations_with_usvs\USV-Data-analyse"
filenames_usvs = get_filenames_in_folder(folder_path_usvs)
num_files = len(filenames_usvs)
filenames_usvs = [filenames_usvs[0], filenames_usvs[2], filenames_usvs[1]] # This one is specifically for the files i analysed. Probably comment this line
paths_usvs_zip = zip([folder_path_usvs] * num_files, filenames_usvs)
paths_usvs = list(paths_usvs_zip)

# For the deepof files
folder_path_deepof = r"C:\Users\Kesselslab\Desktop\M1_happenings\combinations_with_usvs\post_deepof_combined_with_usvs"
filenames_deepof = get_filenames_in_folder(folder_path_deepof)
if len(filenames_deepof) != num_files: raise Exception("You have provided non-equal number of deepof and deeplabcut files")
filenames_deepof = [filenames_deepof[0], filenames_deepof[2], filenames_deepof[1]] # This one is specifically for the files i analysed. Probably comment this line
paths_deepof_zip = zip([folder_path_deepof] * num_files, filenames_deepof)
paths_deepof = list(paths_deepof_zip)

# For the deeplabcut files
folder_path_dlc = r"C:\Users\Kesselslab\Desktop\M1_happenings\combinations_with_usvs\Deeplabcut_for_combination_with_usvs"
filenames_dlc = get_filenames_in_folder(folder_path_dlc)
if len(filenames_dlc) != num_files: raise Exception("You have provided non-equal number of deepsqueak and deeplabcut files")
filenames_dlc = [filenames_dlc[0], filenames_dlc[2], filenames_dlc[1]] # This one is specifically for the files i analysed. Probably comment this line
paths_dlc_zip = zip([folder_path_dlc] * num_files, filenames_dlc)
paths_dlc = list(paths_dlc_zip)


# Make sure you have a good synchronisation and the start times match
starts_trimmed_vid = [99.20, 147.28, 167.22] 
starts_mic_on_vid = [10.13, 7.20, 12.03] # time in seconds on stopwatch visible in video
fps = 30


# To match those in the trimmed video
recording_latencies = [start_trimmed_vid - start_mic_on_vid for (start_trimmed_vid, start_mic_on_vid) in zip(starts_trimmed_vid, starts_mic_on_vid)]

# Zip containing the filenames to load in and the recording latencies
filenames_per_recording_zip = zip(paths_usvs, paths_deepof, paths_dlc, recording_latencies)
filenames_per_recording = list(filenames_per_recording_zip)

folder = r"your_folder"



#%% --- Average distance against number of USVs plot

def calculate_distances(df_coords) -> list[float]:
    """Calculates the distances (in cm) between the centers of 2 mice"""

    # Makes lists of the relevant names in each level of the dataframe
    scorer = df_coords.columns[0][0]
    inds = pd.Series(ind[1] for ind in df_coords.columns).drop_duplicates().tolist()
    # bodyparts = pd.Series(ind[2] for ind in df_coords.columns).drop_duplicates().tolist()
    # coords = ['x', 'y', 'likelihood']
    
    # Save the coordinates to a list for easier processing
    m1_x = df_coords[scorer][inds[0]]['Center']['x'].tolist()
    m1_y = df_coords[scorer][inds[0]]['Center']['y'].tolist()
    
    m2_x = df_coords[scorer][inds[1]]['Center']['x'].tolist()
    m2_y = df_coords[scorer][inds[1]]['Center']['y'].tolist()
    
    
    # Calculate distance between the two animals at each frame:
    distm1m2 = []
    for ind,j in enumerate(m1_x):
        dist = distance.euclidean([m1_x[ind], m1_y[ind]], [m2_x[ind], m2_y[ind]])
        
        # Convert arbitrary units to cm (approximately):
        # Approximately 700 pixels for 40 cm (700/40=17.5)
        distm1m2.append(dist / 17.5)
        
    # color = "red"
    
    return distm1m2

def calculate_av_distances_during_usv(df_USV, distm1m2, recording_latency) -> list[float]:
    """Calculates the average distance (cm) per USV for all USVs. Returns a list of floats with those average distances"""
    av_dist_mice = []
    for USV, j in enumerate(df_USV["Call Length (s)"]):

        # IMPORTANT: Since the microphone starts before the analysed video,
        # the latency should be subtracted from the microphone time
        start_frame = int(math.floor((df_USV["Begin Time (s)"][USV] - recording_latency) * 30))
        end_frame = int(math.ceil((df_USV["End Time (s)"][USV] - recording_latency)* 30))
        
        # This is to make sure that a relevant distance is always found
        # Importantly, a nan will also be inserted if the time does not contain a distance
        if start_frame < end_frame: av_dist = np.mean(distm1m2[start_frame:end_frame])
        elif start_frame == end_frame: av_dist = distm1m2[start_frame]
        else: av_dist = np.nan
       
        av_dist_mice.append(av_dist)
    return av_dist_mice


#%% --- Figure 1: Overview of the correlation of distance between mice to the number of USVs produced

# Makes a subplot of 3 recordings with density of calls per distances, density of distances, 
# and the density of calls per distances normalized against density of distances
fig, axs = plt.subplots(3,3)

for count, recording in enumerate(filenames_per_recording):

    """
    # First off, we find the start point of the USV, and which frame this is in df_coords
    
    # it just saves frame numbers, so we have to use the start times and trust that the frames/second  = 30.
    # This of course tracks when there are few to no image errors
    
    # start_time of trimmed video minus start time of audio recording
    # This latency is the amount of time to add to times in the audio recording,
    # to match those in the trimmed video
    # recording_latency = start_trimmed_vid - start_mic_on_vid 
    """

    # Read in the relevant dataframes
    ((usv_folder, usv_file), (deepof_folder, deepof_file), (dlc_folder, dlc_file), recording_latency) = recording 
    df_coords = pd.read_hdf(os.path.join(dlc_folder,dlc_file))
    df_USV = pd.read_excel(os.path.join(usv_folder, usv_file))

    # The relevant variables for navigating df_coords
    scorer = df_coords.columns[0][0]
    inds = pd.Series(ind[1] for ind in df_coords.columns).drop_duplicates().tolist()
    bodyparts = pd.Series(ind[2] for ind in df_coords.columns).drop_duplicates().tolist()
    coords = ['x', 'y', 'likelihood']

    distm1m2 = calculate_distances(df_coords)
    av_distances_usv = calculate_av_distances_during_usv(df_USV, distm1m2, recording_latency)

#%% Subplot 1: Number of USVs at each distance

    fig.suptitle("Overview of USVs and the correlation with distance between mice")
    
    # Non-normalized plot
    ax0 = axs[0, count]

    xbins = np.arange(0,46,1)
    title =  "USVs per distance"
    if count == 1: ax0.set_title("Density of USVs at distances")
    ax0.set_xticks(np.arange(0,50,5), np.arange(0,50,5))
    counts, bins, patches = ax0.hist(av_distances_usv, bins=xbins, color = "red", alpha=0.5)
    ax0.vlines(8, 0, 1, linestyles='dashed', colors = "dimgrey")
    if count == 0: ax0.set_ylabel('Proportion of calls')
    ax0.set_xlabel('Distance between two mice (cm)')
    ax0.set_ylim([0,max(counts)+0.005])
    ax0.set_xlim([0,45])
    

#%% Subplot 2: Time spent at each distance        
    
    distm1m2 = calculate_distances(df_coords)   
    # Make a distance distribution
    overall_dist, _ = np.histogram(distm1m2, bins=xbins, density = True)

    ax1 = axs[1, count]
    
    counts, bins, patches = ax1.hist(distm1m2, bins=xbins, density = True, color = "red", alpha = 0.5) # weights = np.ones(len(distm1m2)) / 30
    ax1.set_xlabel("Distance (cm)")
    if count == 0: ax1.set_ylabel("Proportion of time")
    ax1.set_xticks(np.arange(0,50,5), np.arange(0,50,5))
    ax1.vlines(8, 0, 1, linestyles='dashed', colors = "dimgrey")
    ax1.set_ylim([0,max(counts)+0.005])
    ax1.set_xlim([0,45])
    if count == 1: ax1.set_title("Density of time spent at distance")

#%% Subplot 3: Expected density of USVs emitted over distance against observed densities
    
    av_dist_mice = calculate_av_distances_during_usv(df_USV, distm1m2, recording_latency)
    
    # Make a distribution of USVs per distance
    USV_dist, _ = np.histogram(av_dist_mice, bins=xbins, density = True)
    
    # Normalize event distances by overall distances
    epsilon = 1e-10
    normalized_USV_dist = USV_dist / (overall_dist + epsilon)

    # Initiation of relevant distance lists
    distances_series = pd.Series(distm1m2)
    event_distances = pd.Series(av_dist_mice)
    sorted_distances = np.sort(distm1m2) 

    # Compute the Kernel Density Estimate (KDE) to estimate the expected density of USVs at each distance 
    # (given the assumption of no effect of distance on the likelihood of USV production)
    kde = gaussian_kde(distances_series)
    x_values = np.linspace(distances_series.min(), distances_series.max(), 1000) # enough points to generate a smooth line
    y_values = kde(x_values) # the corresponding y-values
    
    ax = axs[2, count]
    ax.plot(x_values, y_values, label="All Distances (Expected)", color = "black", linestyle = "dashed") 
    ax.hist(event_distances, bins=np.arange(0, 46, 1), alpha=0.5, color='red', label="Event Distances (Observed)", density=True)
    ax.vlines(8,0,0.40, linestyles = "dashed", color = 'dimgrey')
    ax.set_ylim(0,0.35)

    # Add labels and legend
    ax.set_title(f"{dlc_file[8:17]}")
    ax.set_xlabel("Distance")
    if count == 0: ax.set_ylabel("Density")
        
#%% Inset subplot: Cumulative density

    # Function to generate a cdf
    def custom_empirical_cdf(x: np.Array) -> np.Array:
        # Use np.searchsorted to find where x would fit in the sorted_distances
        idx = np.searchsorted(sorted_distances, x, side='right') - 1
        # idx = np.clip(idx, 0, n-1)  # Clip indices to handle out-of-bound values
        return empirical_cdf[idx]

    # Initiation of relevant variables
    n = len(sorted_distances)
    empirical_cdf = np.arange(1, n+1) / n  # Values from 1/n to 1
    
        
    # Step 1: Sort the distances
    sorted_event_distances = np.sort(event_distances)
        
     # Step 2: Calculate the cumulative distribution (CDF)
    n = len(sorted_event_distances)  # Total number of distances
    cumulative_counts = np.arange(1, n + 1)  # Cumulative count of sorted distances
    cdf = cumulative_counts / n  # Normalize by dividing by the total number of distances

    inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")
    inset_ax.step(sorted_event_distances, cdf, where="post", label="CDF of Distances", alpha=0.5, color="red", linewidth=2)
    inset_ax.step(sorted_distances, custom_empirical_cdf(sorted_distances), color = "black", linestyle = "dashed")
    inset_ax.set_xlim(0,45)
    inset_ax.set_xticks(np.arange(0,46,15), np.arange(0,46,15))
    inset_ax.set_xlabel("Distance (cm)")
    inset_ax.set_ylabel("Cumulative Density of USVs")
    inset_ax.set_ylim(0,1.1)


#%% Statistics and permutation test

    # Define bins for distances
    bins = np.arange(0, 46, 5)
    distance_binned = pd.cut(distances_series, bins=bins)
    
    # Count observed events and time spent per distance bin
    observed_events_per_bin = event_distances.groupby(distance_binned).sum()
    time_spent_per_bin = distances_series.groupby(distance_binned).count()
    
    # Compute expected events assuming that ONLY the time spent at each distance influences the USV rate at that distance
    total_events = observed_events_per_bin.sum()
    expected_events_per_bin = (time_spent_per_bin / time_spent_per_bin.sum()) * total_events
    
    # Compute chi-squared statistic for observed data
    observed_chi2_stat, _ = chisquare(f_obs=observed_events_per_bin, f_exp=expected_events_per_bin)
    

    # Perform permutation test
    n_permutations = 10000
    permuted_chi2_stats = np.zeros(n_permutations)
    
    for count in range(n_permutations):
        # Permute event series
        permuted_events = pd.Series(np.random.permutation(event_distances))
        permuted_events_per_bin = permuted_events.groupby(distance_binned).sum()
    
        # Compute chi-squared statistic for permuted data (in other words, how far the number of USVs is from  expected)
        permuted_chi2_stat, _ = chisquare(f_obs=permuted_events_per_bin, f_exp=expected_events_per_bin)
        permuted_chi2_stats[count] = permuted_chi2_stat
    
    # Compute p-value -> Can be zero if NO permuted scenarios were more extreme than the observed value!
    empirical_p_value = np.mean(permuted_chi2_stats >= observed_chi2_stat)
    
    # Output results
    print(f"Observed chi-squared statistic: {observed_chi2_stat}")
    print(f"Empirical p-value from permutation test: {empirical_p_value}")
    print('Can be zero if NO permuted scenarios were more extreme than the observed value!')
        
    

# Share y-axis for each row
for ax in axs[0, :]:
    ax.sharey(axs[0, 0])  # Share y-axis in the first row

for ax in axs[1, :]:
    ax.sharey(axs[1, 0])  # Share y-axis in the second row

plt.subplots_adjust(hspace=0.5)  # Increase vertical space

fig.show()

fig.savefig(os.path.join(folder, 'summary_calls_against_distance_two_mice_all_files.pdf'))
fig.savefig(os.path.join(folder, 'summary_calls_against_distance_two_mice_all_files.jpg'))



