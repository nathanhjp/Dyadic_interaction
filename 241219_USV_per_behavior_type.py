# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:05:48 2024

@author: Nathan Pieterse
"""


#%% --- Setup
import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


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
starts_mic_on_vid = [10.13, 7.20, 12.03]
fps = 30


# To match those in the trimmed video
recording_latencies = [start_trimmed_vid - start_mic_on_vid for (start_trimmed_vid, start_mic_on_vid) in zip(starts_trimmed_vid, starts_mic_on_vid)]

# Zip containing the filenames to load in and the recording latencies
filenames_per_recording_zip = zip(paths_usvs, paths_deepof, paths_dlc, recording_latencies)
filenames_per_recording = list(filenames_per_recording_zip)

folder = r"C:\Users\Kesselslab\Desktop\M1_happenings\combinations_with_usvs\plots"



#%% --- New behavior calculation method (non-overlapping, 1 per individual per frame)

# Define a function to choose the highest-priority event for a given row (person)
def choose_event(row, event_cols):
    # Priority: Event 1 > Event 2 > Event 3
    for event in event_cols:
        if row[event] == 1:
            return event  # Return the column name of the highest-priority event
    return None  # No event occurring at this time


def choose_empty(row, event_cols):
    for event in event_cols:
        if row[event] == 1:
            return None
    return 0

def clean_deepof_df(df_deepof):

    # This part just determines the order in which the behaviors will take priority over eachother for each of the individuals
    
    priority_per_beh = ['nose2nose', 'climb', 'nose2tail', 'nose2body', 'follow', 'huddle']
    
    inds = ['individual1', 'individual2']
    # Make the list that will contain the priorities
    priority_cols_ind_1 = []
    for beh in priority_per_beh:
        for beh_ind in df_deepof.columns: # choose the key to the specific occurence
            
            # checks whether both individual and behavior are in the key
            if beh in beh_ind and inds[0] in beh_ind: 
                    
                # Checks whether ind1 is mentioned first to determine if it's actually ind1 doing the behavior    
                if beh_ind.find(inds[0]) < beh_ind.find(inds[1]) or beh_ind.find(inds[1]) == -1:
                        priority_cols_ind_1.append(beh_ind)
    
    # same notes as last one
    priority_cols_ind_2 = []
    for beh in priority_per_beh:
        for beh_ind in df_deepof.columns:
            
            if beh in beh_ind and inds[1] in beh_ind: 
                
                if beh_ind.find(inds[0]) > beh_ind.find(inds[1]) or beh == "nose2nose" or beh_ind.find(inds[0]) == -1:
                    priority_cols_ind_2.append(beh_ind)
    
    
    
    # Create new columns for the cleaned-up behavior
    df_deepof['cleaned_event_ind_1'] = df_deepof.apply(lambda row: choose_event(row, priority_cols_ind_1), axis=1)
    df_deepof['cleaned_event_ind_2'] = df_deepof.apply(lambda row: choose_event(row, priority_cols_ind_2), axis=1)
    
    
    
    # Sanity check: is the same number of most prioritized behavior (nose2nose) present in the cleaned version?
    count_per_element = df_deepof['cleaned_event_ind_1'].str.count('nose2nose')
    
    # To get the total count of occurrences in the entire Series
    total_count = count_per_element.sum()
    
    
    # THE ANSWER IS YES!
    print(df_deepof[priority_cols_ind_1[0]].sum() == total_count)
    
    
    
    
    # Sanity check: is there as many empty rows for each individual as there is rows that don't get assigned a behavior?
    
    df_deepof['empty_event_ind_1'] = df_deepof.apply(lambda row: choose_empty(row, priority_cols_ind_1), axis=1) # Selects the rows where all the behaviors for ind1 are 0
    df_deepof['empty_event_ind_2'] = df_deepof.apply(lambda row: choose_empty(row, priority_cols_ind_2), axis=1)
    
    # checks whether there were as many non-empty ones, as ones that were labelled using previous logic, which turns out to be true!
    print(df_deepof['empty_event_ind_1'].isna().sum() == len(df_deepof['cleaned_event_ind_1']) - df_deepof['cleaned_event_ind_1'].isna().sum())
    
    
    return df_deepof[['cleaned_event_ind_1', 'cleaned_event_ind_2']]
    

# Print the cleaned-up DataFrame
# print(df[['time', 'cleaned_event_person_1', 'cleaned_event_person_2']])


#%% --- Number of USVs against type of behavior 


def get_event_times_new(time_series, beh_str) -> list[tuple[int]]:
    """
    Given the 
    """
    event_times = []
    start_time = None
    
    whole_beh_name = [beh for beh in time_series.unique() if (not isinstance(beh, type(None)) and beh_str in beh)]
    
    if whole_beh_name: 
        whole_beh_name = whole_beh_name[0]

        for i in range(1, len(time_series)):
            if time_series.iloc[i-1] != whole_beh_name and time_series.iloc[i] == whole_beh_name:
                # Start of event
                start_time = i

            elif time_series.iloc[i-1] == whole_beh_name and time_series.iloc[i] != whole_beh_name:
                # End of event
                if start_time is not None:
                    event_times.append((start_time, i - 1))
                    start_time = None

    else:
        for i in range(1, len(time_series)):
            if not isinstance(time_series.iloc[i-1], type(None)) and isinstance(time_series.iloc[i], type(None)):
                # Start of event
                start_time = i

            elif isinstance(time_series.iloc[i-1], type(None)) and not isinstance(time_series.iloc[i], type(None)):
                # End of event
                if start_time is not None:
                    event_times.append((start_time, i - 1))
                    start_time = None

                    
    # If the event runs until the end of the time series
    if start_time is not None:
        event_times.append((start_time, len(time_series) - 1))

    return event_times


def get_frames_and_types_usvs(df_USV, recording_latency) -> tuple[list[tuple[int]], list[str]]:
    """Returns the exact start and end frames from the start of the video at which the USVs occur"""
    start_frames = [int(math.floor(USV - recording_latency) * 30) for USV in df_USV["Begin Time (s)"]]
    end_frames = [int(math.ceil((USV - recording_latency)* 30)) for USV in df_USV["End Time (s)"]]
    list_call_types = [USV for USV in df_USV['Label']]
    
    USV_start_ends = list(zip(start_frames, end_frames))
    
    return USV_start_ends, list_call_types


#%% --- Normalized version (new_version)

def merge_behs(beh_start_ends:list[list[tuple]]) -> list[tuple[int]]:
    """
    Merges 2 lists of occurences (start, end) (of a behavior) into 1 list of occurences (start, end)
    """
    
    # Combine the lists into 1 list
    combined_behs = beh_start_ends[0] + beh_start_ends[1] 
    
    # Sort the tuples based on the start frames
    combined_behs.sort(key=lambda x: x[0]) 
    
    # Create a list to be filled with the merged times
    merged = []
    for beh_start_end in combined_behs:
        # If merged is empty or the current behavior period does not overlap with the previous one, add it to merged
        if not merged or merged[-1][1] < beh_start_end[0]:
            merged.append(beh_start_end)
        else:
            # Otherwise, merge the current interval with the previous one
            merged[-1] = (merged[-1][0], max(merged[-1][1], beh_start_end[1]))
    
    return merged


def get_duration_behavior(beh_start_ends: list[tuple[int]], frames_per_second: int = 30 ) -> float:
    """ 
    Gets the total time (s) spent in each behavior.
    """
    num_frames_duration_behavior = 0
    for start,end in beh_start_ends:
        num_frames_duration_behavior += end - start
        
    complete_duration_behavior = num_frames_duration_behavior / 30 # change number of frames to seconds
    return complete_duration_behavior



def get_num_usvs_beh(USV_start_ends: list[tuple[int,int]], start_ends: list[tuple[int,int]]) -> int:
    """
    Gets the number of USVs overlapping the times supplied for a behavior (most likely overlapping another function)
    Uses 2 lists of tuples of ints to get this. Returns an integer.
    """
    
    num_usvs = 0
        
    for j, (start_usv, end_usv) in enumerate(USV_start_ends):
        for start_beh, end_beh in start_ends:
        
            if start_usv < end_beh and end_usv > start_beh:
                num_usvs += 1
                # We now have a nested dictionary for 1 recording containing the raw number of usvs of a type 
                # during a specific behavior, irrespective of which mouse does the behavior
                break  # Stop checking further for this USV if overlap found
    return num_usvs


def get_usv_rates_per_behavior(df_deepof_cleaned, df_USV, recording_latency, behaviors) -> dict[float]:
    """
    Returns the number of USVs per second for each behavior in a dictionary. Uses a non-merged beh_start_ends in current form.
    """
    # initiate the dictionary
    beh_norm_USV_counts = {beh: 0.0 for beh in behaviors}
    
    ### NEWNEW
    beh_times = {beh: [] for beh in behaviors}
  
    
    # get_num_usvs_beh(USV_start_ends, beh_times_merged, behaviors)
    USV_start_ends, _ = get_frames_and_types_usvs(df_USV, recording_latency)
    # For each behavior, calculate and save the USV rate
    for beh in behaviors:
        
        #NEWNEWNEW
        # Get the start and end points of the merged list, and the number of USVs per behavior
        beh_start_ends = [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns]
        
        ### NOTE: THESE ARE NO LONGER MERGED START-ENDS, CHANGE NAME !!!!
        beh_times[beh] = beh_start_ends[0] + beh_start_ends[1]
        
        # Get the complete duration (in seconds) of the behavior and the number of USVs
        complete_duration_behavior = get_duration_behavior(beh_times[beh]) # in seconds !
        num_usvs = get_num_usvs_beh(USV_start_ends, beh_times[beh])
        
        # Calculates the number of usvs per second for the behavior, and saves it
        beh_norm_USV_counts[beh] = num_usvs / complete_duration_behavior
        
    return beh_norm_USV_counts 
        
    

def monte_carlo_test(df_USV,df_deepof_cleaned, behaviors):
    """
    Carries out a Monte Carlo test. The null hypothesis is that the likelihood of a USV occuring during any behaviour is equal,
    and that the behaviour type does not influence the USV rate. Therefore, through random permutation, a theoretical distribution 
    of USVs over specific behaviours is generated. Then, the observed USV rate is tested against the generated ones.
    
    The p-value reflects the extremity of the observed value when compared to the generated values.
    """

    USV_start_ends, _ = get_frames_and_types_usvs(df_USV, recording_latency)
    beh_start_ends = {beh: [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns] for beh in behaviors}
    for beh in behaviors: beh_start_ends[beh] = beh_start_ends[beh][0]+ beh_start_ends[beh][1]
    
    
    # Select the data that will be the "observed" data
    observed_events = np.array([get_num_usvs_beh(USV_start_ends, beh_start_ends[beh]) for beh in behaviors])  # Example counts for each behavior
    total_events = observed_events.sum()
    num_behaviors = len(behaviors)

    # Get the time spent in each behaviour and how much time was tagged for a behaviour in total
    behavior_times = [get_duration_behavior(beh_start_ends[beh]) for beh in behaviors]
    total_time = sum(time_spent)
    expected_counts = [total_events * beh_time/total_time for beh_time in behavior_times] # This is here for better interpretation of results

    # IMPORTANT: Number of events will be much higher than the actual number of USVs. This is due to double to quadruple tagging.
    
    # Variables to be used for the generation of permutations
    N = total_events
    T = total_time
    B = behavior_times # times spent per behavior
    n_permutations = 10000
    permuted_counts = np.zeros((n_permutations + 1, num_behaviors))
    
    # Generating the permutations
    for i in range(n_permutations):

        # This allocates USVs to behaviours purely based on the time spent in each behaviour
        permuted_events = np.random.multinomial(N,[beh_time/T for beh_time in B])
        for j,beh in enumerate(behaviors): permuted_counts[i, :] = permuted_events
    
    permuted_counts[-1, :] = observed_events
    
    # Get the mean and standard deviation of the permutations
    means = np.mean(permuted_counts, axis=0)
    stds = np.std(permuted_counts, axis=0)
    
    
    # This calculates the p-value, however it is possible to get a p = 0.0 
    # This happens when NONE of the generated values are more extreme than the observed value.
    # Due to rules surrounding probability calculation however, the minimal p-value generated should
    # reflect the size of the permutation, so 1/number of permutations

    p_values = []
    # Loop through each behavior (each index of USVs)
    for i in range(len(observed_events)):
        # Extract the permuted distribution for behavior i
        permuted_dist = permuted_counts[:, i]
        
        # Fit a normal distribution to the permuted distribution
        mu, sigma = norm.fit(permuted_dist)
        
        # Calculate the z-score of the observed value under the fitted normal distribution
        z_score = (observed_events[i] + observed_events[i]/5 - mu) / sigma
        
        # Calculate the two-tailed p-value from the z-score
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        p_values.append(p_value)
    
    # Purely the textual output
    for j, beh in enumerate(behaviors): 
    
        if p_values[j] < 1e-10:
                # Use scientific notation for very small p-values
            print(f"The p_value for {beh} was {p_values[j]:.2e}, \nwith the permuted count {means[j]}+- {stds[j]} \nand the observed count {observed_events[j]}")
            print(f"The expected number of USVs was {expected_counts[j]}")
        else:
            print(f"The p_value for {beh} was {p_values[j]:.10f}, \nwith the permuted count {means[j]}+- {stds[j]} \nand the observed count {observed_events[j]}")
            print(f"The expected number of USVs was {expected_counts[j]}")

    
    return permuted_counts
    
    

#%% Figure 2: Number of USVs per behaviour type and the expected number of USVs per behaviour type

fig, axs = plt.subplots(2, 3)
fig.suptitle("Time spent per type of behavior")
plt.subplots_adjust(hspace=0.5)


for count, recording in enumerate(filenames_per_recording):
    """
    This figure will contain the times spent in each behaviour, how many USVs were emitted during each behaviour, 
    and how many USVs one would expect given the null hypothesis that the behaviour types have no effect on the 
    likelihood of a USV being produced.
    """
    # Read in the file
    ((usv_folder, usv_file), (deepof_folder, deepof_file), (dlc_folder, dlc_file), recording_latency) = recording 
    df_deepof = pd.read_hdf(os.path.join(deepof_folder, deepof_file))
    df_USV = pd.read_excel(os.path.join(usv_folder,usv_file))


    df_deepof_cleaned = clean_deepof_df(df_deepof)
    
    behaviors = ['nose2nose', 'nose2body', 'nose2tail',  'follow', 'climb', 'huddle', "Not classified"]
    
    beh_start_ends = {beh: [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns] for beh in behaviors}
    for beh in behaviors: beh_start_ends[beh] = beh_start_ends[beh][0] + beh_start_ends[beh][1]
    time_spent = [get_duration_behavior(beh_start_ends[beh]) for beh in behaviors]
    
    total_time = sum(time_spent)
    
    
#%% Subplot 1: Percentage of time spent in each behavior    
    
    perc_time_spent = [100 * t / total_time for t in time_spent]
    
    ax = axs[0, count]
    
    ax.bar(range(7), perc_time_spent, tick_label=behaviors, color = 'red', alpha = 0.5)
    ax.set_xlabel("Behavior class")
    if count == 0: ax.set_ylabel("Time spent in behavior (%)")

    ax.set_xticklabels(beh_start_ends, rotation=45, ha='right')


    num_USVs = len(df_USV[df_USV.columns[0]])
    
    print(num_USVs)

#%% Subplot 2: Number of USVs per behaviour vs expected number of USVs per behaviour
    
    # Permute events
    permuted_events = monte_carlo_test(df_USV, df_deepof_cleaned, behaviors)

    # To show the little grey lines, randomly choose 100 permutations to plot.
    selected_indices = np.random.choice(range(len(permuted_events)), size=100)
    boot_vals = permuted_events[selected_indices, :]   
    selected_values = [boot_vals[i, :] / time_spent for i in range(len(boot_vals))]
    
    # Gather the time (s) spent in each behaviour
    beh_start_ends = {beh: [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns] for beh in behaviors}
    for beh in behaviors: beh_start_ends[beh] = beh_start_ends[beh][0] + beh_start_ends[beh][1]
    time_spent = [get_duration_behavior(beh_start_ends[beh]) for beh in behaviors]
    
    # Calculate the USV rate
    beh_norm_USV_counts = get_usv_rates_per_behavior(df_deepof_cleaned, df_USV, recording_latency, behaviors)
    

    # Make the second row of plots, showing the actual USV rate against the permuted USV rate
    ax = axs[1, count]
    ax.bar(range(7), beh_norm_USV_counts.values(), tick_label=list(beh_norm_USV_counts.keys()),color = 'red', alpha = 0.5)
    x_values = np.arange(0, len(behaviors), 1)
    for i in range(100): ax.plot(x_values, selected_values[i], color = "grey", linewidth = 0.1)
    ax.set_xlabel("Behavior class")
    ax.set_ylabel("USV rate (#/s)")
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    

plt.show()

plt.savefig(os.path.join(folder, f'USVs_emission_rates_during_specific_behaviors_summary.pdf'))
plt.savefig(os.path.join(folder, f'USVs_emission_rates_during_specific_behaviors_summary.jpg'))
