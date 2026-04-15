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


behaviors = ["nose2nose", "nose2body", "nose2tail", "follow", "huddle", "climb", "Not classified"]
labels = ['Short','Upwards','Flat','Downwards','Chevron','U-Shape','Complex', 'Two Syllable', 'Multi Syllable','Harmonic','Frequency Step','Other']


#%% --- New behavior calculation method (non-overlapping, 1 per individual per frame)

# Choose the highest-priority event for a given row 
def choose_event(row, event_cols) -> str:
    # Priority: Event 1 > Event 2 > Event 3
    for event in event_cols:
        if row[event] == 1:
            return event  # Return the column name of the highest-priority event
    return None  # No event occurring at this time


def choose_empty(row, event_cols) -> int:
    """Sanity check for the choose_event() function"""
    for event in event_cols:
        if row[event] == 1:
            return None
    return 0


def clean_deepof_df(df_deepof) -> None:

    # This part just determines the order in which the behaviors will take priority over eachother for each of the individuals
    
    # DETERMINES THE PRIORITY, arbitrarily chosen.
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
    df_deepof['cleaned_event_ind_1'] = df_deepof.apply(lambda row: choose_event(row, priority_cols_ind_1), axis=1) # row = frame
    df_deepof['cleaned_event_ind_2'] = df_deepof.apply(lambda row: choose_event(row, priority_cols_ind_2), axis=1)
    
    
    
    # Sanity check: is the same number of most prioritized behavior (nose2nose) present in the cleaned version?
    count_per_element = df_deepof['cleaned_event_ind_1'].str.count('nose2nose')
    
    # To get the total count of occurrences in the entire Series
    total_count = count_per_element.sum()
    
    
    # THE ANSWER IS YES!
    print(df_deepof[priority_cols_ind_1[0]].sum() == total_count)

    return

#%% --- Number of USVs against type of behavior 


def get_event_times_new(time_series, beh_str) -> list[tuple[int]]:
    """
    Determine when the behaviours happen.
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


def check_overlaps_freq(events_1, events_2) -> list[float]:
    """
    What is events_1, what is events_2
    """
    overlaps = 0
    list_ = []
    
    # Loop through each event in events_1
    for i, (start1, end1) in enumerate(events_1):
        # Check for overlaps with events in events_2
        for start2, end2 in events_2:
            # Check if there is an overlap
            if start1 < end2 and end1 > start2:
                overlaps += 1
                list_.append(df_USV['Principal Frequency (kHz)'][i])
                break  # Stop checking further for this event_1 if overlap found
                
    return overlaps, list_


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
        
    complete_duration_behavior = num_frames_duration_behavior / frames_per_second # change number of frames to seconds
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
    
    ### Beh times is a list of tuples, start and end
    beh_times = {beh: [] for beh in behaviors}
  
    
    # get_num_usvs_beh(USV_start_ends, beh_times_merged, behaviors)
    USV_start_ends, _ = get_frames_and_types_usvs(df_USV, recording_latency)
    # For each behavior, calculate and save the USV rate
    for beh in behaviors:
        
        #NEWNEWNEW
        # Get the start and end points of the merged list, and the number of USVs per behavior
        beh_start_ends = [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns]
        
        ### NOTE: THESE ARE NO LONGER MERGED START-ENDS, CHANGE NAME !!!!
        # Beh_start_Ends[0] = individual 1, [1] is individual 2
        beh_times[beh] = beh_start_ends[0] + beh_start_ends[1] 
        
        # Get the complete duration (in seconds) of the behavior and the number of USVs
        complete_duration_behavior = get_duration_behavior(beh_times[beh]) # in seconds !
        num_usvs = get_num_usvs_beh(USV_start_ends, beh_times[beh])
        
        # Calculates the number of usvs per second for the behavior, and saves it
        beh_norm_USV_counts[beh] = num_usvs / complete_duration_behavior
        
    return beh_norm_USV_counts 



# START ENDS
def get_num_overlaps_beh(df_deepof_cleaned, behaviors, USV_start_ends) -> dict[int]:
    """Gives the number of USVs which happen during a specific behaviour"""
    num_overlaps_beh = {beh:0 for beh in behaviors}
    for beh in behaviors:
        # Find the number of overlaps
        
        for col in df_deepof_cleaned.columns:
            beh_start_ends = get_event_times_new(df_deepof_cleaned[col], beh)
            
            num_overlaps, _ = check_overlaps_freq(USV_start_ends, beh_start_ends)
            num_overlaps_beh[beh] += num_overlaps
            
    return num_overlaps_beh

#   START ENDS
def get_princ_freqs_usvs_per_beh(df_deepof_cleaned, behaviors, USV_start_ends) -> dict[list[float]]:
    """Gets the principal (or strongest) tone frequency of the USVs occuring during every behaviour type"""
    dict_freqs_per_beh = {beh:[] for beh in behaviors}
    
    for beh in behaviors:
        # Find the number of overlaps
        
        for col in df_deepof_cleaned.columns:
            beh_start_ends = get_event_times_new(df_deepof_cleaned[col], beh)
            
            _, list_freqs = check_overlaps_freq(USV_start_ends, beh_start_ends)
            dict_freqs_per_beh[beh] += list_freqs # Principal frequency of the USVs being counted
    
    return dict_freqs_per_beh



# %% --- Scatter of all principle frequencies per behavior type

for count, recording in enumerate(filenames_per_recording):
    """
    
    """
    # Read in the file
    ((usv_folder, usv_file), (deepof_folder, deepof_file), (dlc_folder, dlc_file), recording_latency) = recording 
    df_deepof = pd.read_hdf(os.path.join(deepof_folder, deepof_file))
    df_USV = pd.read_excel(os.path.join(usv_folder,usv_file))
    
    df_deepof_cleaned = clean_deepof_df(df_deepof)
    
    dict_freqs_per_beh = {beh: [] for beh in behaviors}
    
    for beh in behaviors: dict_freqs_per_beh[beh] = get_princ_freqs_usvs_per_beh(df_deepof_cleaned, behaviors)
    
    fig, ax0 = plt.subplots(1, sharey = False)
    
    # Subplot 1
    
    ax0.set_title('Principal frequencies of USVs for each behaviour')
    ax0.scatter(np.random.normal(1, 0.06, len(dict_freqs_per_beh['nose2body'])),
                dict_freqs_per_beh['nose2body'], color='r', alpha = 0.5)
    ax0.scatter(np.random.normal(2, 0.06, len(dict_freqs_per_beh['nose2nose'])),
                dict_freqs_per_beh['nose2nose'], color='r', alpha = 0.5)
    ax0.scatter(np.random.normal(3, 0.06, len(dict_freqs_per_beh['nose2tail'])),
                dict_freqs_per_beh['nose2tail'], color='r', alpha = 0.5)
    ax0.scatter(np.random.normal(4, 0.06, len(dict_freqs_per_beh['follow'])),
                dict_freqs_per_beh['follow'], color='r', alpha = 0.5)
    ax0.scatter(np.random.normal(5, 0.06, len(dict_freqs_per_beh['climb'])),
                dict_freqs_per_beh['climb'], color='r', alpha = 0.5)
    ax0.scatter(np.random.normal(6, 0.06, len(dict_freqs_per_beh['huddle'])),
                dict_freqs_per_beh['huddle'], color='r', alpha = 0.5)




# ax0.set_ylim([0, 50])
ax0.spines[['right', 'top']].set_visible(False)
# ax0.set_ylabel('% of time\nperforming behavior')
ax0.set_xticks([1, 2, 3, 4, 5, 6], ['Nose-to-body', 'Nose-to-nose', 'Nose-to-tail', 'Follow', 'Stationary', 'Climb'])

#%% --- Call types per behavior (new)

# Plotting manually using plt.bar()
def plot_stacked_bar_manual(pivot_df, colors, title, ax) -> None:
    # List of time categories (the x-axis)
    behaviors = ["nose2nose", "nose2body", "nose2tail", "follow", "huddle", "climb", "Not classified"]
    # Number of time categories
    n_behaviors = len(behaviors)
    # X-axis positions
    x = np.arange(n_behaviors)

    # # Initialize bottom to 0 (for stacking the bars)
    # bottoms = np.zeros(n_behaviors)
    
    
    
    # Plot each event_type as a layer in the stacked bar
    for j, behavior in enumerate(behaviors):
        bottom = 0
        # Get the event rates for the current event_type
        print(behavior)
        for USV_type in pivot_df.columns:
            event_rate = pivot_df.loc[behavior, (USV_type)]
            # Plot the bar for the current event_type, stacked on top of the previous one
            ax.bar(x[j], event_rate, bottom=bottom, color=colors[USV_type], label = USV_type)
        
            # Update the bottom for the next stack
            bottom += event_rate

    # Add labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha='right')  # Set the x-axis labels to the time categories
    # Add labels and title
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('USV rate (#/s)')
    ax.set_title(title)
    
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove duplicate labels
    unique_handles = []
    unique_labels = []
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
    
    # Create legend with unique labels
    plt.legend(title = 'USV type', handles=unique_handles[::-1], labels=unique_labels[::-1], bbox_to_anchor = (1,1.05))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return



#%%  --- Figure 4 : USV types for each behaviour ---


fig, axs = plt.subplots(2, 3)

for count, recording in enumerate(filenames_per_recording):
    """
    
    """
    # Read in the file
    ((usv_folder, usv_file), (deepof_folder, deepof_file), (dlc_folder, dlc_file), recording_latency) = recording 
    df_deepof = pd.read_hdf(os.path.join(deepof_folder, deepof_file))
    df_USV = pd.read_excel(os.path.join(usv_folder, usv_file))
    
    df_deepof_cleaned = clean_deepof_df(df_deepof)
    
    print("Original behaviors order:", behaviors)
    # print(df_deepof_cleaned.head())
    # labels = list(set(df_USV['Label']))
    
    
    # Make a dictionary to save the number of occurences per usv type in
    usv_types_cleaned_beh = {call_type: {beh: 0 for beh in behaviors} for call_type in labels}

    beh_times = {beh: [] for beh in behaviors}
    
    USV_start_ends, list_call_types = get_frames_and_types_usvs(df_USV, recording_latency)
    
    # Choose a continuous colormap (like 'viridis' or 'plasma')
    cmap = plt.get_cmap('tab20')

    # Use list comprehension to assign a color from the colormap to each key
    colors = {key: cmap(i) for i, key in enumerate(labels)}

    
    # Per behavior type, check whether a USV overlaps, then save the occurence based on type of call and behavior
    for beh in behaviors:
        # Get the times of that behavior
        
    
        beh_start_ends = [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns]
        
        ### NOTE: THESE ARE NO LONGER MERGED START-ENDS, CHANGE NAME !!!!
        # Combine the individuals
        beh_times[beh] = beh_start_ends[0] + beh_start_ends[1] # so now we have a dictionary containing in each key a list of tuples of start_end
        
            
            
        # Check whether the occurence of the USV falls within the occurence of behavior
        for j, (start_usv, end_usv) in enumerate(USV_start_ends):
            for start_beh, end_beh in beh_times[beh]:
                
                # The actual check
                if start_usv < end_beh and end_usv > start_beh:
                    if list_call_types[j] == 'Two syllable': list_call_types[j] = 'Two Syllable'
                    usv_types_cleaned_beh[list_call_types[j]][beh] += 1 
                    # We now have a nested dictionary for 1 recording containing the raw number of usvs of a type 
                    # during a specific behavior, irrespective of which mouse does the behavior
                    break  # Stop checking further for this USV if overlap found
    
    
    # Makes a new dictionary
    normalized_usvs_per_type_per_behavior = {call_type: {beh: 0.0 for beh in behaviors} for call_type in labels}
    
    # Saves the USV type along with during which behaviour it happened for all USVs
    for beh in behaviors:
        
        complete_duration_behavior = get_duration_behavior(beh_times[beh])
        
        for usv_type in labels:
            num_usvs = usv_types_cleaned_beh[usv_type][beh]
            usv_rate_per_second = num_usvs / complete_duration_behavior
            normalized_usvs_per_type_per_behavior[usv_type][beh] = usv_rate_per_second
        


#%% --- Plot the stacked bar

    
    # Gets out the list of occurences during each type of behavior per type of USV
    usv_occurences_by_type = [usv_types_cleaned_beh[label] for label in labels]
        
    
    dict_usv_rates_per_beh ={'USV_rate': [], 'USV_type': [], 'Behavior': []}
    
    # Get all the info into a more useful dataframe
    for beh in behaviors:
        for usv_type in labels:
            dict_usv_rates_per_beh['USV_rate'].append(normalized_usvs_per_type_per_behavior[usv_type][beh])
            dict_usv_rates_per_beh['USV_type'].append(usv_type) 
            dict_usv_rates_per_beh['Behavior'].append(beh)
    
    df_usv_rates_beh = pd.DataFrame(dict_usv_rates_per_beh)
    
    df_usv_rates_pivot = df_usv_rates_beh.pivot(index = "Behavior", columns = 'USV_type')
    df_usv_rates_pivot.columns = df_usv_rates_pivot.columns.droplevel(0)
    
    df_usv_rates_pivot.reindex(behaviors)
    plt.figure()
    
    # Choose a continuous colormap (like 'viridis' or 'plasma')
    cmap = plt.get_cmap('tab20')

    # Use list comprehension to assign a color from the colormap to each key
    colors = {key: cmap(i) for i, key in enumerate(labels)}
    
#%%  Subplot 1: Stacked barplot (Non-normalized)

    ax = axs[0,count]
    plot_stacked_bar_manual(df_usv_rates_pivot, colors, 'Rate of occurence of USV types per behavior', ax)
    


#%% Subplot 2: Other stacked bar plot
    ax = axs[1, count]
    # Combine the data into a single array for easier manipulation
    all_data = np.array([list(num_occurences.values()) for num_occurences in usv_occurences_by_type])
    
    # Total for each category (used for relative amounts)
    totals = np.sum(all_data, axis=0)
    
    # Total absolute values across all categories combined
    total_all_categories = np.sum(all_data, axis=1)
    
    rel_occurences_by_type = {usv_type:[] for usv_type in labels}
    

    # Normalize each subcategory to get the relative values (percentages)
    for i, usv_type in enumerate(labels): rel_occurences_by_type[usv_type] = list(usv_occurences_by_type[i].values()) / totals
    
    # The position of the bars on the x-axis
    bar_width = 0.35
    bar_positions = np.arange(len(behaviors))  # Positions for the behaviors
    abs_bar_positions = bar_positions + bar_width  # Shifted position for absolute bar  
    
    # Initialize the bottom array with zeros, which will track the stacking position
    bottom = np.zeros(len(behaviors))
    
    # Iterate over the subcategories and plot each one
    for i, usv_rate in enumerate(rel_occurences_by_type.values()):
        ax.bar(bar_positions, usv_rate, bar_width, bottom=bottom, label=f'{list(rel_occurences_by_type.keys())[i]}', color = colors[list(rel_occurences_by_type.keys())[i]])
        # Update the bottom array to stack the next bar on top of the current one
        bottom += usv_rate
    
    # Add labels, title, and a legend
    ax.set_xlabel('Behavior')
    ax.set_ylabel('Proportion USVs')
    ax.set_title('Proportion of USV types per behavior')
    ax.set_xticks(bar_positions, behaviors)  # Set x-tick labels to the category names
    # Place the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    

# Save and show the plot
plt.savefig(os.path.join(folder, f'Proportion of USV types per behavior{deepof_file[:-3]}.pdf'))
plt.savefig(os.path.join(folder, f'Proportion of USV types per behavior{deepof_file[:-3]}.jpg'))
    
plt.show()
     


#%% --- Figure with the subplots of usv_types per behavior ---


fig, axs = plt.subplots(3,3, sharey = True)


for count, recording in enumerate(filenames_per_recording):
    # Read in the file
    ((usv_folder, usv_file), (deepof_folder, deepof_file), (dlc_folder, dlc_file), recording_latency) = recording 
    df_deepof = pd.read_hdf(os.path.join(deepof_folder, deepof_file))
    df_USV = pd.read_excel(os.path.join(usv_folder, usv_file))
    
    df_deepof_cleaned = clean_deepof_df(df_deepof)
    
    print("Original behaviors order:", behaviors)
    # print(df_deepof_cleaned.head())
    # labels = list(set(df_USV['Label']))
    
    
    # Make a dictionary to save the number of occurences per usv type in
    usv_types_cleaned_beh = {beh: {call_type:  0 for call_type in labels} for beh in behaviors} 

    beh_times = {beh: [] for beh in behaviors}
    
    USV_start_ends, list_call_types = get_frames_and_types_usvs(df_USV, recording_latency)
    
    # Choose a continuous colormap (like 'viridis' or 'plasma')
    cmap = plt.get_cmap('tab20')

    # Use list comprehension to assign a color from the colormap to each key
    colors = {key: cmap(i) for i, key in enumerate(labels)} 
    
    # Per behavior type, check whether a USV overlaps, then save the occurence based on type of call and behavior
    for beh in behaviors:
        # Get the times of that behavior
    
        beh_start_ends = [get_event_times_new(df_deepof_cleaned[col], beh) for col in df_deepof_cleaned.columns]
        
        beh_times[beh] = beh_start_ends[0] + beh_start_ends[1] # so now we have a dictionary containing in each key a list of tuples of start_end
        
            
            
        # Check whether the occurence of the USV falls within the occurence of behavior
        for j, (start_usv, end_usv) in enumerate(USV_start_ends):
            for start_beh, end_beh in beh_times[beh]:
                
                # The actual check
                if start_usv < end_beh and end_usv > start_beh:
                    if list_call_types[j] == 'Two syllable': list_call_types[j] = 'Two Syllable'
                    usv_types_cleaned_beh[beh][list_call_types[j]] += 1 
                    # We now have a nested dictionary for 1 recording containing the raw number of usvs of a type 
                    # during a specific behavior, irrespective of which mouse does the behavior
                    break  # Stop checking further for this USV if overlap found

    
    
    normalized_usvs_per_type_per_behavior =  {beh: {call_type: 0.0 for call_type in labels} for beh in behaviors}
    
    for beh in behaviors:
        
        complete_duration_behavior = get_duration_behavior(beh_times[beh])
        
        for usv_type in labels:
            num_usvs = usv_types_cleaned_beh[beh][usv_type]
            usv_rate_per_second = num_usvs / complete_duration_behavior
            normalized_usvs_per_type_per_behavior[beh][usv_type] = usv_rate_per_second
            
    for j, beh in enumerate(behaviors):       
        ax = axs[j//3, j%3]
        
        ax.set_title(beh)
        num_usvs_per_type = usv_types_cleaned_beh[beh]
        total_usvs = sum(num_usvs_per_type.values())
        usv_type_percentages = {usv_type: 100 * num_usvs_per_type[usv_type] / total_usvs if total_usvs > 0 else None for usv_type in labels }
        
        # Create x-tick positions
        x_ticks = range(len(labels))
        
        # If you want to change how the USV types are marked, consider using colors in stead of these icons
        if count == 0: mark = '^'
        if count == 1: mark = 's'
        if count == 2: mark = 'o'
        
        # Plot each dictionary's values with a different marker
        ax.plot(x_ticks, list(usv_type_percentages.values()), label=f'{deepof_file[8:16]}',marker = mark, markersize=7, color = 'red', alpha = 0.5)
        
        # Set the x-ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')


ax = axs[2,0]
# Add legend to differentiate between the dictionaries
ax.legend(bbox_to_anchor=(1, 0.5))

fig.delaxes(axs[2,1])
fig.delaxes(axs[2,2])

# Display the plot
plt.show()