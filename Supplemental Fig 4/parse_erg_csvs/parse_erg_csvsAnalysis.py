#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##
# This jupyter notebook parses ERG session csv files


# In[146]:


import csv
import glob
import os
import re
import string

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt



# In[194]:


## Constants
# Edit these

# Directory containing your input CSV files
# DATA_INPUT_DIR = '/Users/jennyb_science/Documents/ERG_CSVfiles/PostInjection'
DATA_INPUT_DIR = '/Users/jennyb_science/Documents/ERG_CSVfiles/SynapticBlockers/GYKI_NBQX'
DATA_OUTPUT_NAME = 'erg_pre.csv'
Photopic = 0;

# Helper method to generate CSV cell locations ('A4') for each section and step
def csv_locs_step(key_row, step_row, stim_key_row, stim_step_row):
    return [
        { "label": f'A{key_row}', "value": f'A{step_row}' },                     # Step #
        { "label": f'H{stim_key_row}', "value": f'H{stim_step_row}' },           # StimFreq
        *[                                                                       # Step Data (a [ms] ... Avgs)
            { "label": f'{key_col}{key_row}', "value": f'{key_col}{step_row}' }
            for key_col in ["B", "C", "D", "E", "F"]
        ]
        
    ]

# List of CSV locations to parse, as pairs (location of the data label, and the location of the data value)
CSV_LOCS = [
    { "label": "A3", "value": "A4" }, # Patient
    { "label": "E3", "value": "E4" }, # TestDate
    #Photopic
    # if (Photopic):
        *csv_locs_step(key_row=18, step_row=19, stim_key_row=6, stim_step_row=7),
        *csv_locs_step(key_row=18, step_row=20, stim_key_row=6, stim_step_row=8),
        *csv_locs_step(key_row=37, step_row=38, stim_key_row=6, stim_step_row=22),
        *csv_locs_step(key_row=37, step_row=39, stim_key_row=6, stim_step_row=23),
    # #Scotopic
    # else:
        # *csv_locs_step(key_row=24, step_row=25, stim_key_row=6, stim_step_row=7),
        # *csv_locs_step(key_row=24, step_row=26, stim_key_row=6, stim_step_row=8),
        # *csv_locs_step(key_row=24, step_row=27, stim_key_row=6, stim_step_row=9),
        # *csv_locs_step(key_row=24, step_row=28, stim_key_row=6, stim_step_row=10),
        # *csv_locs_step(key_row=41, step_row=42, stim_key_row=6, stim_step_row=30),
        # *csv_locs_step(key_row=41, step_row=43, stim_key_row=6, stim_step_row=31),
]


# In[211]:


def load_filepaths_in_directory(directory):
    '''
        Search for and return all files (filepaths) in a directory
    '''
    print(f'Loading filepaths from "{directory}"')
    # Check that the directory exists
    exists = os.path.exists(directory)
    if not exists:
        print(f'Directory "${directory}" does not exist')
        return []

    else:
        # Gather all filepaths in this directory
        paths = glob.glob(os.path.join(directory, "*"))
        filepaths = []

        for path in paths:
            # Check that the filepath is for a file, and skip if directory
            if not os.path.isfile(path):
                continue

            print(" - " + path)
            filepaths.append(path)

        return filepaths


filepaths = load_filepaths_in_directory(DATA_INPUT_DIR)


# In[216]:


loc_regex = '^([A-Za-z]+)([0-9]+)$'
def csv_location_to_index(loc):
    '''
        Convert a CSV cell location (e.g. 'A4') to row, col indeces (e.g. (3, 0))
    '''
    match = re.search(loc_regex, loc)

    row = int(match.group(2)) - 1
    col = ord(match.group(1).lower()) - 97
    return (row, col)


def load_erg_csv(filepath):
    '''
        Load a csv file and convert to np array of cells
    '''
    print(f'- Reading data from "{filepath}"')
    max_n_cols = 0
    lines = []
    mat = []

    # Open the CSV file at 'filepath', parse it with encoding "latin-1"
    #  (to handle additional character such as 'mu', etc)
    with open(filepath, 'r', encoding="latin-1") as csvfile:
        reader = csv.reader(csvfile)
        
        # Read line by line
        for row in reader:
            if len(row) > max_n_cols:
                max_n_cols = len(row)
            lines.append(row)
        
    # Convert lines to 2d np array for easier parsing
    for line in lines:
        line = np.array(line)
        line.resize((max_n_cols,))
        mat.append(line)

    mat = np.array(mat)
    return mat

def parse_erg_csv(mat):
    '''
        Parse out all entries in 'CSV_LOCS' from a CSV given as a np 2D array
    '''
    entries = []
    for data_entry in CSV_LOCS:
        label_loc = data_entry["label"]
        label_i = csv_location_to_index(label_loc)

        if (label_i[0] > mat.shape[0]):
            print(f'  - Could not parse csv')
            return []

        label = mat[label_i]
        
        value_loc = data_entry["value"]
        value_i = csv_location_to_index(value_loc)
        value = mat[value_i]
        
        entries.append({ "label": label, "value": value })
    return entries


def parse_erg_csvs(filepaths):
    '''
        
    '''
    print(f'Parsing data from files...')
    
    # Read each CSV, extracting relevant cells to list of (label, value) pairs
    files_data = []
    for filepath in filepaths:
        file_mat = load_erg_csv(filepath)
        file_parsed = parse_erg_csv(file_mat)
        if (len(file_parsed) > 0):
            files_data.append(file_parsed)

    
    # Create a Pandas CSV
    # Assume the labels in the first CSV are the same as the other CSVs,
    #  and use them as column headers for the output table
    files_keys = [[entry["label"] for entry in file_data] for file_data in files_data]
    cols = files_keys[0]

    df = pd.DataFrame(columns=cols)
    for file_data in files_data:
        vals = [entry["value"] for entry in file_data]
        row = pd.DataFrame([vals], columns=cols)
        df = pd.concat([df, row ], ignore_index=True)
    
    df.set_index("Patient")
    
    # Do some parsing for certain fields
    #  Remove ':' from "Step" cells
    df.loc[:, "Step"] = df.loc[:, "Step"].replace(to_replace=r':', value='', regex=True)

    print(df)
    return df


df = parse_erg_csvs(filepaths)


# In[217]:


def save_data(df):
    filepath = os.path.join(DATA_INPUT_DIR, DATA_OUTPUT_NAME)
    print(f'Saving data to "{filepath}"')
    df.to_csv(filepath)
    
save_data(df)


# In[ ]:
    
CompareLR = 1; 
CombineLR=0;
Photopic = 1;
IOP=0;
incStats = 1;
synapticB=0;

    
#constructID = ['SYNwt','CAGmut','SYNmut', 'GRM6mut','NEHJmut'];  

# #Photopic  
if (Photopic):
    # colmToPlotOD = [5,6,7,8, 19, 20, 21, 22]; #Right
    # colmToPlotOS = [12, 13, 14, 15, 26, 27, 28, 29]; #left
    # colTitles = ['a-wave duration (ms)', 'b-wave duration (ms)', 'a-wave amp (uV)', 'b-wave amp (uV)', 'N1 (ms)', 'P1 (ms)', 'N1-P1 (nV)', '28Hz Amp(uV)'];   

    # colmToPlotOD = [5,6,7,8, 19, 21, 22]; #Right
    # colmToPlotOS = [12, 13, 14, 15, 26, 28, 29]; #left
    # colTitles = ['a-wave duration (ms)', 'b-wave duration (ms)', 'a-wave amp (uV)', 'b-wave amp (uV)', 'N1 (ms)', 'N1-P1 (nV)', '28Hz Amp(uV)'];   


    colmToPlotOD = [7,8,21]; #Right
    colmToPlotOS = [14, 15, 28]; #left
    colTitles = ['a-wave amp (uV)', 'b-wave amp (uV)','N1-P1 (nV)'];   

elif (IOP):
    colmToPlotOD = [1,3]; #Right
    colmToPlotOS = [2, 4]; #left 
    colTitles = ['IOP (mmHg)', 'CCT (microns)'];   
 
else:
#Scotopic
    # colmToPlotOD = [5,6,7,8, 19, 20, 21, 22, 33, 35]; #Right
    # colmToPlotOS = [12, 13, 14, 15, 26, 27, 28, 29, 40, 42]; #left 
    # colTitles = ['a-wave duration (ms)', 'b-wave duration (ms)', 'a-wave amp (uV)', 'b-wave amp (uV)','a-wave duration (ms)', 'b-wave duration (ms)', 'a-wave amp (uV)', 'b-wave amp (uV)', 'N2 (ms)', 'OS2 (nV)'];   


    colmToPlotOD = [21, 22, 35]; #Right
    colmToPlotOS = [28,29, 42]; #left 
    colTitles = ['a-wave amp (uV)', 'b-wave amp (uV)','OS2 (nV)'];   



colrs = ["black", "Green", "Orange", "Violet", "Red"]

def uniqueish_color():
    return plt.cm.gist_ncar(np.random.random())
       
for index, colRef in enumerate(colTitles):
    
        ODpre=list(map(float, (erg_precsv[1:len(erg_precsv),colmToPlotOD[index]])))
        OSpre=list(map(float, (erg_precsv[1:len(erg_precsv),colmToPlotOS[index]])))

        if 'erg_postcsv' in locals():
            ODpost=list(map(float, (erg_postcsv[1:len(erg_postcsv),colmToPlotOD[index]])))
            OSpost=list(map(float, (erg_postcsv[1:len(erg_postcsv),colmToPlotOS[index]])))

        if (CombineLR): 
            preCombo = np.append(OSpre,ODpre)
            postCombo = np.append(OSpost,ODpost)
    
    
    #rank sum test
    #OSvOD
        if (incStats):
            if (CombineLR):
            #    tStat1 = stats.wilcoxon(preCombo,postCombo)
                tStat1, pValue1,z = stats.wilcoxon(preCombo,postCombo,method='approx')
                
                print(tStat1)
                print(pValue1)
            else:
                if (CompareLR):
                    tStat1, pValue1 = stats.wilcoxon(ODpre,OSpre) #t statistic, p, 
                    tStat2, pValue2 = stats.wilcoxon(ODpost,OSpost)
                else:
                    tStat1, pValue1 = stats.wilcoxon(ODpre,ODpost)
                    tStat2, pValue2 = stats.wilcoxon(OSpre,OSpost)

 
    # print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic
        if (synapticB):
            x1 = x1=list(range(1,len(ODpre)+1));

        elif (CombineLR):
            x1 = [1] *len(preCombo);
            x2 = [2] *len(postCombo);           
            
        else:
            x1 = [1] *len(ODpre);
            x2 = [2] *len(ODpost);
            x3 = [3] *len(OSpre);
            x4 = [4] *len(OSpost);
    
    #Left v Right
    
    
        if (synapticB):
            
            X=[x1,x1]
            Y=[ODpre,OSpre]
            
        elif (CombineLR):
            X = [x1,x2]
            Y = [preCombo,postCombo]   
            
        else:
            if (CompareLR):
                X = [x1,x2]
                Y = [ODpre,OSpre]  
                
                X2 = [x3,x4]
                Y2 = [ODpost,OSpost]
                
            else:
            #prev post
                X = [x1,x2]
                Y = [ODpre,ODpost]  
                
                X2 = [x3,x4]
                Y2 = [OSpre,OSpost]
            
        
        ### plotting
        if (CombineLR):
            labelIDs = np.append(erg_precsv[1:,1], erg_precsv[1:,1])
        else:
            labelIDs = erg_precsv[1:,1]
    
        fig, ax = plt.subplots(1) 
        for i in range(len(Y)):
            
            if (synapticB):
                ax.plot(X[i],Y[i],'o-',color=colrs[i])
            else:
                for j in range(len(Y[i])):
                   # ax.plot(X[i][j],Y[i][j],'o',color=colrs[j])
                   # ax.plot(X2[i][j],Y2[i][j],'o',color=colrs[j])
                   
                    ax.plot(X[i][j],Y[i][j],'o',color="k")
                    
                    if not (CombineLR):
                        ax.plot(X2[i][j],Y2[i][j],'o',color="k")
                        
                    if i==1:
                    #    ax.plot([X[i-1][j],X[i][j]],[Y[i-1][j],Y[i][j]],color=colrs[j],  label='ID %s ' % labelIDs[j]) #, label='Inline label'
                    #    ax.plot([X2[i-1][j],X2[i][j]],[Y2[i-1][j],Y2[i][j]],color=colrs[j])
                        
                        ax.plot([X[i-1][j],X[i][j]],[Y[i-1][j],Y[i][j]],color="k",  label='ID %s ' % labelIDs[j]) #, label='Inline label'
                        if not (CombineLR):
                            ax.plot([X2[i-1][j],X2[i][j]],[Y2[i-1][j],Y2[i][j]],color="k")
                        
      #  ax.legend()
        
        ###   
        
        
        if (Photopic):
            if (CombineLR):
                plt.xlim([0.5,2.5])
                plt.xticks([1, 2], ['Pre', 'Post'])
                
                if colmToPlotOD[index]<9:
                    plt.title("10Hz Photopic, PrePval = %.4f" % pValue1)
                
                else:
                    plt.title("20Hz Photopic, PrePval = %.4f" % pValue1)
                
                
            else :    
                plt.xlim([0.5,4.5])
                if (synapticB):
                    plt.xticks([1, 2, 3, 4, 5], ['pre', '+30', '+60', '+90', '+120'])

                elif (CompareLR):
                    plt.xticks([1, 2, 3, 4], ['ODPre', 'OSPre', 'ODPost', 'OSPost'])
                else:
                    plt.xticks([1, 2, 3, 4], ['ODPre', 'ODPost', 'OSPre', 'OSPost'])
                    
                if (incStats):
                    
                    if colmToPlotOD[index]<9:
                        plt.title("10Hz Photopic, PrePval = %.4f" % pValue1 + ", PostPval = %.4f" %pValue2)
                
                    else:
                        plt.title("20Hz Photopic, PrePval = %.4f" % pValue1 + ", PostPval = %.4f" %pValue2)
                    
        elif (IOP):
            plt.xlim([0.5,2.5])
            plt.xticks([1, 2], ['Pre', 'Post'])
            
            if colmToPlotOD[index]<2:
                    plt.title("IOP, Pval = %.4f" % pValue1)
                    
            else :
                    plt.title("Central Corneal Thickness, Pval = %.4f" % pValue1)
                    
        else:
            plt.xlim([0.5,2.5])
            plt.xticks([1, 2], ['Pre', 'Post'])
            
            if (incStats):
                if colmToPlotOD[index]<16:
                        plt.title("Scotopic -20dB, Pval = %.4f" % pValue1)
                        
                elif colmToPlotOD[index]>30:
                            plt.title("Scotopic oscillatory potential 0dB, Pval = %.4f" % pValue1)
                    
                else :
                        plt.title("Scotopic max.potential 0db, Pval = %.4f" % pValue1)
                    
            
        plt.ylabel(colRef)
      #  plt.legend()
        plt.savefig(colRef+'.svg', dpi=500)
       # plt.savefig(colRef+'.svg', format='svg', dpi=1200)

        



