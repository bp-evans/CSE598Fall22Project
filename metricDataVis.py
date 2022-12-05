# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns
import numpy as np
from pandas.plotting import parallel_coordinates

# read text file into pandas
dataBCRRT = pd.read_csv("BCRRT_Data.txt", sep=",", header=None, names=["time", "success", "expansions"])
dataIMGBCRRT_SA = pd.read_csv("IMGBCRRT_StaticAll.txt", sep=",", header=None, names=["time", "success", "expansions"])
dataIMGBCRRT_SDO = pd.read_csv("IMGBCRRT_StaticStartGoal_Dynamic_Obs.txt", sep=",", header=None, names=["time", "success", "expansions"])
dataIMGBCRRT_DA = pd.read_csv("IMGBCRRT_DynamicAll.txt", sep=",", header=None, names=["time", "success", "expansions"])

# Create dataframes
dfBCRRT = pd.DataFrame(dataBCRRT)
dfIMGBCRRT_SA = pd.DataFrame(dataIMGBCRRT_SA)
dfIMGBCRRT_SDO = pd.DataFrame(dataIMGBCRRT_SDO)
dfIMGBCRRT_DA = pd.DataFrame(dataIMGBCRRT_DA)
##########################################################################################################
# Count BCRRT rows
rowBCRRTCount = dfBCRRT.shape[0]
#print(rowRRTCount)

# Count IMGBCRRT_SA rows
rowIMGBCRRT_SACount = dfIMGBCRRT_SA.shape[0]
#print(rowBCRRTCount)

# Count IMGBCRRT_SDO rows
rowIMGBCRRT_SDOCount = dfIMGBCRRT_SDO.shape[0]
#print(rowBCRRTCount)

# Count IMGBCRRT_DA rows
rowIMGBCRRT_DACount = dfIMGBCRRT_DA.shape[0]
#print(rowBCRRTCount)
##########################################################################
# BCRRT - Count and filter success data
successesBCRRT = dfBCRRT['success'].value_counts()[True]
print('Count of True Values in BCRRT:', successesBCRRT)

failuresBCRRT = dfBCRRT['success'].value_counts()[False]
print('Count of False Values in BCRRT:', failuresBCRRT)

print("The BCRRT success rate is:", round((successesBCRRT / rowBCRRTCount), 2))

# IMGBCRRT_SA - Count and filter success data
successesIMGBCRRT_SA = dfIMGBCRRT_SA['success'].value_counts()[True]
print('Count of True Values in IMGBCRRT_SA:', successesIMGBCRRT_SA)

#failuresIMGBCRRT_SA = dfIMGBCRRT_SA['success'].value_counts()[False]
#print('Count of False Values in IMGBCRRT_SA:', failuresIMGBCRRT_SA)

print("The IMGBCRRT_SA success rate is:", round((successesIMGBCRRT_SA / rowIMGBCRRT_SACount), 2))

# IMGBCRRT_SDO - Count and filter success data
successesIMGBCRRT_SDO = dfIMGBCRRT_SDO['success'].value_counts()[True]
print('Count of True Values in IMGBCRRT_SDO:', successesIMGBCRRT_SDO)

#failuresIMGBCRRT_SDO = dfIMGBCRRT_SDO['success'].value_counts()[False]
#print('Count of False Values in IMGBCRRT_SDO:', failuresIMGBCRRT_SDO)

print("The IMGBCRRT_SDO success rate is:", round((successesIMGBCRRT_SDO / rowIMGBCRRT_SDOCount), 2))

# IMGBCRRT_DA - Count and filter success data
successesIMGBCRRT_DA = dfIMGBCRRT_DA['success'].value_counts()[True]
print('Count of True Values in IMGBCRRT_DA:', successesIMGBCRRT_DA)

failuresIMGBCRRT_DA = dfIMGBCRRT_DA['success'].value_counts()[False]
print('Count of False Values in IMGBCRRT_DA:', failuresIMGBCRRT_DA)

print("The IMGBCRRT_DA  success rate is:", round((successesIMGBCRRT_DA / rowIMGBCRRT_DACount), 2))
############################################################################################
# BCRRT Time Data
timeDataBCRRT = dfBCRRT['time']
timeDiffBCRRT = timeDataBCRRT.diff()
print("BCRRT Time - Mean: "+str(timeDiffBCRRT.mean())+" Std. Dev: "+str(timeDiffBCRRT.std()))

# IMGBCRRT_SA Time Data
timeDataIMGBCRRT_SA = dfIMGBCRRT_SA['time']
timeDiffIMGBCRRT_SA = timeDataIMGBCRRT_SA.diff()
print("IMGBCRRT_SA Time - Mean: "+str(timeDiffIMGBCRRT_SA.mean())+" Std. Dev: "+str(timeDiffIMGBCRRT_SA.std()))

# IMGBCRRT_SDO Time Data
timeDataIMGBCRRT_SDO = dfIMGBCRRT_SDO['time']
timeDiffIMGBCRRT_SDO = timeDataIMGBCRRT_SDO.diff()
print("IMGBCRRT_SDO Time - Mean: "+str(timeDiffIMGBCRRT_SDO.mean())+" Std. Dev: "+str(timeDiffIMGBCRRT_SDO.std()))

# IMGBCRRT_DA Time Data
timeDataIMGBCRRT_DA = dfIMGBCRRT_DA['time']
timeDiffIMGBCRRT_DA = timeDataIMGBCRRT_DA.diff()
print("IMGBCRRT_DA Time - Mean: "+str(timeDiffIMGBCRRT_DA.mean())+" Std. Dev: "+str(timeDiffIMGBCRRT_DA.std()))

################################################################################
# Create Box Plot of Time Data
sns.boxplot(data = [timeDiffBCRRT,timeDiffIMGBCRRT_SA,timeDiffIMGBCRRT_SDO,timeDiffIMGBCRRT_DA]).set(xlabel="Algorithm", ylabel="Time to Terminal", xticks=([0, 1, 2, 3]), xticklabels = (['BCRRT', 'IMGBCRRT_SA', 'IMGBCRRT_SDO', 'IMGBCRRT_DA']))
#ax = plt.axes()
plt.title('Time to Goal')
#ax.set_xticklabels(["BCRRT","IMGBCRRT_SA","IMGBCRRT_SDO","IMGBCRRT_DA"])
plt.show()

######################################################################################
# BCRRT Expansion Data
expansionDataBCRRT = dfBCRRT['expansions']
print("BCRRT Expansion - Mean: "+str(expansionDataBCRRT.mean())+" Std. Dev: "+str(expansionDataBCRRT.std()))

# IMGBCRRT_SA Expansion Data
expansionDataIMGBCRRT_SA = dfIMGBCRRT_SA['expansions']
print("IMGBCRRT_SA Expansion - Mean: "+str(expansionDataIMGBCRRT_SA.mean())+" Std. Dev: "+str(expansionDataIMGBCRRT_SA.std()))

# IMGBCRRT_SDO Expansion Data
expansionDataIMGBCRRT_SDO = dfIMGBCRRT_SDO['expansions']
print("IMGBCRRT_SDO Expansion - Mean: "+str(expansionDataIMGBCRRT_SDO.mean())+" Std. Dev: "+str(expansionDataIMGBCRRT_SDO.std()))

# IMGBCRRT_DA Expansion Data
expansionDataIMGBCRRT_DA = dfIMGBCRRT_DA['expansions']
print("IMGBCRRT_DA Expansion - Mean: "+str(expansionDataIMGBCRRT_DA.mean())+" Std. Dev: "+str(expansionDataIMGBCRRT_DA.std()))

#############################################################
# Create Box Plot of Expansion Data
ax = sns.boxplot(data = [expansionDataBCRRT,expansionDataIMGBCRRT_SA,expansionDataIMGBCRRT_SDO,expansionDataIMGBCRRT_DA])\
    .set(xlabel="Algorithm", ylabel="Expansions", xticks=([0, 1, 2, 3]), xticklabels = (['BCRRT', 'IMGBCRRT_SA', 'IMGBCRRT_SDO', 'IMGBCRRT_DA']))
plt.title('Number of Expansions')
plt.show()