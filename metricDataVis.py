# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns
import numpy as np
from pandas.plotting import parallel_coordinates

# read text file into pandas
dataRRT = pd.read_csv("RRT_Data.txt", sep=",", header=None, names=["time", "success", "expansions"])
dataBCRRT = pd.read_csv("BCRRT_Data.txt", sep=",", header=None, names=["time", "success", "expansions"])
dataIMGBCRRT = pd.read_csv("IMGBCRRT_Data.txt", sep=",", header=None, names=["time", "success", "expansions"])

# Create dataframes
dfRRT = pd.DataFrame(dataRRT)
dfBCRRT = pd.DataFrame(dataBCRRT)
dfIMGBCRRT = pd.DataFrame(dataIMGBCRRT)
##########################################################################################################
# Count RRT rows
rowRRTCount = dfRRT.shape[0]
#print(rowRRTCount)

# Count BCRRT rows
rowBCRRTCount = dfBCRRT.shape[0]
#print(rowBCRRTCount)

# Count BCRRT rows
rowIMGBCRRTCount = dfIMGBCRRT.shape[0]
#print(rowBCRRTCount)

# RRT - Count and filter success data
successesRRT = dfRRT['success'].value_counts()[True]
print('Count of True Values in RRT:', successesRRT)

failuresRRT = dfRRT['success'].value_counts()[False]
print('Count of False Values in RRT:', failuresRRT)

print("The RRT success rate is:", round((successesRRT / rowRRTCount), 2))

# BCRRT - Count and filter success data
successesBCRRT = dfBCRRT['success'].value_counts()[True]
print('Count of True Values in BCRRT:', successesBCRRT)

failuresBCRRT = dfBCRRT['success'].value_counts()[False]
print('Count of False Values in BCRRT:', failuresBCRRT)

print("The BCRRT success rate is:", round((successesBCRRT / rowBCRRTCount), 2))

# IMGBCRRT - Count and filter success data
successesIMGBCRRT = dfIMGBCRRT['success'].value_counts()[True]
print('Count of True Values in IMGBCRRT:', successesIMGBCRRT)

failuresIMGBCRRT = dfIMGBCRRT['success'].value_counts()[False]
print('Count of False Values in IMGBCRRT:', failuresIMGBCRRT)

print("The IMGBCRRT success rate is:", round((successesIMGBCRRT / rowIMGBCRRTCount), 2))
############################################################################################
# RRT Time Data
timeDataRRT = dfRRT['time']
timeDiffRRT = timeDataRRT.diff()
print("RRT Time - Mean: "+str(timeDiffRRT.mean())+" Std. Dev: "+str(timeDiffRRT.std()))

# BCRRT Time Data
timeDataBCRRT = dfBCRRT['time']
timeDiffBCRRT = timeDataBCRRT.diff()
print("BCRRT Time - Mean: "+str(timeDiffBCRRT.mean())+" Std. Dev: "+str(timeDiffBCRRT.std()))

# IMGBCRRT Time Data
timeDataIMGBCRRT = dfIMGBCRRT['time']
timeDiffIMGBCRRT = timeDataIMGBCRRT.diff()
print("IMGBCRRT Time - Mean: "+str(timeDiffIMGBCRRT.mean())+" Std. Dev: "+str(timeDiffIMGBCRRT.std()))

################################################################################
# Create Box Plot of Time Data
sns.boxplot(data = [timeDiffRRT,timeDiffBCRRT,timeDiffIMGBCRRT]).set(xlabel="Algorithm", ylabel="Time to Terminal")
plt.title('RRT vs BCRRT vs IMGBCRRT: Time to Goal')
plt.show()

####################################################################################
# RRT Expansion Data
expansionDataRRT = dfRRT['expansions']
print("RRT Expansion - Mean: "+str(expansionDataRRT.mean())+" Std. Dev: "+str(expansionDataRRT.std()))

# BCRRT Expansion Data
expansionDataBCRRT = dfBCRRT['expansions']
print("BCRRT Expansion - Mean: "+str(expansionDataBCRRT.mean())+" Std. Dev: "+str(expansionDataBCRRT.std()))

# IMGBCRRT Expansion Data
expansionDataIMGBCRRT = dfIMGBCRRT['expansions']
print("IMGBCRRT Expansion - Mean: "+str(expansionDataIMGBCRRT.mean())+" Std. Dev: "+str(expansionDataIMGBCRRT.std()))
#############################################################
# Create Box Plot of Expansion Data
sns.boxplot(data = [expansionDataRRT,expansionDataBCRRT,expansionDataIMGBCRRT]).set(xlabel="Algorithm", ylabel="Expansions")
plt.title('RRT vs BCRRT vs IMGBCRRT: Expansions')
plt.show()