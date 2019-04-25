    import colours as colours
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import os
import pandas as pd
import statistics
import researchpy as rp
import operator

import statistics as statistics
import xlrd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot
from statsmodels.formula.api import ols
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.multicomp
from scipy import stats
import math
import csv

plotly.tools.set_credentials_file(username='Bowhaven', api_key='naNWxEKc6cZy3b1bxLFy')

speciesCoverPath = "C:/Users/chris/OneDrive/Documents/RP1/CleanedData/2004to2014plotSpecies.csv"
plotTreatmentYieldPath = "C:/Users/chris/OneDrive/Documents/RP1/CleanedData/plotsAllData.csv"

yieldYearList = [2004, 2005, 2006, 2007, 2008, 2011, 2012, 2013, 2014]
treatmentList = ['TpRb', 'Fert', 'Seed', 'FYM', 'Rm']

#                                   START OF YIELD CALCULATIONS

# Create a dictionary which describes the different data types in the CSV input file
treatmentYieldDataTypes = {
    'Plot': int,
    'TpRb': int,
    'Fert': int,
    'Seed': int,
    'FYM': int,
    'Rm': int,
    'yield2004': float,
    'yield2005': float,
    'yield2006': float,
    'yield2007': float,
    'yield2008': float,
    'yield2011': float,
    'yield2012': float,
    'yield2013': float,
    'yield2014': float
}

# Read in the treatment and yield data CSV file with appropriate data types
treatmentYieldData = pd.read_csv(plotTreatmentYieldPath, dtype=treatmentYieldDataTypes)

# Create dictionaries for storing the mean, SD, and stability values for each plot across all 9 recording years
treatmentYieldMeans = {}
treatmentYieldSD = {}
plotStability = {}

# Fill those dictionaries with appropriate values from the yield CSV file
for index, row in treatmentYieldData.iterrows():

    plotNumber = row['Plot']

    plotYieldList = []

    for eachYear in yieldYearList:
        plotYieldList.append(row['yield' + str(eachYear)])

    treatmentYieldMeans[plotNumber] = sum(x for x in plotYieldList) / len(plotYieldList)

    treatmentYieldSD[plotNumber] = statistics.stdev(x for x in plotYieldList)

    # Create the plot stability dictionary (plotStability) for each individual plot
    # The keys are the individual plot numbers and the values are the yield stability values
    plotStability[plotNumber] = treatmentYieldMeans.get(plotNumber) / treatmentYieldSD.get(plotNumber)

# Identify and group (in threes) those plots which share the same treatment combinations
plotGroups = {}

# Iterate through the treatment/yield CSV file and extract which plot numbers have which treatments
for index, row in treatmentYieldData.iterrows():

    plotNumber = row['Plot']

    plotTreatmentValueList = []

    duplicate = 0

    for eachTreatment in treatmentList:
        plotTreatmentValueList.append(row[eachTreatment])

    plotTreatmentValueStr = ','.join(str(x) for x in plotTreatmentValueList)

    if not plotGroups:
        plotGroups[plotTreatmentValueStr] = plotNumber

    else:

        for eachItem in plotGroups.items():

            dictTreatmentValues = eachItem[0]
            dictPlotGroup = eachItem[1]

            if plotTreatmentValueStr == dictTreatmentValues:
                keyToChange = dictTreatmentValues
                valueToChange = dictPlotGroup

                duplicate += 1

        if duplicate == 1:

            plotGroups[keyToChange] = str(valueToChange) + ',' + str(plotNumber)

        else:

            # Create dictionary filled with treatment combinations (1's and 0's) as keys
            # and a list of plot numbers for that treatment as values
            plotGroups[plotTreatmentValueStr] = plotNumber

# Group all of the yield stability values based on their plot number to sort into treatment groups

masterDictionary = {}
dataMasterList = []

for eachGroupItem in plotGroups.items():

    # Iterate through the dictionary created immediately above which identifies plot groups based on treatment
    plotGroupTreatment = eachGroupItem[0].split(',')
    plotGroupPlots = eachGroupItem[1].split(',')

    plotStabilityValueList = []
    plotStabilityPlotList = []

    # Iterate through the dictionary of plot numbers and yield stability values, add them to a new dict
    for eachStabilityItem in plotStability.items():

        stabilityPlotNumber = eachStabilityItem[0]
        plotStabilityValue = eachStabilityItem[1]

        if re.search('\'' + re.escape(str(stabilityPlotNumber)) + '\'', str(plotGroupPlots)):
            # print('Found plot: ' + str(stabilityPlotNumber) + ' in the group: ' + str(plotGroupPlots))

            plotStabilityValueList.append(plotStabilityValue)
            plotStabilityPlotList.append(stabilityPlotNumber)

            # Calculate means of each stability value set and add to a new list

            stabilityMean = sum(x for x in plotStabilityValueList) / len(plotStabilityValueList)

    # Add all row data into a single list to be joined to the list of lists afterwards

    dataMasterList.append([plotStabilityPlotList, plotStabilityValueList, plotGroupTreatment, stabilityMean])

# Set header list
# dataMasterList.insert(0, ['plotGroups', 'stabilityValues', 'treatmentValues', 'stabilityMeans'])

# Make a data frame which has the columns: plotNumbers (x3), treatmentCombo (x1), stabilityValues (x3)
masterStabilityTable = pd.DataFrame(dataMasterList)

masterStabilityTable.columns = ['plotGroups', 'stabilityValues', 'treatmentValues', 'stabilityMeans']

masterStabilityTable.to_csv("C:/Users/chris/OneDrive/Documents/RP1/CleanedData/testOutput2.csv", header=True)

# NEXT STEP IS TO DO ANOVA ON EACH TREATMENT GROUP

# Iterate through all of the stability values of each treatment group and compare
kruskalDataList = []

for eachMasterItem in masterStabilityTable['stabilityValues']:
    kruskalDataList.append(eachMasterItem)

# Kruskal-Wallis test performed on the yield stability data finds no significant difference between treatments (p=0.47)
# - chose K-W because there are only 3 data points for each sample group and so normality cannot be assumed
# - really, it is too few repeats to do any meaningful test but this is the only data available
# print(stats.kruskal(*masterStabilityTable['stabilityValues']))

# Graph the box plots of the yield stability values to get a visual comparison of results
boxPlotData = []

for eachMasterItem in masterStabilityTable['stabilityValues']:
    boxPlotData.append(go.Scatter(x=eachMasterItem))

# py.plot(boxPlotData)

xMeanCoord = []
ySDCoord = []

for eachMean in treatmentYieldMeans:

    for eachSD in treatmentYieldSD:

        if eachMean == eachSD:
            xMeanCoord.append(treatmentYieldMeans.get(eachMean))
            ySDCoord.append(treatmentYieldSD.get(eachSD))

# print(xMeanCoord)

# Create graph of the single plots' means and SDs in treatment groups - plot mean vs SD
masterXCoordList = []

masterYCoordList = []

traceList = []

counter = 0

for eachGroupItem in plotGroups.items():

    plotGroupPlots = eachGroupItem[1].split(',')

    tempXCoordList = []
    tempYCoordList = []

    traceName = 'trace' + str(counter)

    for eachMean in treatmentYieldMeans:

        for eachSD in treatmentYieldSD:

            if eachMean == eachSD:

                if re.search('\'' + re.escape(str(eachMean)) + '\'', str(plotGroupPlots)):
                    tempXCoordList.append(treatmentYieldMeans.get(eachMean))
                    tempYCoordList.append(treatmentYieldSD.get(eachSD))

    traceList.append(go.Scatter(x=tempXCoordList, y=tempYCoordList, mode='markers+text'))

    counter += 1

meanSDData = traceList
meanSDLayout = go.Layout(xaxis=dict(title='Yield Mean'), yaxis=dict(title='Yield SD'))
meanSDFig = go.Figure(data=meanSDData, layout=meanSDLayout)
# Graph produces mean vs SD for each of the plots' yield stability values, grouped by their treatment groups
# py.plot(meanSDFig)
#                               END OF YIELD STABILITY CALCULATIONS

#                               START OF RICHNESS CALCULATIONS

speciesCoverData = pd.read_csv(speciesCoverPath)

speciesYearColumn = speciesCoverData['Year']
speciesYearList = np.unique(speciesYearColumn)

speciesPlotColumn = speciesCoverData['Plot']
speciesPlotList = np.unique(speciesPlotColumn)

speciesNameList = speciesCoverData.columns[2:78]
speciesList = speciesCoverData[speciesNameList]

speciesRichnessData = pd.DataFrame()

# Loop through each year of records
for eachSpeciesYear in speciesYearList:

    # Reset the richness list in between each year
    eachYearRichness = []
    eachYearPlot = []

    # Loop through each row of the dataframe
    for eachSpeciesIndex, eachSpeciesRow in speciesCoverData.iterrows():

        eachEntryYear = eachSpeciesRow['Year']
        eachEntryPlot = eachSpeciesRow['Plot']

        # Only consider entries with the year that matches the year of the loop
        if eachEntryYear == eachSpeciesYear:

            # Loop through each plot number for the focused year
            for eachPlotNumber in speciesPlotList:

                # Restart the richness counter in between each plot
                eachPlotRichness = 0

                # Only consider entries with the plot number that matches the plot number of the loop
                if eachPlotNumber == eachEntryPlot:

                    # Loop through each species column for every species in the dataframe
                    for eachSpecies in eachSpeciesRow[speciesNameList]:

                        # If that species cover value is not 0
                        if eachSpecies != 0:
                            # Add 1 to the richness counter
                            eachPlotRichness += 1

                    # Once all species have been considered for this plot, add the total richness counter value to list
                    eachYearRichness.append(eachPlotRichness)
                    eachYearPlot.append(eachPlotNumber)

    # Once all plots in a year have been considered, add the richness list to the dataframe under a year-based header
    speciesRichnessData['plot'] = eachYearPlot
    yearTitle = str(eachSpeciesYear)
    speciesRichnessData[yearTitle] = eachYearRichness

speciesRichnessData = pd.melt(speciesRichnessData, var_name='year', value_name='richness')

# Export so can add the plot numbers in Excel very quickly then read back in
# speciesRichnessData.to_csv("D:/Users/Chris/Documents/MSc Bioinformatics/RP1/CleanedData/richnessOutput.csv",
# header=True)
speciesRichnessData = pd.read_csv("C:/Users/chris/OneDrive/Documents/RP1/CleanedData/richnessInput.csv")

# Get final (2014) richness values in a dictionary where key = plot and value = richness
richness2014 = {}

for eachIndex, eachRow in speciesRichnessData.iterrows():

    if eachRow['year'] == 2014:
        richness2014[eachRow['plot']] = eachRow['richness']


#                               END OF RICHNESS CALCULATIONS

#                               START OF SDI CALCULATIONS


# Use a function to calculate the total cover for each plot
def getTotalCover(speciesDataframe, selectYear, selectPlot):
    coverCounter = 0

    for eachCoverIndex, eachCoverRow in speciesDataframe.iterrows():

        if eachCoverRow['Year'] == selectYear:

            if eachCoverRow['Plot'] == selectPlot:

                for eachIndividualSpecies in eachCoverRow[speciesNameList]:

                    if eachIndividualSpecies != 0:
                        coverCounter += eachIndividualSpecies

    return coverCounter


# print(getTotalCover(speciesCoverData, 2008, 46))

sdi2014 = {}

for eachIndex, eachRow in speciesCoverData.iterrows():

    currentPlotYear = eachRow['Year']

    if currentPlotYear == 2014:

        currentPlotNumber = eachRow['Plot']
        currentPlotTotalCover = getTotalCover(speciesCoverData, currentPlotYear, currentPlotNumber)

        sdiSpeciesCalculationsList = []

        for eachSpeciesCol in eachRow[speciesNameList]:

            if eachSpeciesCol != 0:
                # print('Species value is: ' + str(eachSpeciesCol) + ' and total plot cover is: ' + str(currentPlotTotalCover))

                plotCoverProportion = eachSpeciesCol / currentPlotTotalCover

                sdiSpeciesCalculation = plotCoverProportion * (math.log(plotCoverProportion))

                # print('So SDI species calculation is: ' + str(sdiSpeciesCalculation))

                sdiSpeciesCalculationsList.append(sdiSpeciesCalculation)

        plotSDI = -1 * np.sum(sdiSpeciesCalculationsList)

        sdi2014[currentPlotNumber] = plotSDI

#                               END OF SDI CALCULATIONS

#                               START OF EVENNESS CALCULATIONS

# Evenness is equal to the SDI value divided by the natural log of species richness (total number of species per plot)
# Based on Pielou's evenness calculation

evenness2014 = {}

for eachSDIplot, eachSDIvalue in sdi2014.items():
    evenness2014[eachSDIplot] = sdi2014.get(eachSDIplot) / math.log(richness2014[eachSDIplot])

#                               END OF EVENNESS CALCULATIONS

#                               START OF CORRELATION TESTING

evenData = np.array([a for a in evenness2014.values()])
richData = np.array([b for b in richness2014.values()])
sdiData = np.array([c for c in sdi2014.values()])

# Cheekily write all diversity data to a csv for ease of use in ANOVA testing
# masterDiversityTable = pd.DataFrame({'evenness': evenness2014, 'richness': richness2014, 'sdi': sdi2014})
# masterDiversityTable.to_csv("C:/Users/chris/OneDrive/Documents/RP1/CleanedData/diversityOutput.csv", header=True)

# Test for normality in the three data sets
# print('Evenness normality test result: ' + str(stats.normaltest(evenData)))
# print('Richness normality test result: ' + str(stats.normaltest(richData)))
# print('SDI normality test result: ' + str(stats.normaltest(sdiData)))

# Great success! None of the p-values are below 0.05 so we fail to reject the null hypothesis that these data come
# from a normal distribution (yay)

# Data is normally distributed so can use Pearson's correlation test:
# print('Correlation between evenness and richness : ' + str(stats.pearsonr(evenData, richData)))
# print('Correlation between evenness and SDI : ' + str(stats.pearsonr(evenData, sdiData)))
# print('Correlation between richness and SDI : ' + str(stats.pearsonr(richData, sdiData)))

# And let's see if the results match with a simple scatter plot:
# evenRichPlot = plt.scatter(evenData, richData)
# evenSDIPlot = plt.scatter(evenData, sdiData)
# richSDIPlot = plt.scatter(richData, sdiData)

# plt.show()

# Success again! Sig correlation between richness and SDI (it follows by maths that evenness and SDI are correlated)
# Interesting but not really addressing the question at hand (diversity vs stability!)

# Now to compare the various diversity measures with the yield stability values for each plot

stabilityData = np.array([d for d in plotStability.values()])

# Check for normality again
# print('Stability normality test result: ' + str(stats.normaltest(stabilityData)))

# Significant normality test so reject null hypothesis that the sample is from a normal distribution
# So it is not normally distributed so compare with Spearman's correlation:
# print('Correlation between evenness and stability : ' + str(stats.spearmanr(evenData, stabilityData)))
# print('Correlation between richness and stability : ' + str(stats.spearmanr(richData, stabilityData)))
# print('Correlation between SDI and stability : ' + str(stats.spearmanr(sdiData, stabilityData)))

# Both evenness and SDI are significant so plot:

# plt.scatter(evenData, stabilityData)
# plt.show()
#
# plt.scatter(sdiData, stabilityData)
# plt.show()

#                               END OF CORRELATION TESTING

#                               START OF LINEAR REGRESSIONS

# Calculate linear regression between each pairing
# print('Linear regression between evenness and stability : ' + str(stats.linregress(evenData, stabilityData)))
# print('Linear regression between richness and stability : ' + str(stats.linregress(richData, stabilityData)))
# print('Linear regression between SDI and stability : ' + str(stats.linregress(sdiData, stabilityData)))

# Both evenness and SDI are significant again so plot:
# slope, intercept, r_value, p_value, std_err = stats.linregress(evenData, stabilityData)
# plt.plot(evenData, stabilityData, 'o', label='original data')
# plt.plot(evenData, intercept + slope*evenData, 'r', label='fitted line')
# plt.legend()
# plt.xlabel('Species Evenness')
# plt.ylabel('Yield Stability')
# plt.show()

# slope, intercept, r_value, p_value, std_err = stats.linregress(sdiData, stabilityData)
# plt.plot(sdiData, stabilityData, 'o', label='original data')
# plt.plot(sdiData, intercept + slope*sdiData, 'r', label='fitted line')
# plt.legend()
# plt.xlabel('Shannon Index')
# plt.ylabel('Yield Stability')
# plt.show()

#                               END OF LINEAR REGRESSIONS

#                               START OF FIVE-WAY ANOVA CREATION

treatmentStabilityPath = "C:/Users/chris/OneDrive/Documents/RP1/CleanedData/plotsTreatmentStabilityData.csv"
treatmentStabilityGroupPath = "C:/Users/chris/OneDrive/Documents/RP1/CleanedData/" \
                              "plotsTreatmentStabilityDataWithGroups.xlsx"

treatmentStabilityCSV = pd.read_csv(treatmentStabilityPath)

standardisedVariables = pd.DataFrame(preprocessing.scale(treatmentStabilityCSV[treatmentList]))


# normalizedVariables = pd.DataFrame(preprocessing.normalize(treatmentStabilityCSV[treatmentList]))


def prepData(transformedData, referenceTable):
    transformedData['plot'] = referenceTable['Plot']
    transformedData['stability'] = referenceTable['stability']
    transformedData['richness'] = referenceTable['richness']
    transformedData['evenness'] = referenceTable['evenness']
    transformedData['sdi'] = referenceTable['sdi']
    headerList = []
    headerList.extend(treatmentList)
    headerList.extend(['plot', 'stability', 'richness', 'evenness', 'sdi'])
    transformedData.columns = headerList

    return transformedData


# print(prepData(standardisedVariables, treatmentStabilityCSV))

# print(prepData(normalizedVariables, treatmentStabilityCSV))

# for eachTreatmentName in treatmentList:

#    for eachSecondTreatmentName in treatmentList:

#        if not eachTreatmentName == eachSecondTreatmentName:

#            print(rp.summary_cont(treatmentStabilityCSV.groupby([eachTreatmentName, eachSecondTreatmentName]))
#            ['stability'])


def fiveWayAnova(inputDataframe, userModel):
    model = ols(userModel, inputDataframe).fit()

    # print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, "
    #       f"p = {model.f_pvalue: .4f}")
    #
    # print(model.summary())

    # Create ANOVA table with the type 2 sum of squares
    anovaTable = sm.stats.anova_lm(model, type=2)

    print(anovaTable)


# print(treatmentStabilityCSV)

# print('\n')

# print(standardisedVariables)

formattedAllData = prepData(standardisedVariables, treatmentStabilityCSV)
# print('Normal TpRb col: ' + str(formattedAllData['TpRb']))

appliedDict2 = {}
appliedDict1 = {}
noTreatDict0 = {}

for eachTreat in treatmentList:

    if eachTreat == 'TpRb':

        appliedDict2[eachTreat] = formattedAllData.loc[formattedAllData[str(eachTreat)] > 0]
        appliedDict1[eachTreat] = formattedAllData.loc[formattedAllData[str(eachTreat)] == 0]
        noTreatDict0[eachTreat] = formattedAllData.loc[formattedAllData[str(eachTreat)] < 0]

    else:

        appliedDict1[eachTreat] = formattedAllData.loc[formattedAllData[str(eachTreat)] == 1]
        noTreatDict0[eachTreat] = formattedAllData.loc[formattedAllData[str(eachTreat)] == -1]

# print('Fert 1 richness: ' + str(fert1['richness']))
# print('Fert 0 richness: ' + str(fert0['richness']))

# For individual significant factors
# trace2 = go.Box(y=appliedDict2.get('Seed')['sdi'], name='TpRb added')  # , xbins=dict(size=0.05), opacity=0.65)
trace1 = go.Histogram(x=appliedDict1.get('Seed')['sdi'], name='Seed added', xbins=dict(size=0.05), opacity=0.75)
trace0 = go.Histogram(x=noTreatDict0.get('Seed')['sdi'], name='No seed added', xbins=dict(size=0.05), opacity=0.75)
data = [trace0, trace1]
layout = go.Layout(xaxis=dict(title='SDI'), yaxis=dict(title='Frequency'), barmode='overlay')
figure = go.Figure(data=data, layout=layout)
# py.plot(figure, filename='histoSdiSeed')

combo0 = {}
combo1 = {}
combo2 = {}
combo3 = {}
combo4 = {}
combo5 = {}
combo6 = {}
combo7 = {}

counter = 0

for eachTreat in treatmentList:

    treatment1 = eachTreat

    if treatment1 == 'FYM':

        for eachTreat2 in treatmentList:
            treatment2 = eachTreat2

            if treatment2 == 'Seed':

                # combo0[treatment1, treatment2, treatment3] = formattedAllData.loc[
                #     (formattedAllData[str(treatment1)] == -1) &
                #     (formattedAllData[str(treatment2)] == -1)

                for eachTreat3 in treatmentList:
                    treatment3 = eachTreat3

                    if treatment3 == 'Rm':

                        combo0[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == -1) &
                            (formattedAllData[str(treatment2)] == -1) &
                            (formattedAllData[str(treatment3)] == -1)]

                        combo1[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == 1) &
                            (formattedAllData[str(treatment2)] == -1) &
                            (formattedAllData[str(treatment3)] == -1)]

                        combo2[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == -1) &
                            (formattedAllData[str(treatment2)] == 1) &
                            (formattedAllData[str(treatment3)] == -1)]

                        combo3[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == -1) &
                            (formattedAllData[str(treatment2)] == -1) &
                            (formattedAllData[str(treatment3)] == 1)]

                        combo4[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == 1) &
                            (formattedAllData[str(treatment2)] == 1) &
                            (formattedAllData[str(treatment3)] == -1)]

                        combo5[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == 1) &
                            (formattedAllData[str(treatment2)] == -1) &
                            (formattedAllData[str(treatment3)] == 1)]

                        combo6[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == -1) &
                            (formattedAllData[str(treatment2)] == 1) &
                            (formattedAllData[str(treatment3)] == 1)]

                        combo7[treatment1, treatment2, treatment3] = formattedAllData.loc[
                            (formattedAllData[str(treatment1)] == 1) &
                            (formattedAllData[str(treatment2)] == 1) &
                            (formattedAllData[str(treatment3)] == 1)]

# For interactions between 2 factors

# trace0 = go.Box(x=['None', 'FYM, no Fert', 'Fert, no FYM', 'Fert and FYM'],
#                 y=[statistics.mean(combo0.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo1.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo2.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo3.get(('Fert', 'FYM'))['evenness'])])

# trace0 = go.Box(y=combo0.get(('Fert', 'Seed'))['stability'], name='None')
# trace1 = go.Box(y=combo1.get(('Fert', 'Seed'))['stability'], name='Seed, no Fert')
# trace2 = go.Box(y=combo2.get(('Fert', 'Seed'))['stability'], name='Fert, no Seed')
# trace3 = go.Box(y=combo3.get(('Fert', 'Seed'))['stability'], name='Fert and Seed')
#
# data = [trace0, trace1, trace2, trace3]
# layout = go.Layout(xaxis=dict(title='Treatment Combination'), yaxis=dict(title='Stability'))
# figure = go.Figure(data=data, layout=layout)
# py.plot(figure, filename='comboBoxTreatSeedFertStability')

# For interactions between 3 factors

# trace0 = go.Box(x=['None', 'FYM, no Fert', 'Fert, no FYM', 'Fert and FYM'],
#                 y=[statistics.mean(combo0.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo1.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo2.get(('Fert', 'FYM'))['evenness']),
#                    statistics.mean(combo3.get(('Fert', 'FYM'))['evenness'])])

trace0 = go.Box(y=combo0.get(('FYM', 'Seed', 'Rm'))['evenness'], name='None')
trace1 = go.Box(y=combo1.get(('FYM', 'Seed', 'Rm'))['evenness'], name='FYM')
trace2 = go.Box(y=combo2.get(('FYM', 'Seed', 'Rm'))['evenness'], name='Seed')
trace3 = go.Box(y=combo3.get(('FYM', 'Seed', 'Rm'))['evenness'], name='Rm')
trace4 = go.Box(y=combo4.get(('FYM', 'Seed', 'Rm'))['evenness'], name='FYM, Seed')
trace5 = go.Box(y=combo5.get(('FYM', 'Seed', 'Rm'))['evenness'], name='FYM, Rm')
trace6 = go.Box(y=combo6.get(('FYM', 'Seed', 'Rm'))['evenness'], name='Seed, Rm')
trace7 = go.Box(y=combo7.get(('FYM', 'Seed', 'Rm'))['evenness'], name='FYM, Seed, Rm')

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(xaxis=dict(title='Treatment Combination'), yaxis=dict(title='Evenness'))
figure = go.Figure(data=data, layout=layout)
# py.plot(figure, filename='comboBoxTreatFymSeedRmEvenness')

# Normalized data results in memory error for some reason...? Have to just use standardised
anovaModel = 'stability ~ C(Fert)*C(FYM)*C(TpRb)*C(Seed)*C(Rm)'
# fiveWayAnova(formattedAllData, anovaModel)

# print(standardisedVariables)

#                       END OF FIVE-WAY ANOVA CALCULATIONS

#                       START OF MORE LINEAR REGRESSIONS

# New linear model aiming to answer overall question of best treatment for stability
lmModel = 'stability ~ C(Fert)*C(FYM)*C(TpRb)*C(Seed)*C(Rm)'
treatStabilityLM = smf.ols(formula=lmModel, data=formattedAllData).fit()


# print(treatStabilityLM.summary())
# print(sm.stats.anova_lm(treatStabilityLM, type=2))

# PLOT a scatter of each significant treatment effect (up to 3D for two-ways)
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# x = formattedAllData['FYM']
# y = formattedAllData['Fert']
#
# z = formattedAllData['stability']
#
# ax.scatter(x, y, z)
# # plt.scatter(x, y)
# plt.ylabel('Fert')
# plt.xlabel('FYM')
# plt.zlabel('Fert')
# plt.show()


def linRegress(x, y):
    # Individual linear regressions to see the explained variance from each significant variable from the regression
    print('For the regression of ' + str(x) + ' on ' + str(y))
    print(stats.linregress(formattedAllData[str(x)], formattedAllData[str(y)]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(formattedAllData[str(x)], formattedAllData[str(y)])
    plt.plot(formattedAllData[str(x)], formattedAllData[str(y)], 'o', label='original data')
    plt.plot(formattedAllData[str(x)], intercept + slope * formattedAllData[str(x)], 'r', label='fitted line')
    plt.legend()
    plt.xlabel(str(x))
    plt.ylabel(str(y))
    plt.show()


linRegress('Rm', 'stability')

# treatmentList.remove('Rm')

# for eachTreat in treatmentList:
#
#     linRegress(str(eachTreat), 'stability')

# treatmentList.append('Rm')
#                         END OF MULTIPLE LINEAR REGRESSIONS

#                       START OF SEPARATING FERT AND FYM LEVELS

# fertPresent = treatmentYieldData['Fert']==1

# treatYieldData

# for index, row in treatmentYieldData.iterrows():
#
#     if row['Fert'] == 1:
#
#         # Fert present, FYM absent
#         if row['FYM'] == 0:
#
#         # Fert present, FYM present
#         elif row['FYM'] == 1:
#
#     elif row['Fert'] == 1:
#
#         # Fert absent, FYM absent
#         if row['FYM'] == 0:
#
#         # Fert absent, FYM absent
#         elif row['FYM'] == 1:

#                       END OF SEPARATING FERT AND FYM LEVELS

#                               PCA Calculations


# treatmentStabilityGroupsCSV = pd.read_excel(treatmentStabilityGroupPath, dtype={'treatGroup': str})
#
# treatStabilityGroups = pd.DataFrame(treatmentStabilityGroupsCSV)
#
# treatStabilityGroups = treatStabilityGroups.drop(columns='Plot')
#
# # Separate out the features:
# xFeatures = treatStabilityGroups.iloc[:, 0:6]
#
# # Rescale the features so that all values are between 0 and 1
# rescaler = MinMaxScaler(feature_range=(0, 1))
# rescaledFeatures = rescaler.fit_transform(xFeatures)
#
# # Standardise the features (all treatment columns + stability)
# standardisedFeatures = StandardScaler().fit_transform(rescaledFeatures)
#
# # Normalise the features (rescale each row to have a unit norm of 1)
# normalizer = Normalizer().fit(X=standardisedFeatures)
# X_std = normalizer.transform(standardisedFeatures)
#
# # Separate out the target/classes:
# y = treatStabilityGroups['treatGroup'].values
#
# uniqueClasses = np.unique(y)
#
# # Eigendecomposition
# # Print covariance matrix
# # print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))
#
# # Print eigenvectors and eigenvalues based on the covariance matrix values
# # (would be same result if based off correlation matrix values because data is standardised)
# cov_mat = np.cov(X_std.T)
#
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#
# # print('Eigenvectors \n%s' %eig_vecs)
# # print('\nEigenvalues \n%s' %eig_vals)
#
# # Singular vector decomposition???
# u,s,v = np.linalg.svd(X_std.T)
# # print(u)
#
# # Check that all of the unit norms are equal to zero and so analysis can continue...
# for ev in eig_vecs:
#     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
# # print('Everything ok!')
#
# # Make a list of (eigenvalue, eigenvector) tuples
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#
# # Sort the (eigenvalue, eigenvector) tuples from high to low
# eig_pairs.sort()
# eig_pairs.reverse()
#
# # Visually confirm that the list is correctly sorted by decreasing eigenvalues
# # print('Eigenvalues in descending order:')
# # for i in eig_pairs:
# #    print(i[0])
#
# # Visually plot the contribution of each principal component to determine how many I need
# tot = sum(eig_vals)
# var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# trace1 = dict(
#     type='bar',
#     x=['PC %s' % i for i in range(1, 7)],
#     y=var_exp,
#     name='Individual'
# )
#
# trace2 = dict(
#     type='scatter',
#     x=['PC %s' % i for i in range(1, 7)],
#     y=cum_var_exp,
#     name='Cumulative'
# )
#
# data = [trace1, trace2]
#
# layout = dict(
#     title='Explained variance by different principal components',
#     yaxis=dict(
#         title='Explained variance in percent'
#     ),
#     annotations=list([
#         dict(
#             x=1.16,
#             y=1.05,
#             xref='paper',
#             yref='paper',
#             text='Explained Variance',
#             showarrow=False,
#         )
#     ])
# )
#
# fig = dict(data=data, layout=layout)
# # py.plot(fig, filename='selecting-principal-components')
#
# # Reduce from 6-dimensions to 3:
# matrix_w = np.hstack((eig_pairs[0][1].reshape(6, 1),
#                       eig_pairs[1][1].reshape(6, 1)))
#                       # ,eig_pairs[2][1].reshape(6, 1)))
#
# # print('Matrix W:\n', matrix_w)
#
# # Create actual final chart:
#
#
# def get_spaced_colors(n):
#     max_value = 16581375  # 255**3
#     interval = int(max_value / n)
#     colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
#
#     return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
#
#
# Y = X_std.dot(matrix_w)
# # print(len(Y))
# data = []
#
# colourSet = get_spaced_colors(48)
#
# for name, col in zip(uniqueClasses, colourSet):
#     # print('Name: ' + str(name) + ' Colour: ' + str(col))
#
#     trace = dict(
#         type='scatter',  # 3d
#         x=Y[y == name, 0],
#         y=Y[y == name, 1],
#         # z=Y[y == name, 2],
#         mode='markers',
#         name=name,
#         marker=dict(
#             color=col,
#             size=12,
#             line=dict(
#                 color='rgba(217, 217, 217, 0.14)',
#                 width=0.5),
#             opacity=0.8)
#     )
#     data.append(trace)
#
# layout = dict(
#     showlegend=True,
#     scene=dict(
#         xaxis=dict(title='PC1'),
#         yaxis=dict(title='PC2')  # ,
#         # zaxis=dict(title='PC3')
#     )
# )
#
# fig = dict(data=data, layout=layout)
# # py.plot(fig, filename='2d-projection-matrix')

#                                       MEAN YIELD VS YEAR GRAPHING START
#
treatMeanYieldByYear = {}

for eachYear in yieldYearList:

    treatMeanYield = {}

    for eachItem in plotGroups.items():

        plotGroupTreats = eachItem[0]

        plotGroupPlots = eachItem[1].split(',')

        tempYearTreatYieldList = []

        for index, row in treatmentYieldData.iterrows():

            plotNumber = row['Plot']

            thisYearPlotYield = row['yield' + str(eachYear)]

            if re.search('\'' + re.escape(str(plotNumber)) + '\'', str(plotGroupPlots)):
                tempYearTreatYieldList.append(thisYearPlotYield)

        # print('Treatment: ' + str(plotGroupTreats) + ' Year: ' + str(eachYear) + ' Values: ' + str(tempYearTreatYieldList))

        meanForTreatmentYear = sum(x for x in tempYearTreatYieldList) / len(tempYearTreatYieldList)

        treatMeanYield[plotGroupTreats] = meanForTreatmentYear

    treatMeanYieldByYear[eachYear] = treatMeanYield

totalYearMeans = {}

for eachYear in yieldYearList:

    tempYearYieldList = []

    for index, row in treatmentYieldData.iterrows():
        thisYearYield = row['yield' + str(eachYear)]

        tempYearYieldList.append(thisYearYield)

    yearMean = sum(x for x in tempYearYieldList) / len(tempYearYieldList)

    totalYearMeans[eachYear] = yearMean

treatMeanYieldDf = pd.DataFrame(data=treatMeanYieldByYear)

traceList = []

# Add all of the lower opacity lines for the individual treatment means each year
for index, row in treatMeanYieldDf.iterrows():
    yValues = row[yieldYearList]

    traceList.append(
        go.Scatter(x=yieldYearList, y=yValues, name=index, mode='lines', opacity=0.4, line=dict(width=0.5)))

yValues = [x for x in totalYearMeans.values()]
traceList.append(
    go.Scatter(x=yieldYearList, y=yValues, name='Total mean', mode='lines', line=dict(width=2, color='blue')))

# data = traceList
# layout = go.Layout(xaxis=dict(title='Year'), yaxis=dict(title='Mean Treatment Yield'))
# fig = dict(data=data, layout=layout)
# py.plot(fig, filename='meanYieldPerTreatmentOverTime')

#                                       MEAN YIELD VS YEAR GRAPHING END

#                              RUNNING YIELD STABILITY VS DIVERSITY GRAPHING START
newYieldYearList = [2005, 2006, 2007, 2008, 2011, 2012, 2013, 2014]

runningStability = {}

for eachYear in newYieldYearList:

    yearStabilityValues = {}

    for eachItem in plotGroups.items():

        plotGroupTreats = eachItem[0]
        plotGroupPlots = eachItem[1].split(',')

        tempYieldList = []

        for index, row in treatmentYieldData.iterrows():

            plotNumber = row['Plot']
            # print(plotNumber)
            thisYearPlotYield = row['yield' + str(eachYear)]

            if eachYear == 2011:

                lastYearPlotYield = row['yield2008']
                # print('2011 only' + str(lastYearPlotYield))

            else:

                lastYearPlotYield = row['yield' + str(eachYear - 1)]
                # print(lastYearPlotYield)

            if re.search('\'' + re.escape(str(plotNumber)) + '\'', str(plotGroupPlots)):
                tempYieldList.append(thisYearPlotYield)
                tempYieldList.append(lastYearPlotYield)

        treatMean = sum(x for x in tempYieldList) / len(tempYieldList)
        treatSD = statistics.stdev(x for x in tempYieldList)

        treatStability = treatMean / treatSD

        yearStabilityValues[plotGroupTreats] = treatStability

    runningStability[eachYear] = yearStabilityValues

# print(runningStability)

runningStabilityFrame = pd.DataFrame(data=runningStability)

traceList2 = []
for index, row in runningStabilityFrame.iterrows():
    yValues = row[newYieldYearList]

    traceList2.append(go.Scatter(x=newYieldYearList, y=yValues, name=index, mode='lines', line=dict(width=0.7)))

# data = traceList2
# layout = go.Layout(xaxis=dict(title='Year'), yaxis=dict(title='Running Stability'))
# fig = dict(data=data, layout=layout)
# py.plot(fig, filename='runningStabilityPerTreatmentOverTime')
# Want to plot this above with perhaps another moving mean to show the general trend in stabilities?
