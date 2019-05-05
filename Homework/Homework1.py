import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Show all columns and do not truncate in a DF
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

# Set random seed with NumPy
np.random.seed(1)

## Setting Seaborn Styles
sns.set(style="whitegrid")
sns.set_palette((sns.color_palette('colorblind', 8)))
dims = (11.7, 8.27)

# Create a list of column names to apply to df
columnNames = ['Row Id', 'Row Id.', 'Region1', 'Region2', 'Region3', 'Region4', 'HomeOwner', 'NumChild', 'Income',
               'Gender', 'Wealth', 'HomeVal', 'MedIncome', 'AvgIncome', 'LowIncomePercent', 'NumProm', 'TotalGift',
               'MaxGift', 'LastGift', 'MonthsSinceLast', 'TimeBetween1And2', 'AvgGift', 'PredDonor', 'PredAmount']

# Read in csv to DF
df = pd.read_csv('data\\donors_data.csv', names=columnNames, skiprows=1)

# Drop useless columns
dropCols = ['Row Id', 'Row Id.', 'Income', 'PredDonor', 'PredAmount']
df.drop(labels=dropCols, axis=1, inplace=True)

# Check if any donor is in two regions
checkDonorRegion1 = df[df['Region1'] == 1]
checkDonorRegion2 = df[df['Region2'] == 1]
checkDonorRegion3 = df[df['Region3'] == 1]
checkDonorRegion4 = df[df['Region4'] == 1]
checkDonorRegion5 = df[(df['Region1'] == 0) & (df['Region2'] == 0) & (df['Region3'] == 0) & (df['Region4'] == 0)]

# Function to flatten the region columns
def combine_region(row):
    if row['Region1'] == 1:
        val = 1
    elif row['Region2'] == 1:
        val = 2
    elif row['Region3'] == 1:
        val = 3
    elif row['Region4'] == 1:
        val = 4
    else:
        val = 5
    return val

# Apply the combine_region function
df['Region'] = df.apply(combine_region, axis=1)

# Get number of donors in each region
# print(len(checkDonorRegion1)) #669
# print(len(checkDonorRegion2)) #578
# print(len(checkDonorRegion3)) #669
# print(len(checkDonorRegion4)) #1200
# print(len(checkDonorRegion5)) #4

# Combine all the lengths and divide by total length
checkDonorAllRegions = (len(checkDonorRegion1) + len(checkDonorRegion2) + len(checkDonorRegion3) +
                        len(checkDonorRegion4) + len(checkDonorRegion5)) / len(df)

# This should equal out to 1.0
# print(checkDonorAllRegions)
# = 1.0

# Create data frames with just male and female donors
male_donors = df[df['Gender'] == 0]
female_donors = df[df['Gender'] == 1]

# Get lengths of each DF
# print(len(male_donors)) # 1219
# print(len(female_donors)) # 1901

# Male donor dist charts
male_income_fig, male_income_ax = plt.subplots(figsize=dims)
male_income_ax.set(ylim=(0,225))
male_income_plot = sns.distplot(male_donors['AvgIncome'], kde=False)
male_income_fig.suptitle('Male Donors Avg Income')
male_income_figs = male_income_plot.get_figure()
male_income_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Male_Income.png')

male_wealth_fig, male_wealth_ax = plt.subplots(figsize=dims)
male_wealth_ax.set(ylim=(0,1200))
male_wealth_plot = sns.distplot(male_donors['Wealth'], kde=False, bins=7)
male_wealth_fig.suptitle('Male Donors Wealth Rating')
male_wealth_figs = male_wealth_plot.get_figure()
male_wealth_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Male_Wealth.png')

male_home_fig, male_home_ax = plt.subplots(figsize=dims)
male_home_ax.set(ylim=(0,275))
male_home_plot = sns.distplot(male_donors['HomeVal'], kde=False)
male_home_fig.suptitle('Male Donors Home Value')
male_home_figs = male_home_plot.get_figure()
male_home_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Male_Home.png')

# Female donor dist charts
female_income_fig, female_income_ax = plt.subplots(figsize=dims)
female_income_ax.set(ylim=(0,225))
female_income_plot = sns.distplot(female_donors['AvgIncome'], kde=False)
female_income_fig.suptitle('Female Donors Avg Income')
female_income_figs = female_income_plot.get_figure()
female_income_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Female_Income.png')

female_wealth_fig, female_wealth_ax = plt.subplots(figsize=dims)
female_wealth_ax.set(ylim=(0,1200))
female_wealth_plot = sns.distplot(female_donors['Wealth'], kde=False, bins=7)
female_wealth_fig.suptitle('Female Donors Wealth Rating')
female_wealth_figs = female_wealth_plot.get_figure()
female_wealth_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Female_Wealth.png')

female_home_fig, female_home_ax = plt.subplots(figsize=dims)
female_home_ax.set(ylim=(0,275))
female_home_plot = sns.distplot(female_donors['HomeVal'], kde=False)
female_home_fig.suptitle('Female Donors Home Value')
female_home_figs = female_home_plot.get_figure()
female_home_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Female_Home.png')


# Show distribution of donation by region
region_donation_fig_1, region_donation_ax_1 = plt.subplots(figsize=dims)
region_donation_plot_1 = sns.distplot(checkDonorRegion1['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_1.suptitle('Region 1 Total Gift')
region_donation_fig_1s = region_donation_plot_1.get_figure()
region_donation_fig_1s.savefig('C:\\Git\\IST652\\Homework\\media\\Region1_Total.png')

region_donation_fig_2, region_donation_ax_2 = plt.subplots(figsize=dims)
region_donation_plot_2 = sns.distplot(checkDonorRegion2['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_2.suptitle('Region 2 Total Gift')
region_donation_fig_2s = region_donation_plot_2.get_figure()
region_donation_fig_2s.savefig('C:\\Git\\IST652\\Homework\\media\\Region2_Total.png')

region_donation_fig_3, region_donation_ax_3 = plt.subplots(figsize=dims)
region_donation_plot_3 = sns.distplot(checkDonorRegion3['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_3.suptitle('Region 3 Total Gift')
region_donation_fig_3s = region_donation_plot_3.get_figure()
region_donation_fig_3s.savefig('C:\\Git\\IST652\\Homework\\media\\Region3_Total.png')

region_donation_fig_3a, region_donation_ax_3a = plt.subplots(figsize=dims)
region_donation_ax_3a.set(xlim=(0,1500))
region_donation_plot_3a = sns.distplot(checkDonorRegion3['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_3a.suptitle('Region 3 Total Gift')
region_donation_fig_3as = region_donation_plot_3a.get_figure()
region_donation_fig_3as.savefig('C:\\Git\\IST652\\Homework\\media\\Region3a_Total.png')

region_donation_fig_4, region_donation_ax_4 = plt.subplots(figsize=dims)
region_donation_plot_4 = sns.distplot(checkDonorRegion4['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_4.suptitle('Region 4 Total Gift')
region_donation_fig_4s = region_donation_plot_4.get_figure()
region_donation_fig_4s.savefig('C:\\Git\\IST652\\Homework\\media\\Region4_Total.png')

region_donation_fig_5, region_donation_ax_5 = plt.subplots(figsize=dims)
region_donation_plot_5 = sns.distplot(checkDonorRegion5['TotalGift'], norm_hist=True, kde=False)
region_donation_fig_5.suptitle('Region 5 Total Gift')
region_donation_fig_5s = region_donation_plot_5.get_figure()
region_donation_fig_5s.savefig('C:\\Git\\IST652\\Homework\\media\\Region5_Total.png')

# Plot number of promos received by number of gifts given
promo_given_fig, promo_given_ax = plt.subplots(figsize=dims)
promo_given_plot = sns.regplot(x='AvgGift', y='NumProm', data=df, scatter=True, fit_reg=False, scatter_kws={'alpha': 0.25})
promo_given_fig.suptitle('Number of Promotions Sent by Average Gift Amount')
promo_given_ax.set_ylabel('Promotions')
promo_given_ax.set_xlabel('Average Gift Amount (USD)')
promo_given_figs = promo_given_plot.get_figure()
promo_given_fig.savefig('C:\\Git\\IST652\\Homework\\media\\Promo_Given.png')

# Plot number of promos received by months since last
promo_since_fig, promo_since_ax = plt.subplots(figsize=dims)
promo_since_plot = sns.regplot(x='MonthsSinceLast', y='NumProm', data=df, scatter=True, fit_reg=True, scatter_kws={'alpha': 0.25})
promo_since_fig.suptitle('Number of Promotions Sent by Months Since Last Gift')
promo_since_ax.set_ylabel('Promotions')
promo_since_ax.set_xlabel('Months Since Last Gift')
promo_since_figs = promo_since_plot.get_figure()
promo_since_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Promo_TimeSinceLast.png')

# Plot number of promos received by months between 1st and 2nd
promo_between_fig, promo_between_ax = plt.subplots(figsize=dims)
promo_between_plot = sns.lmplot(x='TimeBetween1And2', y='NumProm', hue='Region', data=df, scatter=True, fit_reg=False,
                                scatter_kws={'alpha': 0.25})
promo_between_plot.set_titles('Number of Promotions Sent by Time Between 1st and 2nd Gift')
promo_between_plot.set_axis_labels('Time Between 1st and 2nd Gift (Months)', 'Promotions')
promo_between_plot.savefig('C:\\Git\\IST652\\Homework\\media\\Promo_TimeBetween.png')


# Creating a copy of the df for the correlation matrix
# Dropping some columns as well
corr_df = df.copy()
corr_df.drop('Region1', axis=1, inplace=True)
corr_df.drop('Region2', axis=1, inplace=True)
corr_df.drop('Region3', axis=1, inplace=True)
corr_df.drop('Region4', axis=1, inplace=True)
corr_df.drop('TimeBetween1And2', axis=1, inplace=True)

# Creating the correlation df and rounds to 2 decimal places
correlation = corr_df.corr().round(2)

# Creating the mask so it's halved.
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

corr_fig, corr_ax = plt.subplots(figsize=dims)
corr_plot = sns.heatmap(data=correlation, mask=mask, vmax=1, cmap='Set2', square=True, linewidths=1,
                        annot=True, cbar=False)
corr_fig.suptitle('Correlation Matrix')
corr_figs = corr_plot.get_figure()
corr_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Correlation_Matrix.png')

# Create copy df for clustering
cluster_validate = df[['Wealth']]
cluster_df = df[['HomeVal', 'MaxGift', 'TotalGift']]
cluster_validate3 = df[['Wealth']]
cluster_df3 = df[['HomeVal', 'MaxGift', 'TotalGift']]

# Splitting the data into test/train
clus_train, clus_test = train_test_split(cluster_df, test_size=.3, random_state=1)
clusters=range(1,10)
meandist=[]

# Create an elbow chart for determining how many Ks
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    / clus_train.shape[0])
elbow_fig, elbow_ax = plt.subplots(figsize=dims)
elbow_plot = plt.plot(clusters, meandist)
elbow_fig.suptitle('Finding K')
elbow_ax.set_ylabel('Average distance')
elbow_ax.set_xlabel('Number of clusters')
plt.savefig('C:\\Git\\IST652\\Homework\\media\\Finding_K.png')

# Should be about 3 according to the elbow plot, but I know there should be 9
# # KMeans Clustering
# kmeans3 = KMeans(n_clusters=3)
# kmeans3_fit = pd.Series(kmeans3.fit_predict(cluster_df3))
#
# # Combine the predicted and wealth columns into a single DF
# cluster_validate3.insert(loc=0, column='Cluster', value=kmeans3_fit)
#
# # Calculate accuracy
# print(len(cluster_validate3[cluster_validate3['Cluster'] == cluster_validate3['Wealth']]) / len(cluster_validate3))
#
# # Plot the clusters that were found by KMeans
# kmc3_fig, kmc3_ax = plt.subplots(figsize=dims)
# kmc3_ax.set(ylim=(0,200))
# kmc3_plot = plt.scatter(cluster_df3['HomeVal'], cluster_df3['MaxGift'], c=kmeans3.labels_.astype(float), s=50, alpha=0.25)
# kmc3_ax.set_ylabel('Home Value')
# kmc3_ax.set_xlabel('Max Single Gift Value')
# kmc3_figs = kmc3_plot.get_figure()
# kmc3_figs.savefig('C:\\Git\\IST652\\Homework\\media\\KMeans_3.png')

# Redo KMeans with 9 clusters
# # KMeans Clustering
kmeans = KMeans(n_clusters=9)
kmeans_fit = pd.Series(kmeans.fit_predict(cluster_df))

# Combine the predicted and wealth columns into a single DF
cluster_validate.insert(loc=0, column='Cluster', value=kmeans_fit)

# Calculate accuracy
print(len(cluster_validate[cluster_validate['Cluster'] == cluster_validate['Wealth']]) / len(cluster_validate))

# Plot the clusters that were found by KMeans
kmc_fig, kmc_ax = plt.subplots(figsize=dims)
kmc_ax.set(ylim=(0,200))
kmc_plot = plt.scatter(cluster_df['HomeVal'], cluster_df['MaxGift'], c=kmeans.labels_.astype(float), s=50, alpha=0.25)
kmc_ax.set_ylabel('Home Value')
kmc_ax.set_xlabel('Max Single Gift Value')
kmc_figs = kmc_plot.get_figure()
kmc_figs.savefig('C:\\Git\\IST652\\Homework\\media\\KMeans_9.png')

# Plot the wealth level clusters normally
wealth_fig, wealth_ax = plt.subplots(figsize=dims)
wealth_ax.set(ylim=(0,200))
wealth_plot = plt.scatter(df['HomeVal'], df['MaxGift'], c=df['Wealth'], s=50, alpha=0.25)
wealth_ax.set_ylabel('Home Value')
wealth_ax.set_xlabel('Max Single Gift Value')
wealth_figs = wealth_plot.get_figure()
wealth_figs.savefig('C:\\Git\\IST652\\Homework\\media\\Wealth.png')