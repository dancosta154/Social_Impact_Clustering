import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

####### Load Dataset #####################

df = pd.read_csv('KMeans5_allClusters.csv')

########################################################

st.set_page_config(layout="wide")

st.markdown("## Economic Resiliency - Project Five")   # Main Title

################# Scatter Chart Logic #################

## This mostly works

# st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Clusters :")

# columns = df.drop(columns='date').columns.tolist()

# x_axis = st.sidebar.selectbox("X-Axis", columns)
# y_axis = st.sidebar.selectbox("Y-Axis", columns, index=1)

# if x_axis and y_axis:
    # scatter_fig = plt.figure(figsize=(6,4))

#     scatter_ax = scatter_fig.add_subplot(111)

#     cluster_zero_df = df[df["Cluster"] == 0]
#     cluster_one_df = df[df["Cluster"] == 1]
#     cluster_two_df = df[df["Cluster"] == 2]
#     cluster_three_df = df[df["Cluster"] == 3]
#     cluster_four_df = df[df["Cluster"] == 4]

#     cluster_zero_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label=0)
#     cluster_one_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="green", alpha=0.6, ax=scatter_ax, label=1)
#     cluster_two_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="yellow", alpha=0.6, ax=scatter_ax, label=2)
#     cluster_three_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="black", alpha=0.6, ax=scatter_ax, label=3)
#     cluster_four_df.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
#                            title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label=4);


########## Bar Chart Logic ##################

st.sidebar.markdown("### Bar Chart: Distribution of Variables : ")

df_2018_clusters = pd.read_csv('2018_with_clusters.csv')
df_2018_clusters_df = df_2018_clusters.groupby(by='clusters-5').agg('mean').T.iloc[11:-2,::]

columns = df_2018_clusters.columns.tolist()

# scatter_fig = plt.figure(figsize=(6,4))

bar_axis = st.sidebar.multiselect(label="Select Cluster",
                                  options=columns
                                  # default=["Cluster","State",]
                                 )

if bar_axis:
    bar_fig = plt.figure(figsize=(8,6))

    bar_ax = bar_fig.add_subplot(111)

    df_2018_clusters_df = df_2018_clusters_df[bar_axis]

    df_2018_clusters_df.plot.barh(alpha=0.8, ax=bar_ax, title="More Words");

else:
    bar_fig = plt.figure(figsize=(8,6))

    bar_ax = bar_fig.add_subplot(111)

    # df_2018_clusters_df = df_2018_clusters_df[["Cluster","State",]]

    df_2018_clusters_df.plot.barh(alpha=0.8, ax=bar_ax, title="Economic Industry Breakdown per Cluster");

################# Histogram Logic ########################

st.sidebar.markdown("### Histogram: Explore Distribution of Clusters : ")

hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=columns, ) # default=["mean radius", "mean texture"]
bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)

if hist_axis:
    hist_fig = plt.figure(figsize=(6,4))

    hist_ax = hist_fig.add_subplot(111)

    df_2018_clusters_df = df_2018_clusters_df[hist_axis]

    df_2018_clusters_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution");
else:
    hist_fig = plt.figure(figsize=(6,4))

    hist_ax = hist_fig.add_subplot(111)

    # df_2018_clusters_df = df_2018_clusters_df[["mean radius", "mean texture"]]

    df_2018_clusters_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution");

#################### Hexbin Chart Logic ##################################

## Haven't tried this one yet

# st.sidebar.markdown("### Hexbin Chart: Explore Concentration of Measurements :")

# hexbin_x_axis = st.sidebar.selectbox("Hexbin-X-Axis", measurements, index=0)
# hexbin_y_axis = st.sidebar.selectbox("Hexbin-Y-Axis", measurements, index=1)

# if hexbin_x_axis and hexbin_y_axis:
#     hexbin_fig = plt.figure(figsize=(6,4))

#     hexbin_ax = hexbin_fig.add_subplot(111)

#     breast_cancer_df.plot.hexbin(x=hexbin_x_axis, y=hexbin_y_axis,
#                                  reduce_C_function=np.mean,
#                                  gridsize=25,
#                                  #cmap="Greens",
#                                  ax=hexbin_ax, title="Concentration of Measurements");

##################### Layout Application ##################

container1 = st.container() # These control where on the page the graph displays
col1, col2 = st.columns(2) # These control where on the page the graph displays

with container1:
    with col1:
        bar_fig
    # with col2:
    #     scatter_fig


container2 = st.container() # These control where on the page the graph displays
col3, col4 = st.columns(2) # These control where on the page the graph displays

with container2:
    with col3:
        hist_fig
    with col4:
        hexbin_fig
