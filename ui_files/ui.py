import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import inflect

def get_test_size():
    return st.sidebar.slider(
        "Set Test Size",
        min_value = 0.10,
        max_value = 0.50,
        step = 0.10,
        value = 0.30
    )

def get_random_state():
    return st.sidebar.number_input(
        "Set Random State",
        min_value = 0,
        max_value = 42,
        value = 41
    )

def get_features(data):
    attributes = list(data.columns.drop(data.columns[-1]))
    input = []
    
    for col in attributes:
        with st.sidebar.expander(col, False):
            input.append(
                st.number_input("Set " + col)
            )
    
    return input

def plot_corr_feat(df_corr):
    df_corr = df_corr.round(2)
    fig_corr = go.Figure([go.Heatmap(z=df_corr.values,
                                    x=df_corr.index.values,
                                    y=df_corr.columns.values,
                                    # text=df_corr.values,
                                    # texttemplate="%{text}", 
                                    # textfont={"size":12}
                                    )])
    fig_corr.update_layout(height=350,
                        width=1000,
                        margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
    return fig_corr

def plot_confusion(cm):
    cm = pd.DataFrame(cm, columns=["0", "1"])
    cm.index = ["1", "0"]
    fig_corr = go.Figure([go.Heatmap(z=cm.values,
                                    x=cm.index.values,
                                    y=cm.columns.values,
                                    # text=cm.values,
                                    # texttemplate="%{text}", 
                                    # textfont={"size":14}
                                    )])
    fig_corr.update_layout(height=300,
                        width=500,
                        margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
    
    return fig_corr