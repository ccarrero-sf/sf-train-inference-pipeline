# Import python packages
import streamlit as st
from snowflake.snowpark import Session
from snowflake.ml.registry import registry
import json
from datetime import datetime
import pandas as pd

# Connect to Snowflake

from snowflake.snowpark.context import get_active_session
session = get_active_session()

# Create model registry and add to session state
model_registry = registry.Registry(session=session, database_name=session.get_current_database(), schema_name=session.get_current_schema())

if 'model_registry' not in st.session_state:
    st.session_state.model_registry = model_registry

if 'session' not in st.session_state:
    st.session_state.session = session

# Small intro
st.title("Insurance ML Pipeline")
st.write(f"This Streamlit app allows the user to view various aspects of the ML pipeline built for the insurance dataset.")


gold_df = session.table('INSURANCE_GOLD').limit(600).to_pandas()

# Create a scatterplot with 'PREDICTED_CHARGES' on the x-axis and 'CHARGES' on the y-axis
st.subheader('Scatterplot of Predicted vs Actual Charges')
st.scatter_chart(gold_df, x='PREDICTED_CHARGES', y='CHARGES')
