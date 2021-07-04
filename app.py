import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import glob

st.image("assets/logo.png")
st.text("H2020-MSCA-ITN-2017")
st.title("AiPBAND")
st.header("An Integrated Platform for Developing Brain Cancer Diagnostic Techniques")

st.text("Starting date of the project: 01/01/2018\nDuration: 48 months")

st.title("Deliverable 5.3")
st.subheader("Report on developmentof cloud-based gliomasdiagnostics system")
st.write()

toc=[
    "Executive summary",
    "Process"
]
current = toc[0]


st.sidebar.multiselect("Sections.", options=toc)

t_diss = "Dissemination level"
t_conf = "Confidential, only for members of the consortium (including the Commission Services)"
t_class = "Classified, information as referred to in Commission Decision 2001/844/EC"

st.markdown("Dissemination  level || \n\
           | --- | --- | --- |\n\
           | PU | Public | _ | \n\
           | CO | {} | _ |\n\
           | CI | {} | _ |".format(t_conf, t_class))

st.write("This project has received funding from the European \
Union's Horizon 2020 research and innovation programme  under  grant  agreement  No  764281. ")