import streamlit as st

# Add a sidebar checkbox for the theme toggle
theme = st.sidebar.checkbox("Dark Mode")

# Set the theme based on the checkbox value
if theme:
    st.markdown(
        """
        <style>
        body {
            color: white;
            background-color: #1E1E1E;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            color: black;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add your Streamlit app code here
# ...
