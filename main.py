"""
Amazon Nova 2 Omni Streamlit Demo
"""
import streamlit as st
import sys
sys.path.append('./src')

from src.streamlit_ui import main

st.set_page_config(
    page_title="Amazon Nova 2 Omni 데모",
    page_icon="🤖",
    layout="wide"
)

if __name__ == "__main__":
    main()
