import streamlit_app as st
from st_pages import Page, show_pages, add_page_title
from PIL import Image
import config as cfg

logo = Image.open(cfg.FILEPATH_LOGO_JPG)

st.set_page_config(page_title="ML Home", page_icon=logo)

show_pages(
    [
        Page("index.py", "Home"),
        Page("st_data.py", "Data viewer"),
        Page("st_preprocessing.py", "Pre-processing"),
        Page("st_classification.py", "Classification"),
    ]
)

col = st.columns([1, 4])
with col[0]:
    st.image(logo, width=100)
with col[1]:
    st.title("ML Home")

"## Data Processing Toolkit"

st.markdown('- #### <a href="Data viewer" target="_self">Data viewer</a>  \n View & explore the data.', unsafe_allow_html=True)
st.markdown('- #### <a href="Pre-processing" target="_self">Preprocessing</a>  \n Apply smoothing, filtering, baseline correction and other preprocessing steps.', unsafe_allow_html=True)
st.markdown('- #### <a href="Classification" target="_self">Classification</a>  \n Train a classification model using basic machine learning methods. Evaluate class prediction performance using cross-validation.', unsafe_allow_html=True)



