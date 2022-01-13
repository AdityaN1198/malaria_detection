import streamlit as st
import os
from models.util import predictions
from tempfile import NamedTemporaryFile
from PIL import UnidentifiedImageError

st.title('IQGateway Malaria Detection App')

st.write('This is an app that tells if a cell is infected with malaria or not. You can upload the photo of a cell below. It also show the infected areas that the image uses to indentify the infection')

# Form to submit imagew
with st.form("my-form", clear_on_submit=True):
    img = st.file_uploader(label='Cell Image',type=['png','jpg','jpeg'],accept_multiple_files=False)
    submitted = st.form_submit_button("UPLOAD!")

# Make a temperary file to save image path
temp_file = NamedTemporaryFile(delete=False)

# check if the image if uploaded
if img:

    # Save the image to a temporary file
    temp_file.write(img.getvalue())
    st.image(img)

    # Try except block to catch error in case temporary file is not loaded
    try:
        pred,confidence,grad_img =  predictions(temp_file.name)
        if pred[0][0] == 1.:
            pred = "Infected"
            confidence =  confidence[0][0]*100
        elif pred[0][0] == 0:
            pred = 'Uninfected'
            confidence =  (100-confidence[0][0])
        st.write('The image is predicted to be {}, with confidence of {:.2f} percent'.format(pred,confidence))

        st.write('The predected areas in the image are highlighted in red below for infected cell, and if the cell is not infected then no area is highlighted in red')

        # Show heatmap superimposed image
        st.image(grad_img)

        # Remove the temporary file
        os.remove(grad_img)
    
    # Catch the UnidentifiedImageErorr
    except UnidentifiedImageError as e:
        st.write('image was not processed, please reupload the image')


