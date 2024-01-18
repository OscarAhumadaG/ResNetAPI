import streamlit as st
import requests


def main():
    st.title("ResNest Classifier")

    uploaded_file = st.file_uploader("Please select an image: ")
    
    btn_classify = st.button("Classify")
    
    # If user attempts to upload a file.
    if btn_classify and  uploaded_file is not None:
        with st.spinner("Please wait while we process your request...."):
            image_binary = uploaded_file.read()
            
        
            image = {"file": image_binary}
            url = "http://127.0.0.1:8002/classify"
        
            api = requests.post(url, files=image)
            result = api.json()
        
            st.write(result)
            st.subheader(f"Classification: {result['prediction'][0][0]}")
            
if __name__ == "__main__":
    main()