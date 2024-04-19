import streamlit as st
import os
import base64
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Load the service account credentials from Streamlit secrets
service_account_info = {
    "type": st.secrets["gcp"]["type"],
    "project_id": st.secrets["gcp"]["project_id"],
    "private_key_id": st.secrets["gcp"]["private_key_id"],
    "private_key": st.secrets["gcp"]["private_key"],
    "client_email": st.secrets["gcp"]["client_email"],
    "client_id": st.secrets["gcp"]["client_id"],
    "auth_uri": st.secrets["gcp"]["auth_uri"],
    "token_uri": st.secrets["gcp"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp"]["client_x509_cert_url"]
}

# Authenticate with Vertex AI
credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)
vertexai.init(project=service_account_info["project_id"], credentials=credentials)

# Hardcoded system prompt
system_prompt = Part.from_text("The provided document contains information that is not in Elgish. Your job is to created a detailed and informative summary of the document, in English for the user. Make sure the summary is thorough, well structured and written.")

# Function to generate content (modified for multiple files)
def generate_content(file_contents, file_names, system_prompt):
    file_parts = []
    for file_content, file_name in zip(file_contents, file_names):
        mime_type = None
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            mime_type = "image/jpeg"
        elif file_name.lower().endswith('.mp4'):
            mime_type = "video/mp4"
        elif file_name.lower().endswith('.pdf'):
            mime_type = "application/pdf"
        elif file_name.lower().endswith(('.mp3', '.wav')):
            mime_type = "audio/mpeg"
        if mime_type is None:
            raise ValueError("Unsupported file type")

        file_parts.append(Part.from_data(mime_type=mime_type, data=file_content))

    model = GenerativeModel("gemini-1.5-pro-preview-0409")  # Replace with your model name
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    chat = model.start_chat()
    response = chat.send_message(
        [system_prompt, *file_parts],  # Include files and system prompt
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return response.candidates[0].content.parts[0].text  # Extract text output

# Streamlit App (modified)
st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    st.title("🌐 Non-English Document Summariser")
    st.subheader("Disclaimer")
    st.write("Please note: The summaries generated are based on machine translation and natural language processing techniques. They may not capture nuanced meanings perfectly.    st.write("---")  # Adding a visual break
    st.write("---")  # Adding a visual break

    st.header("Upload your non-English documents in Image, PDF or text format below")
    uploaded_files = st.file_uploader("Choose files", type=["jpg", "jpeg", "png", "mp4", "pdf", "mp3", "wav"], accept_multiple_files=True)
    if uploaded_files:
        file_contents = [file.read() for file in uploaded_files]
        file_names = [file.name for file in uploaded_files]
        with st.spinner('Generating content...'):
            generated_content = generate_content(file_contents, file_names, system_prompt)
        st.write(generated_content)
