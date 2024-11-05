import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import keras
import keras_nlp
import os

# Set credentials programmatically
os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle_username"]
os.environ['KAGGLE_KEY'] = st.secrets["kaggle_api_key"]

# # Initialize and authenticate with the Kaggle API
api = KaggleApi()
api.authenticate()

def download_model():
    # Define the model path and filename
    model_path = 'icchencecilia/gemma/keras/finetuned_gemma/1'

    # Download the model (it will be saved as a .zip file)
    # Downloads and unzips in the current directory
    api.model_instance_version_download(model_path, path='./kaggle', untar=True)  

@st.cache_resource
def load_kaggle_model():
    filepath = './kaggle/model.weights.h5'
    if not os.path.exists(filepath):
        download_model()

    # Open the downloaded file and load it directly into Keras
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    model = keras_nlp.models.GemmaCausalLM.load_weights(self=gemma_lm, filepath=filepath)
    # model = keras.models.load_model(filepath)
        
    return model

# Load the model
model = load_kaggle_model()

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    model_prompt = template.format(
        # instruction="I can't even take care of myself. I will never be a good dog mom.",
        instruction=prompt,
        response="",
    )
    model_response = model.generate(model_prompt, max_length=256)

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(model_response)
    st.session_state.messages.append({"role": "assistant", "content": response})