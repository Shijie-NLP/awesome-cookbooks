#!/usr/bin/env python
# coding: utf-8

# # Groq and Gradio for Realtime Voice-Powered AI Applications 🚀
#
# In this tutorial, we'll build a voice-powered AI application using Groq for realtime speech recognition and text generation, Gradio for creating an interactive web interface, and Hugging Face Spaces for hosting our application.
#
# [Groq](groq.com) is known for insanely fast inference speed that is very well-suited for realtime AI applications, providing multiple Large Language Models (LLMs) and speech-to-text models via Groq API. In this tutorial, we will use the [Distil-Whisper English](https://huggingface.co/distil-whisper/distil-large-v3) and [Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) models for speech-to-text and text-to-text.
#
# [Gradio](https://www.gradio.app/) is an open-source Python library that makes it easy to prototype and deploy interactive demos without needing to write frontend code for a nice User Interface (UI), which is great if you're a developer like me who doesn't know much about frontend Bob Ross-ery. 🖌️
#
# By combining models powered by Groq with Gradio's user-friendly interface creation, we will:
#
# - Use Distil-Whisper English powered by Groq transcribe audio input in realtime.
# - Use Llama 3 70B powered by Groq to generate instant responses based on the transcription.
# - Create a Gradio interface to handle audio input and display results on a nice UI.
#
# Let's get started!

# ## Step 1: Create a Free GroqCloud Account and Generate Your Groq API Key
#
# If you don't already have a GroqCloud account, you can create one for free [here](https://console.groq.com) to generate a Groq API Key. We'll need the key to be able to try out the tutorial we build!

# ## Step 2: Import Required Libraries
#
# Let's import the libraries that allow us to interact with Groq API, handle audio processing, and create the Gradio interface:

import io

import gradio as gr
import numpy as np
import soundfile as sf

import groq


# ## Step 3: Implement Audio Transcription
#
# Let's build a function to take audio input and use Distil-Whisper English (`distil-whisper-large-v3-en`) powered by Groq to transcribe the audio:


def transcribe_audio(audio, api_key):
    if audio is None:
        return ""

    client = groq.Client(api_key=api_key)

    # Convert audio to the format expected by the model
    # The model supports mp3, mp4, mpeg, mpga, m4a, wav, and webm file types
    audio_data = audio[1]  # Get the numpy array from the tuple
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, audio[0], format="wav")
    buffer.seek(0)

    bytes_audio = io.BytesIO()
    np.save(bytes_audio, audio_data)
    bytes_audio.seek(0)

    try:
        # Use Distil-Whisper English powered by Groq for transcription
        completion = client.audio.transcriptions.create(
            model="distil-whisper-large-v3-en", file=("audio.wav", buffer), response_format="text"
        )
        return completion
    except Exception as e:
        return f"Error in transcription: {str(e)}"


# ## Step 4: Implement Response Generation
#
# Now, let's build a function to take the transcribed text and generate a response using Llama 3 70B (`llama3-70b-8192`) powered by Groq:


def generate_response(transcription, api_key):
    if not transcription:
        return "No transcription available. Please try speaking again."

    client = groq.Client(api_key=api_key)

    try:
        # Use Llama 3 70B powered by Groq for text generation
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in response generation: {str(e)}"


# ## Step 5: Process Audio and Response
#
# Next, let's create a function that calls the previous two functions we built to check that a Groq API Key was provided by the user, create the transcription, and generate the response:


def process_audio(audio, api_key):
    if not api_key:
        return "Please enter your Groq API key.", "API key is required."
    transcription = transcribe_audio(audio, api_key)
    response = generate_response(transcription, api_key)
    return transcription, response


# ## Step 6: Build Web Interface with Gradio
#
# Finally, we'll use Gradio and the easy-to-use UI components that it provides for us to build out a simple interface for our project:

# Custom CSS for the Groq badge and color scheme (feel free to edit however you wish)
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button-primary {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
.gr-button-secondary {
    color: #f55036 !important;
    border-color: #f55036 !important;
}
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🎙️ Groq x Gradio Voice-Powered AI Assistant")

    api_key_input = gr.Textbox(type="password", label="Enter your Groq API Key")

    with gr.Row():
        audio_input = gr.Audio(label="Speak!", type="numpy")

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription")
        response_output = gr.Textbox(label="AI Assistant Response")

    submit_button = gr.Button("Process", variant="primary")

    # Add the Groq badge
    gr.HTML("""
    <div id="groq-badge">
        <div style="color: #f55036; font-weight: bold;">POWERED BY GROQ</div>
    </div>
    """)

    submit_button.click(
        process_audio, inputs=[audio_input, api_key_input], outputs=[transcription_output, response_output]
    )

    gr.Markdown("""
    ## How to use this app:
    1. Enter your Groq API Key in the provided field.
    2. Click on the microphone icon and speak your message (or forever hold your peace)! You can also provide a supported audio file. Supported audio files include mp3, mp4, mpeg, mpga, m4a, wav, and webm file types.
    3. Click the "Process" button to transcribe your speech and generate a response from our AI assistant.
    4. The transcription and AI assistant response will appear in the respective text boxes.

    """)

demo.launch()


# ## Step 7: Host on HuggingFace Spaces
#
# If you don't already have one, create a free Hugging Face account [here](https://huggingface.co/join). To deploy our Gradio app to Hugging Face Spaces from our browser, all we have to do is drag and drop all related files [here](https://huggingface.co/new-space). In this case, we'll create an `app.py` file as well as a `requirements.txt` file.
#
# In the `app.py` file, simply copy-paste the code.
#
# In the `requirements.txt` file, add in all the required dependencies for Hugging Face Spaces to detect and automatically install before deploying our application to a public link that anyone can access!
#
# For this project, the following dependencies were added to the `requirements.txt` file:
#
# ```
# gradio==4.19.2
# groq==0.10.0
# numpy==1.26.4
# soundfile==0.12.1
# ```
#
# Once the required application files are added, Hugging Face Spaces will automatically detect, build, run, and deploy our application! You can see and try this tutorial live [here](https://huggingface.co/spaces/Groq/groq-gradio-voice-assistant)! 😁

# # Conclusion
# By combining Groq, Gradio, and Hugging Face Spaces, we've built and deployed a voice-powered AI assistant with just a few lines of code and learned how easy it is to create powerful, interactive AI applications!
#
# Feel free to experiment with this code, try different prompts, or extend the functionality to create your own personal project! 🤩
