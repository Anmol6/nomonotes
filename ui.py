import argparse
import streamlit as st
import time
import os
from datetime import datetime, timedelta
from queue import Queue
from subprocess import PIPE, Popen
from sys import platform
from time import sleep
import asyncio

import numpy as np
import speech_recognition as sr
import torch
import whisper
from llama_cpp import Llama
from openai import OpenAI
from anthropic import Anthropic

PROMPTS_DIR = "prompts"
DEFAULT_PROMPT = open(f"{PROMPTS_DIR}/meeting_prompt", "r").read()

def create_summary(transcript, model, system_prompt, api_key=None):
    if not model.endswith(".gguf"):
        if "gpt" in model:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize these meeting notes: {transcript}"},
                ],
            )
            return completion.choices[0].message.content
        
        elif "claude" in model:
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                temperature=0.0,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": f" {system_prompt}\nSummarize these meeting notes: {transcript}"},
                ],
            )
            return message.content[0].text
        
        else:
            raise Exception(f"Model {model} not supported")

    if not os.path.exists(model):
        raise Exception(f"No model found at {model}")

    # Llama code (unchanged)
    llm = Llama(
        model_path=model,
        n_ctx=30000,
        n_threads=8,
        n_gpu_layers=35,
        chat_format="llama-2",
    )

    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize these meeting notes: {transcript}"},
        ]
    )

    return completion["choices"][0]["message"]["content"]
def create_apple_note(folder_name, note_title, transcript):
    scpt = f"""
            -- Define the note content and the title
            set noteTitle to "{note_title}"
            set noteContent to "{transcript}"

            -- Define the folder name where the note should be created
            set folderName to "{folder_name}"

            -- Create the new note
            tell application "Notes"
                -- Check if the folder exists
                if not (exists folder folderName of default account) then
                    -- If not, create the folder
                    make new folder in default account with properties {{name:folderName}}
                end if
                
                -- Get the folder
                set targetFolder to folder folderName of default account
                
                -- Create the note in the specified folder
                tell targetFolder
                    make new note with properties {{body:noteContent}}
                end tell
            end tell
            """

    p = Popen(["osascript", "-e", scpt], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.communicate()

    return

def main():
    st.set_page_config(page_title="NOMO Notes", layout="wide")

    st.markdown("""
    <style>
    .stApp {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .content-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #3E3E3E;
        color: #FFFFFF;
        margin-bottom: 20px;
    }
    .transcription-box {
        height: 300px;
        overflow-y: auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>NOMO Notes</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'transcription' not in st.session_state:
        st.session_state.transcription = [""]
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    
    def on_click_recording():
        st.session_state.is_recording = not st.session_state.is_recording
        if not st.session_state.is_recording:  # Stopped recording
            st.session_state.summary_generated = False
    def on_change_api_key():
        st.session_state.api_key = st.session_state.user_input
    # Sidebar
    with st.sidebar:
        st.button("Start Recording" if not st.session_state.is_recording else "Stop Recording", on_click=on_click_recording)

        st.header("Settings")
        model = st.selectbox("Whisper Model", ["base", "small", "medium", "large"], index=1)
        llm_model = st.selectbox("LLM Model", ["claude-3-5-sonnet-20240620", "gpt-4o", "local"], index=0)
        if llm_model == "local":
            local_model_path = st.file_uploader("Choose a local model file", type=['gguf'])
            if local_model_path:
                llm_model = local_model_path.name
            else:
                raise Exception("Please select a .gguf file for the local model.")
        if ("gpt" in llm_model or "claude" in llm_model):
            st.text_input("LLM API Key", type="password", placeholder="Enter your API key here", key="user_input", on_change=on_change_api_key)


        notes_folder_name = st.text_input("Notes Folder Name", value="NOMO Notes")
        
        st.header("Audio Settings")
        energy_threshold = st.slider("Energy Threshold", 100, 5000, 1000)
        record_timeout = st.slider("Record Timeout", 0.5, 5.0, 2.0)
        phrase_timeout = st.slider("Phrase Timeout", 0.5, 10.0, 3.0)

        st.header("Summary Prompt")
        with st.expander("Edit Summary Prompt"):
            summary_prompt = st.text_area("", value=DEFAULT_PROMPT, height=400)

    # Main content
    if st.session_state.api_key:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.markdown("<p class='section-header'>Transcription</p>", unsafe_allow_html=True)
            transcription_placeholder = st.empty()
        transcription_placeholder.markdown("""
        <div class="content-box transcription-box">
        </div>
        """, unsafe_allow_html=True)
        # Recording logic
        if st.session_state.is_recording:
            with col2:
                with st.spinner(f"Loading Transcription model..."):
                    audio_model = whisper.load_model(model)
                    recorder = sr.Recognizer()
                    recorder.energy_threshold = energy_threshold
                    recorder.dynamic_energy_threshold = False

                    source = sr.Microphone(sample_rate=16000)

                    with source:
                        recorder.adjust_for_ambient_noise(source)

                    data_queue = Queue()

                    def record_callback(_, audio: sr.AudioData) -> None:
                        data = audio.get_raw_data()
                        data_queue.put(data)

                    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

                while st.session_state.is_recording:
                    try:
                        if not data_queue.empty():
                            audio_data = b"".join(list(data_queue.queue))
                            data_queue.queue.clear()

                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                            text = result["text"].strip()

                            st.session_state.transcription.append(text)
                            
                            transcription_placeholder.markdown(f"""
                            <div class="content-box transcription-box">
                                {'<br>'.join(st.session_state.transcription)}
                            </div>
                            """, unsafe_allow_html=True)

                        time.sleep(0.25)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        break
        else:
            with col2:
                transcription_placeholder.markdown(f"""
                    <div class="content-box transcription-box">
                        {'<br>'.join(st.session_state.transcription)}
                    </div>
                    """, unsafe_allow_html=True)

        # Summary generation and display
        if st.session_state.transcription != [""]:
            with col2:
                st.markdown("<p class='section-header'>Meeting Summary</p>", unsafe_allow_html=True)                
                if not st.session_state.summary:
                    full_transcript = "\n".join(st.session_state.transcription)
                    

                    with st.spinner("Generating summary..."):
                        summary =create_summary(full_transcript, llm_model, summary_prompt, st.session_state.api_key)
                        
                    st.session_state.summary = summary
                
                st.markdown(f"""
                <div class="content-box">
                    {st.session_state.summary}
                </div>
                """, unsafe_allow_html=True)

        # Export to Notes button
        if st.session_state.summary:
            if st.sidebar.button("Export to Notes"):
                meeting_title = f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                create_apple_note(notes_folder_name, meeting_title, st.session_state.summary)
                st.sidebar.success("Meeting note created and exported successfully!")
    else:
        st.warning("Please provide the required API key to use the application.")
if __name__ == "__main__":
    main()