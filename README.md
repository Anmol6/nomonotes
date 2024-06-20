# NomoNotes

NomoNotes is a powerful tool designed to streamline your note-taking process during meetings, podcasts, and other audio sessions. It transcribes audio in real-time, summarizes the transcript using a language model, and saves the summary directly into Apple Notes.

## Features

- **Real-time Audio Transcription**: Utilizes Whisper for accurate and efficient transcription (code from https://github.com/davabase/whisper_real_time)
- **Summarization**: Summarizes the transcript using a language model (LLM).
- **Apple Notes Integration**:  Saves the summary into Apple Notes.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/nomonotes.git
    cd nomonotes
    ```
2. Install the required dependencies:
    `pip install -r requirements.txt`


## Usage

Run the script with the following command:
    `python transcribe.py --model medium --llm_model gpt-4o --summary_prompt meeting_prompt --notes_folder_name "NOMO Notes"`

### Command Line Arguments

- `--model`: Model to use for transcription (choices: `tiny`, `base`, `small`, `medium`, `large`).
- `--non_english`: Use non-English model.
- `--energy_threshold`: Energy level for mic to detect.
- `--llm_model`: LLM model to use for summarization (this should either be an OAI model or a `.gguf` model you've downloaded to `models/`)
- `--summary_prompt`: Path to the summary prompt (this is the name of the file with the prompt in the `prompts/` folder)
- `--notes_folder_name`: Name of the folder where the note will be created.

### Example

    `python transcribe.py --model medium --llm_model gpt-4o --summary_prompt meeting_prompt --notes_folder_name "NOMO Notes"`

## Use your own models

You can use your own local models for summarization. For example, if you have a fine-tuned model that summarizes meetings and podcasts in a specific format, you can specify it using the `--llm_model` argument.

To use an off-the-shelf llama, you can run `run_llama.sh` which downloads and runs the script with an 8b llama model.