if [ ! -d "env" ]; then
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
fi

# download llama 3 8b quantized if no models found
if [ ! -d "models" ] || [ ! "$(ls -A models)" ]; then
    mkdir models
    wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf -O models/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf
fi

python3 transcribe.py

