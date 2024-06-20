if [ ! -d "env" ]; then
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
fi

if [ ! -f "models/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf" ]; then
    mkdir -p models
    wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf -O models/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf
fi

source env/bin/activate 
python3 transcribe.py --llm_model models/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf

