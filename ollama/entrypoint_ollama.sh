#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

MODEL_TO_PULL="${OLLAMA_LLM_MODEL}"
echo "Retrieve model: $MODEL_TO_PULL ..."
ollama pull "$MODEL_TO_PULL"

EMBED_MODEL_TO_PULL="${OLLAMA_EMBED_MODEL}"
echo "Retrieve embedding model: $EMBED_MODEL_TO_PULL ..."
ollama pull "$EMBED_MODEL_TO_PULL"

ollama start
echo "Done!"

# Wait for Ollama process to finish.
wait $pid
