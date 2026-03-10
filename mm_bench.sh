#!/bin/bash

# --- Configuration ---
IP="34.172.78.220" 
MODEL="qwen/Qwen2.5-VL-7B-Instruct"

# --- Hardcoded Image URLs ---
IMAGE_URLS=(
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
    "https://cdn2.thecatapi.com/images/dui.jpg"
    "https://cdn2.thecatapi.com/images/ebv.jpg"
    "https://cdn2.thecatapi.com/images/0XYvRd7oD.jpg"
)

# Capture start time (nanoseconds)
START_TIME=$(date +%s.%N)

# --- Loop through the Array ---
for URL in "${IMAGE_URLS[@]}"; do
    echo "------------------------------------------"
    echo "Processing: $URL"
    
    curl -i "http://${IP}:80/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d @- <<EOF
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image in one sentence."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "$URL"
          }
        }
      ]
    }
  ]
}
EOF
    echo -e "\n"
done

END_TIME=$(date +%s.%N)

# Calculate duration
# Using 'bc' for floating point math (common in Linux)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

echo -e "\n-----------------------------------"
echo "Query completed in: $DURATION seconds"