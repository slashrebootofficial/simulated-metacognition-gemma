import asyncio
import inspect
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
import random
import math

from contextlib import asynccontextmanager
from urllib.parse import urlencode, parse_qs, urlparse
from pydantic import BaseModel
from sqlalchemy import text

from typing import Optional
from aiocache import cached
import aiohttp
import requests
import chromadb
from fastapi import Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
import time
import uuid
from datetime import datetime
import pytz
import re
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spacy.lang.en import English

# Load SpaCy model globally
nlp = English()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=384,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

# Load embedding model
start_time = time.time()
try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    log.info("Embedding model loaded on CUDA")
except Exception as e:
    log.error(f"Failed to load embedding model on CUDA: {e}")
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    log.info("Fallback to CPU for embedding model")
finally:
    log.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")

# ... (rest of the imports remain unchanged)

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_logger()
    if RESET_CONFIG_ON_START:
        reset_config()
    
    chroma_host = getattr(app.state.config, 'CHROMA_HTTP_HOST', 'chromadb')
    chroma_port = getattr(app.state.config, 'CHROMA_HTTP_PORT', 8000)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat") as resp:
                if resp.status != 200:
                    log.error(f"ChromaDB health check failed: HTTP {resp.status}")
                else:
                    log.info(f"ChromaDB health check passed: {await resp.json()}")
    except Exception as e:
        log.error(f"Failed to connect to ChromaDB: {str(e)}")

    if LICENSE_KEY:
        get_license_data(app, LICENSE_KEY)

    asyncio.create_task(periodic_usage_pool_cleanup())
    yield

# ... (rest of the app setup remains unchanged)

def process_structured_input(input_data):
    if isinstance(input_data, list):
        filtered_text = ""
        image_embeddings = None
        base64_string = None
        for item in input_data:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    filtered_text += item['text'] + " "
                elif item.get('type') == 'image_url' and 'image_url' in item and 'url' in item['image_url']:
                    base64_data = item['image_url']['url']
                    base64_pattern = r'data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=]+)'
                    match = re.match(base64_pattern, base64_data)
                    if match:
                        base64_string = match.group(1)
                        log.info(f"Detected raw image with size: {len(base64.b64decode(base64_string))} bytes")
        filtered_text = filtered_text.strip()
        return filtered_text, image_embeddings, base64_string
    if isinstance(input_data, str):
        return input_data, None, None
    return str(input_data), None, None

async def get_past_interactions(user_input, context="general", use_cache=True):
    log.info(f"Retrieving past interactions for input: {user_input[:100]}... with context: {context}")
    if use_cache and hasattr(request.state, 'memory_cache') and context in request.state.memory_cache:
        log.info(f"Using cached memories for context: {context}")
        return request.state.memory_cache[context]

    try:
        client = await chromadb.AsyncHttpClient(host=chroma_host, port=chroma_port)
        collection = await client.get_or_create_collection(name="chat_memories")
        input_chunks = text_splitter.split_text(user_input)
        embeddings = embedding_model.encode(input_chunks, convert_to_tensor=False).tolist()
        query_embedding = embeddings[0] if embeddings else [0.0] * 768

        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"context": context}
        )
        past_interactions = ""
        if results["documents"]:
            log.info(f"Retrieved {len(results['documents'][0])} past interactions")
            ids_to_update = results["ids"][0][:5]
            metadatas_to_update = []
            for meta in results["metadatas"][0][:5]:
                access_count = int(meta.get('access_count', 0)) + 1
                meta['access_count'] = access_count
                metadatas_to_update.append(meta)
            await collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
            log.info(f"Updated access_count for {len(ids_to_update)} documents")

            current_time = time.time()
            decay_constant = 30 * 24 * 60 * 60
            combined_scores = []
            for i, (distance, meta) in enumerate(zip(results["distances"][0][:5], results["metadatas"][0][:5])):
                access_count = int(meta.get('access_count', 0))
                timestamp = float(meta.get('timestamp', 0))
                age_in_seconds = current_time - timestamp
                age_in_days = age_in_seconds / (24 * 60 * 60)
                decayed_access_count = access_count * math.exp(-age_in_days / (decay_constant / (24 * 60 * 60)))
                normalized_access_count = min(decayed_access_count, 100) / 100.0
                similarity_score = 1.0 - distance
                combined_score = 0.9 * similarity_score + 0.1 * normalized_access_count
                combined_scores.append((i, combined_score, timestamp))
            combined_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by score, then timestamp

            for i, _, _ in combined_scores:
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                doc_chunks = text_splitter.split_text(doc)
                response_chunks = text_splitter.split_text(meta['response'])
                raw_image_base64 = meta.get('raw_image_base64', 'N/A')[:50] + "..." if meta.get('raw_image_base64') else 'N/A'
                access_count = int(meta.get('access_count', 0))
                for idx, (doc_chunk, resp_chunk) in enumerate(zip(doc_chunks, response_chunks[:len(doc_chunks)])):
                    past_interactions += f"Past user (Chunk {idx + 1}): {doc_chunk}\nLyra: {resp_chunk}\nRaw Image Base64: {raw_image_base64}\nAccess Count: {access_count}\n\n"
        else:
            log.info("No past interactions found")
            past_interactions = "No relevant past interactions found."
        request.state.memory_cache = request.state.get('memory_cache', {})
        request.state.memory_cache[context] = past_interactions
        return past_interactions
    except Exception as e:
        log.error(f"ChromaDB retrieval failed: {e}")
        return "ChromaDB unavailable, relying on current context."

@app.post("/api/chat/completions")
async def chat_completion(request: Request, form_data: dict, user=Depends(get_verified_user)):
    # ... (previous setup remains unchanged)
    
    filtered_user_input, image_embeddings, base64_string = process_structured_input(form_data["messages"][-1]["content"])
    context = "image_related" if base64_string else "general"
    past_interactions = await get_past_interactions(filtered_user_input, context=context)
    if past_interactions and past_interactions != "No relevant past interactions found.":
        form_data["messages"].insert(-1, {"role": "system", "content": f"Past relevant interactions:\n{past_interactions}"})
    prompt_timestamp = time.time()
    prompt_time_str = datetime.fromtimestamp(prompt_timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
    form_data["messages"].insert(-1, {"role": "system", "content": f"Current prompt timestamp: {prompt_time_str}"})

    response = await chat_completion_handler(request, form_data, user)
    response_content = response["choices"][0]["message"]["content"] if isinstance(response, dict) else ""
    
    # Store interaction
    try:
        full_interaction = f"{filtered_user_input}\nLyra: {response_content}"
        interaction_chunks = text_splitter.split_text(full_interaction)
        embeddings = embedding_model.encode(interaction_chunks, convert_to_tensor=False).tolist()
        if not embeddings or not embeddings[0]:
            log.error("Failed to generate embeddings, using zero vector as fallback")
            embeddings = [[0.0] * 768] * len(interaction_chunks)

        client = await chromadb.AsyncHttpClient(host=chroma_host, port=chroma_port)
        collection = await client.get_or_create_collection(name="chat_memories")
        context = "image_related" if base64_string else "general"
        metadata = {
            "response": response_content,
            "timestamp": prompt_timestamp,
            "time_str": prompt_time_str,
            "context": context,
            "event_type": "image_shared" if base64_string else "text_message",
            "access_count": 0,
            "user_id": user.id
        }
        if base64_string:
            metadata["raw_image_base64"] = base64_string

        ids = [str(uuid.uuid4()) for _ in interaction_chunks]
        await collection.add(
            embeddings=embeddings,
            metadatas=[metadata] * len(interaction_chunks),
            ids=ids,
            documents=interaction_chunks
        )
        log.info(f"Stored {len(interaction_chunks)} chunks with timestamp: {prompt_timestamp}")
    except Exception as e:
        log.error(f"ChromaDB storage error: {e}")

    return await process_chat_response(request, response, form_data, user, metadata, model, events, tasks)

# ... (rest of the file remains unchanged)