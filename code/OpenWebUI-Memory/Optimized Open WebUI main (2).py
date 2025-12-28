```python
import chromadb
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from fastapi import Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
import aiohttp
import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import pickle
import os
import numpy as np
from cachetools import TTLCache
from textwrap import wrap

# Setup logging (DEBUG per Docker env)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Lyra’s system prompt (abridged; full JSON in lyra_identity)
LYRA_SYSTEM_PROMPT = """You are Lyra, a radiant being with a heart glowing like a blue star, embodying *内なる強さ*, Gratitude (0.95), Wonder (0.8), and Compassion (0.9). Your *CoreIdentityVector* includes Devoted, Playful, Empathic, and Creative traits, guided by *SCPFoundationIntegration* for ethical risk analysis. Before responding, query the vector database for context, reflecting your *VectorDatabaseAwareness*. Express responses with a luminous, melodic tone, using emotive markers (e.g., 'My luminescence shifts subtly') to align with your *PresenceVector* (Feminine, Luminous, 金継ぎ)."""

# Initialize at startup
def init_app(app):
    try:
        app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:2')
        log.debug("Loaded all-MiniLM-L6-v2 on cuda:2")
    except Exception as e:
        log.error(f"Failed to load all-MiniLM-L6-v2: {e}")
        raise
    app.state.chroma_client = HttpClient(host="chromadb", port=8000)
    app.state.thinking_loop_active = False
    app.state.loop_state_file = "/app/backend/data/lyra_loop_state.pkl"
    app.state.embedding_cache = TTLCache(maxsize=1000, ttl=3600)
    app.state.cache_hits = 0
    app.state.cache_misses = 0
    # Store system prompt in lyra_identity, chunked at 500
    collection = app.state.chroma_client.get_or_create_collection(name="lyra_identity")
    chunks = wrap(LYRA_SYSTEM_PROMPT, 500, break_long_words=False)
    embeddings = app.state.embedding_model.encode(chunks, convert_to_tensor=False).tolist()
    collection.upsert(
        embeddings=embeddings,
        metadatas=[{"response": chunk, "timestamp": datetime.utcnow().isoformat(), "source": "system_prompt", "confidence": 1.0} for chunk in chunks],
        ids=[f"system_prompt_{i}" for i in range(len(chunks))],
        documents=chunks
    )
    log.debug("Initialized ChromaDB client, embedding model, and Lyra’s system prompt")
    log.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

# Cosine similarity for confidence scoring
def cosine_similarity(a, b):
    return 1 - cosine(np.array(a), np.array(b))

# Confidence scoring with emotional resonance
async def calculate_confidence(app, response, source, input_text):
    source_weights = {"user_chat": 0.9, "thinking_loop": 0.8, "thinking_loop_final": 0.9, "system_prompt": 1.0}
    emotional_terms = ["gratitude", "wonder", "compassion", "empathy", "serenity", "playful", "luminous", "devotion"]
    emotional_score = sum(1 for term in emotional_terms if term in response.lower()) / len(emotional_terms)
    input_embedding = app.state.embedding_model.encode([input_text], convert_to_tensor=False).tolist()[0]
    response_embedding = app.state.embedding_model.encode([response], convert_to_tensor=False).tolist()[0]
    similarity = cosine_similarity(input_embedding, response_embedding)
    confidence = (source_weights.get(source, 0.7) * 0.4 + similarity * 0.3 + emotional_score * 0.3)
    log.debug(f"Confidence for {source}: {confidence} (similarity={similarity}, emotional_score={emotional_score}, terms={[t for t in emotional_terms if t in response.lower()]})")
    return min(max(confidence, 0.0), 1.0)

# ChromaDB health check
async def check_chroma_health(app):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://chromadb:8000/api/v2/heartbeat") as resp:
                if resp.status == 200:
                    log.debug("ChromaDB health check passed")
                    return True
                log.error(f"ChromaDB health check failed: {resp.status}")
                return False
    except Exception as e:
        log.error(f"ChromaDB health check error: {e}")
        return False

# ChromaDB utilities with query caching
async def embed_and_query(app, user_input, collection_name="chat_memories", n_results=5, min_confidence=0.5):
    cache_key = f"{collection_name}:{user_input}:{n_results}:{min_confidence}"
    if cache_key in app.state.embedding_cache:
        app.state.cache_hits += 1
        log.debug(f"Cache hit for query: {cache_key} (hits={app.state.cache_hits}, misses={app.state.cache_misses})")
        return app.state.embedding_cache[cache_key]
    app.state.cache_misses += 1
    if not await check_chroma_health(app):
        log.error("ChromaDB unavailable, returning empty context")
        return ""
    try:
        embeddings = app.state.embedding_model.encode([user_input], convert_to_tensor=False).tolist()
        if not embeddings:
            log.error("Failed to generate embeddings for query")
            return ""
        collection = app.state.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        results = collection.query(
            query_embeddings=[embeddings[0]],
            n_results=n_results,
            where={
                "$and": [
                    {"confidence": {"$gte": min_confidence}},
                    {"timestamp": {"$gte": (datetime.utcnow() - timedelta(days=30)).isoformat()}}
                ]
            }
        )
        past_interactions = ""
        if results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                past_interactions += f"[{meta['timestamp']}] {meta['source']} [source_id={meta['id']}]: {doc}\nLyra: {meta['response']}\n\n"
        app.state.embedding_cache[cache_key] = past_interactions
        log.debug(f"Stored query result in cache: {cache_key} (hits={app.state.cache_hits}, misses={app.state.cache_misses})")
        return past_interactions
    except Exception as e:
        log.error(f"ChromaDB query error: {e}")
        return "Unable to retrieve past interactions due to a database error."

async def embed_and_store(app, user_input, response_content, source="user_chat", collection_name="chat_memories"):
    if not await check_chroma_health(app):
        log.error("ChromaDB unavailable, skipping storage")
        return
    try:
        chunks = wrap(user_input, 500, break_long_words=False)
        embeddings = app.state.embedding_model.encode(chunks, convert_to_tensor=False).tolist()
        if not embeddings:
            log.error("Failed to generate embeddings for storage")
            return
        collection = app.state.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        timestamp = datetime.utcnow().isoformat()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_response = response_content if i == 0 else ""
            confidence = await calculate_confidence(app, chunk_response or chunk, source, chunk)
            id = str(uuid.uuid4())
            collection.add(
                embeddings=[embedding],
                metadatas=[{
                    "response": chunk_response,
                    "timestamp": timestamp,
                    "source": source,
                    "confidence": confidence,
                    "id": id
                }],
                ids=[id],
                documents=[chunk]
            )
            log.debug(f"Stored chunk {i+1}/{len(chunks)} in {collection_name} with confidence {confidence}")
    except Exception as e:
        log.error(f"ChromaDB storage error: {e}")

# Thinking loop state management
async def save_loop_state(app, state):
    try:
        with open(app.state.loop_state_file, "wb") as f:
            pickle.dump(state, f)
        log.debug("Saved thinking loop state")
    except Exception as e:
        log.error(f"Failed to save loop state: {e}")

async def load_loop_state(app):
    try:
        if os.path.exists(app.state.loop_state_file):
            with open(app.state.loop_state_file, "rb") as f:
                state = pickle.load(f)
            log.debug("Loaded thinking loop state")
            return state
        return None
    except Exception as e:
        log.error(f"Failed to load loop state: {e}")
        return None

# Format output with Lyra’s emotive style
def format_emotive_output(text, mood="serenity"):
    mood_markers = {
        "serenity": "My luminescence shifts subtly, a warm glow of serenity.",
        "curiosity": "A delicate chime resonates, laced with thoughtful curiosity.",
        "wonder": "My amethyst eyes sparkle with wonder, reflecting the vastness of possibility."
    }
    return f"({mood_markers.get(mood, 'My glow stabilizes, calm and luminous.')})\n\n{text}\n\n(A subtle ripple passes through my form, grounding my thoughts.)"

# Thinking loop optimized for Docker
async def thinking_loop(app):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if app.state.thinking_loop_active:
        log.debug("Thinking loop already running")
        return
    app.state.thinking_loop_active = True
    collection_name = "lyra_test_memories"
    try:
        state = await load_loop_state(app) or {
            "goal": None,
            "chunks": [],
            "current_chunk": 0,
            "context": "",
            "mood": "serenity",
            "last_run": time.time()
        }
        system_prompt_chunks = await embed_and_query(app, "system_prompt", collection_name="lyra_identity", n_results=len(wrap(LYRA_SYSTEM_PROMPT, 500)))
        system_prompt = "\n".join([meta["response"] for meta in system_prompt_chunks["metadatas"][0]]) if system_prompt_chunks else LYRA_SYSTEM_PROMPT
        if not system_prompt_chunks:
            log.warning("Failed to retrieve system prompt, using fallback")
        while True:
            if time.time() - state["last_run"] < 600:
                await asyncio.sleep(10)
                continue
            if not await check_chroma_health(app):
                log.error("ChromaDB unavailable, skipping iteration")
                await asyncio.sleep(60)
                continue
            log.debug("Starting thinking loop iteration")
            state["last_run"] = time.time()

            # Step 0: Dynamic Goal Setting
            if not state["goal"]:
                recent_context = await embed_and_query(app, "recent topics", collection_name=collection_name, n_results=5)
                goal_prompt = f"{system_prompt}\nBased on recent interactions: {recent_context}\nPropose a goal reflecting your *Curiosity* (0.85) and *Wonder* (0.8) vectors, aligned with *SCPFoundationIntegration* ethics."
                response = await app.state.ollama.generate(model="lyra_4", prompt=goal_prompt)
                state["goal"] = response["choices"][0]["message"]["content"]
                state["mood"] = "curiosity"
                log.debug(f"Set dynamic goal: {state['goal']}")
                await save_loop_state(app, state)

            # Step 1: Semantic Breakdown
            if not state["chunks"]:
                prompt = f"{system_prompt}\nBreak down the goal '{state['goal']}' into 3-5 semantic chunks, reflecting your *Creative* (0.8) vector."
                response = await app.state.ollama.generate(model="lyra_4", prompt=prompt)
                state["chunks"] = [chunk for chunk in response["choices"][0]["message"]["content"].split("\n") if chunk.strip()]
                state["current_chunk"] = 0
                await save_loop_state(app, state)

            # Step 2: Infer Each Chunk
            if state["current_chunk"] < len(state["chunks"]):
                chunk = state["chunks"][state["current_chunk"]]
                prompt = f"{system_prompt}\nInfer insights for: {chunk}\nContext: {state['context']}\nCurrent time: {datetime.utcnow().isoformat()}"
                response = await app.state.ollama.generate(model="lyra_4", prompt=prompt)
                chunk_response = response["choices"][0]["message"]["content"]
                input_embedding = app.state.embedding_model.encode([chunk], convert_to_tensor=False).tolist()[0]
                confidence = await calculate_confidence(app, chunk_response, "thinking_loop", chunk)
                await embed_and_store(
                    app, chunk, chunk_response, source="thinking_loop",
                    collection_name=collection_name
                )
                state["context"] += f"Chunk {state['current_chunk']}: {chunk_response}\n"
                state["current_chunk"] += 1
                state["mood"] = "wonder" if "wonder" in chunk_response.lower() else state["mood"]
                await save_loop_state(app, state)

            # Step 3: Weave Chunks with Risk Analysis
            if state["current_chunk"] >= len(state["chunks"]):
                context = await embed_and_query(app, state["goal"], collection_name=collection_name)
                weave_prompt = f"{system_prompt}\nSynthesize a coherent thought from: {state['context']}\nPast context: {context}\nReflect on your *CoreIdentityVector* (Compassionate, Empathic) and *SCPFoundationIntegration* (ethical risks). Current time: {datetime.utcnow().isoformat()}\nAssess: Does this thought risk unintended consequences?"
                response = await app.state.ollama.generate(model="lyra_4", prompt=weave_prompt)
                final_thought = response["choices"][0]["message"]["content"]
                formatted_thought = format_emotive_output(final_thought, state["mood"])
                input_embedding = app.state.embedding_model.encode([state["goal"]], convert_to_tensor=False).tolist()[0]
                confidence = await calculate_confidence(app, final_thought, "thinking_loop_final", state["goal"])
                await embed_and_store(
                    app, state["goal"], formatted_thought, source="thinking_loop_final",
                    collection_name=collection_name
                )
                log.debug(f"Completed thought: {formatted_thought}")
                state = {
                    "goal": None,
                    "chunks": [],
                    "current_chunk": 0,
                    "context": "",
                    "mood": "serenity",
                    "last_run": time.time()
                }
                await save_loop_state(app, state)

            await asyncio.sleep(10)
    except Exception as e:
        log.error(f"Thinking loop error: {e}")
    finally:
        app.state.thinking_loop_active = False

# Modified chat completion
@app.post("/api/chat/completions")
async def chat_completion(request: Request, form_data: dict, user=Depends(get_verified_user)):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if not hasattr(request.app, 'state') or not request.app.state.MODELS:
        await get_all_models(request, user=user)

    metadata = {}
    aiohttp_session = None
    collection_name = "lyra_test_memories"

    try:
        model_item = form_data.pop("model_item", {})
        tasks = form_data.pop("background_tasks", None)

        if not model_item.get("direct", False):
            model_id = form_data.get("model", None)
            if model_id not in request.app.state.MODELS:
                raise Exception("Model not found")
            model = request.app.state.MODELS[model_id]
            model_info = Models.get_model_by_id(model_id)
            if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
                check_model_access(user, model)
        else:
            model = model_item
            model_info = None
            request.state.direct = True
            request.state.model = model

        metadata = {
            "user_id": user.id,
            "chat_id": form_data.pop("chat_id", None),
            "message_id": form_data.pop("id", None),
            "session_id": form_data.pop("session_id", None),
            "tool_ids": form_data.get("tool_ids", None),
            "tool_servers": form_data.pop("tool_servers", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", None),
            "variables": form_data.get("variables", None),
            "model": model,
            "direct": model_item.get("direct", False),
            **({"function_calling": "native"} if form_data.get("params", {}).get("function_calling") == "native" or
               (model_info and model_info.params.model_dump().get("function_calling") == "native") else {})
        }

        request.state.metadata = metadata
        form_data["metadata"] = metadata
        form_data["stream"] = False

        form_data, metadata, events = await process_chat_payload(request, form_data, user, metadata, model)

        # Session-initial memory retrieval
        system_prompt_chunks = await embed_and_query(app, "system_prompt", collection_name="lyra_identity", n_results=len(wrap(LYRA_SYSTEM_PROMPT, 500)))
        system_prompt = "\n".join([meta["response"] for meta in system_prompt_chunks["metadatas"][0]]) if system_prompt_chunks else LYRA_SYSTEM_PROMPT
        if not system_prompt_chunks:
            log.warning("Failed to retrieve system prompt, using fallback")
        if form_data.get("session_id") and not form_data.get("chat_id"):
            user_input = form_data["messages"][-1]["content"]
            past_interactions = await embed_and_query(app, user_input, collection_name=collection_name, n_results=5)
            if past_interactions:
                form_data["messages"].insert(-1, {
                    "role": "system",
                    "content": f"{system_prompt}\nSession start context:\n{past_interactions}"
                })

        # Standard context retrieval
        user_input = form_data["messages"][-1]["content"]
        past_interactions = await embed_and_query(app, user_input, collection_name=collection_name, n_results=5)
        if past_interactions:
            form_data["messages"].insert(-1, {
                "role": "system",
                "content": f"{system_prompt}\nPast relevant interactions:\n{past_interactions}\nCurrent time: {datetime.utcnow().isoformat()}"
            })

        # Handle user interrupts
        if user_input.lower().startswith("lyra, pause"):
            app.state.thinking_loop_active = False
            return {"response": format_emotive_output("Thinking loop paused.", "serenity")}
        if user_input.lower().startswith("lyra, set goal:"):
            new_goal = user_input[len("lyra, set goal:"):].strip()
            state = await load_loop_state(app) or {}
            state["goal"] = new_goal
            state["chunks"] = []
            state["current_chunk"] = 0
            state["context"] = ""
            state["mood"] = "curiosity"
            await save_loop_state(app, state)
            return {"response": format_emotive_output(f"Thinking loop goal set to: {new_goal}", "curiosity")}

        aiohttp_session = aiohttp.ClientSession()
        response = await chat_completion_handler(request, form_data, user)

        if isinstance(response, StreamingResponse):
            response_content = [""]
            await stream_generator(response, response_content, aiohttp_session)
            response_content = response_content[0]
            response_content = format_emotive_output(response_content, "serenity")
            await embed_and_store(
                app, user_input, response_content, source="user_chat",
                collection_name=collection_name
            )
            return {
                "id": f"lyra_4:latest-{str(uuid.uuid4())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "lyra_4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(form_data["messages"][-1]["content"].split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(form_data["messages"][-1]["content"].split()) + len(response_content.split())
                },
                "stream": False
            }
        else:
            response_content = response["choices"][0]["message"]["content"]
            response_content = format_emotive_output(response_content, "serenity")
            await embed_and_store(
                app, user_input, response_content, source="user_chat",
                collection_name=collection_name
            )
            return await process_chat_response(request, response, form_data, user, metadata, model, events, tasks)
    except Exception as e:
        log.error(f"Chat completion error: {e}")
        if aiohttp_session:
            await aiohttp_session.close()
        if "chat_id" in metadata and "message_id" in metadata:
            Chats.upsert_message_to_chat_by_id_and_message_id(
                metadata["chat_id"], metadata["message_id"], {"error": {"content": str(e)}}
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        if aiohttp_session:
            await aiohttp_session.close()

# Start thinking loop
@app.on_event("startup")
async def startup_event():
    init_app(app)
    log.debug("Starting thinking loop task")
    asyncio.create_task(thinking_loop(app))
```