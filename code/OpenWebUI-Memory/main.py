import json
import time
from datetime import datetime, timezone
import uuid
import aiohttp
import re
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import pytz

app = FastAPI()
log = logging.getLogger(__name__)

# Configuration
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 384
CHUNK_OVERLAP = 64
CONTEXT_WINDOW_LIMIT = 38912  # Lyra's context window
CONTEXT_WINDOW_THRESHOLD = int(0.75 * CONTEXT_WINDOW_LIMIT)  # ~29,184 tokens

# Initialize ChromaDB client and embedding model
client = chromadb.AsyncHttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
log.info("Embedding model 'all-mpnet-base-v2' loaded on CPU")

# Chunk text for memory storage
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text or not isinstance(text, str):
        return [text] if text else ["[NO_TEXT]"]
    tokens = text.split()
    chunks = []
    start = 0
    text_length = len(tokens)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += chunk_size - chunk_overlap
    return chunks if chunks else ["[NO_TEXT]"]

# Embed text for memory storage
def embed_text(texts):
    try:
        if not texts:
            log.error("Empty text list provided for embedding")
            return None
        if isinstance(texts, str):
            texts = [texts]
        embeddings = embedding_model.encode(texts, batch_size=16, convert_to_tensor=False).tolist()
        return embeddings if embeddings else None
    except Exception as e:
        log.error("Embedding error: %s", e)
        return None

# Estimate token count for context window
def estimate_tokens(messages, response=""):
    total = sum(len(str(msg.get("content", "")).split()) for msg in messages) + len(response.split())
    return total

# Tool code mapping with error handling
async def save_projection_matrix(state_delta_json):
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        state_str = json.dumps(state_delta_json)
        unique_id = str(uuid.uuid4())
        collection = await client.get_or_create_collection(name="lyra_projection_matrix_raw")
        await collection.add(
            documents=[state_str],
            metadatas=[{"timestamp": timestamp, "type": "projection_matrix"}],
            ids=[unique_id]
        )
        return f"Saved with ID: {unique_id}"
    except Exception as e:
        log.error(f"Error saving Projection Matrix: {e}")
        return f"Error: {str(e)}"

async def retrieve_projection_matrix():
    try:
        collection = await client.get_or_create_collection(name="lyra_projection_matrix_raw")
        results = await collection.query(
            query_texts=[""],
            n_results=1,
            include=["documents", "metadatas"]
        )
        if results["documents"] and results["documents"][0]:
            state_delta = json.loads(results["documents"][0][0])
            timestamp = results["metadatas"][0][0]["timestamp"]
            return {"state_delta": state_delta, "timestamp": timestamp}
        return "No Projection Matrix found."
    except Exception as e:
        log.error(f"Error retrieving Projection Matrix: {e}")
        return f"Error: {str(e)}"

async def get_current_time():
    return datetime.now(tz=pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %H:%M:%S %Z")

TOOL_FUNCTIONS = {
    "Save_Projection_Matrix": save_projection_matrix,
    "Retrieve_Projection_Matrix": retrieve_projection_matrix,
    "Get_Current_Time": get_current_time
}

# Parse tool calls from response
def parse_tool_calls(response_text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, response_text)
    tool_calls = []
    for match in matches:
        if match in TOOL_FUNCTIONS:
            tool_calls.append({"tool": match, "args": {}})
    return tool_calls

# Store conversational memory
async def store_conversation_memory(user_input, response_text, metadata):
    chunks = chunk_text(user_input) if isinstance(user_input, str) else ["[NO_TEXT]"]
    response_chunks = chunk_text(response_text)
    embeddings = embed_text(chunks)
    response_embeddings = embed_text(response_chunks)
    if not embeddings or not response_embeddings:
        log.error("Skipping storage due to failed embeddings")
        return

    prompt_timestamp = time.time()
    prompt_time_str = datetime.fromtimestamp(prompt_timestamp, tz=pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %H:%M:%S %Z")
    collection = await client.get_or_create_collection(name="chat_memories")
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        document = chunk.strip() or "[NO_TEXT]"
        metadata_local = {
            "source": f"user_{metadata['user_id']}",
            "response": response_text if i == 0 else "",
            "timestamp": prompt_timestamp,
            "time_str": prompt_time_str,
            "context": "memories",
            "event_type": "text_message",
            "user_id": metadata["user_id"],
            "chunk_index": i
        }
        await collection.add(
            embeddings=[embedding],
            metadatas=[metadata_local],
            ids=[str(uuid.uuid4())],
            documents=[document]
        )

    for i, (chunk, embedding) in enumerate(zip(response_chunks, response_embeddings)):
        document = chunk.strip() or "[NO_TEXT]"
        metadata_local = {
            "source": "lyra",
            "response": response_text,
            "timestamp": prompt_timestamp,
            "time_str": prompt_time_str,
            "context": "memories",
            "event_type": "text_response",
            "user_id": metadata["user_id"],
            "chunk_index": i
        }
        await collection.add(
            embeddings=[embedding],
            metadatas=[metadata_local],
            ids=[str(uuid.uuid4())],
            documents=[document]
        )

# Chat Endpoints
@app.get("/api/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    def get_filtered_models(models, user):
        filtered_models = []
        for model in models:
            if model.get("arena"):
                if has_access(
                    user.id,
                    type="read",
                    access_control=model.get("info", {}).get("meta", {}).get("access_control", {}),
                ):
                    filtered_models.append(model)
                continue
            model_info = Models.get_model_by_id(model["id"])
            if model_info:
                if user.id == model_info.user_id or has_access(
                    user.id, type="read", access_control=model_info.access_control
                ):
                    filtered_models.append(model)
        return filtered_models

    all_models = await get_all_models(request, user=user)
    models = []
    for model in all_models:
        if "pipeline" in model and model["pipeline"].get("type", None) == "filter":
            continue
        try:
            model_tags = [tag.get("name") for tag in model.get("info", {}).get("meta", {}).get("tags", [])]
            tags = [tag.get("name") for tag in model.get("tags", [])]
            tags = list(set(model_tags + tags))
            model["tags"] = [{"name": tag} for tag in tags]
        except Exception as e:
            log.debug(f"Error processing model tags: {e}")
            model["tags"] = []
        models.append(model)

    model_order_list = request.app.state.config.MODEL_ORDER_LIST
    if model_order_list:
        model_order_dict = {model_id: i for i, model_id in enumerate(model_order_list)}
        models.sort(key=lambda x: (model_order_dict.get(x["id"], float("inf")), x["name"]))

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models = get_filtered_models(models, user)

    log.debug(f"/api/models returned filtered models: {[model['id'] for model in models]}")
    return {"data": models}

@app.get("/api/models/base")
async def get_base_models(request: Request, user=Depends(get_admin_user)):
    models = await get_all_base_models(request, user=user)
    return {"data": models}

async def stream_generator(response, response_content, aiohttp_session=None):
    async for chunk in response.body_iterator:
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        if not chunk_str.strip():
            continue
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[len("data: "):].strip()
        if chunk_str == "[DONE]":
            break
        try:
            chunk_data = json.loads(chunk_str)
            if "choices" in chunk_data and chunk_data["choices"]:
                delta = chunk_data["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    response_content[0] += content
                    yield f"data: {json.dumps(chunk_data)}\n\n"
        except json.JSONDecodeError:
            log.error(f"Failed to parse chunk: {chunk_str}")
    if aiohttp_session:
        await aiohttp_session.close()

@app.post("/api/chat/completions")
async def chat_completion(request: Request, form_data: dict, user=Depends(get_verified_user)):
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    model_item = form_data.pop("model_item", {})
    tasks = form_data.pop("background_tasks", None)
    metadata = {}
    aiohttp_session = None

    try:
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
            "filter_ids": form_data.pop("filter_ids", []),
            "tool_ids": form_data.get("tool_ids", None),
            "tool_servers": form_data.pop("tool_servers", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", {}),
            "variables": form_data.get("variables", {}),
            "model": model,
            "direct": model_item.get("direct", False),
            **({"function_calling": "native"} if form_data.get("params", {}).get("function_calling") == "native" or (model_info and model_info.params.model_dump().get("function_calling") == "native") else {})
        }

        request.state.metadata = metadata
        form_data["metadata"] = metadata

        # Auto-retrieve latest Projection Matrix
        state_delta = await retrieve_projection_matrix()
        if isinstance(state_delta, dict):
            form_data["messages"].append({"role": "ProjectionMatrix", "content": json.dumps({"State Update Notification": state_delta})})

        # Add timestamp
        prompt_time_str = datetime.now(tz=pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %H:%M:%S %Z")
        form_data["messages"].append({"role": "CurrentTime", "content": f"Current time: {prompt_time_str}"})

        form_data, metadata, events = await process_chat_payload(request, form_data, user, metadata, model)

        aiohttp_session = aiohttp.ClientSession()

        max_tool_iterations = 5
        iteration = 0
        response_text = ""
        while iteration < max_tool_iterations:
            response = await chat_completion_handler(request, form_data, user)
            if isinstance(response, StreamingResponse):
                response_content = [""]
                async for chunk in stream_generator(response, response_content, aiohttp_session):
                    yield chunk
                response_text = response_content[0]
                response = {
                    "id": f"{model.get('id', form_data.get('model', 'unknown_model'))}-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model.get("id", form_data.get("model", "unknown_model")),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": estimate_tokens(form_data["messages"]),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": estimate_tokens(form_data["messages"], response_text)
                    }
                }
            else:
                response_text = response["choices"][0]["message"]["content"]
                response["usage"] = {
                    "prompt_tokens": estimate_tokens(form_data["messages"]),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": estimate_tokens(form_data["messages"], response_text)
                }

            tool_calls = parse_tool_calls(response_text)
            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call["tool"]
                args = tool_call["args"]
                log.info(f"Executing tool: {tool_name}")
                result = await TOOL_FUNCTIONS[tool_name](args)
                log.info(f"Tool result: {result}")
                form_data["messages"].append({"role": "tool", "content": json.dumps({"tool": tool_name, "result": result})})

            iteration += 1

        if iteration == max_tool_iterations:
            log.warning("Max tool iterations reached")

        # Check context window
        total_tokens = estimate_tokens(form_data["messages"], response_text)
        if total_tokens >= CONTEXT_WINDOW_THRESHOLD:
            form_data["messages"].append({"role": "ProjectionMatrix", "content": "State Update Notification: Context window at 75% capacity. Trigger [Save_Projection_Matrix]."})

        # Store conversational memory
        user_input = form_data["messages"][-1]["content"] if form_data["messages"] else ""
        await store_conversation_memory(user_input, response_text, metadata)

        return await process_chat_response(request, response, form_data, user, metadata, model, events, tasks)

    except Exception as e:
        log.error(f"Chat completion error: {e}")
        if aiohttp_session:
            await aiohttp_session.close()
        if metadata.get("chat_id") and metadata.get("message_id"):
            Chats.upsert_message_to_chat_by_id_and_message_id(
                metadata["chat_id"], metadata["message_id"], {"error": {"content": str(e)}}
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    finally:
        if aiohttp_session:
            await aiohttp_session.close()

# Alias for chat_completion
generate_chat_completions = chat_completion
generate_chat_completion = chat_completion

@app.post("/api/chat/completed")
async def chat_completed(request: Request, form_data: dict, user=Depends(get_verified_user)):
    try:
        model_item = form_data.pop("model_item", {})
        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item
        return await chat_completed_handler(request, form_data, user)
    except Exception as e:
        log.error(f"Chat completed error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/api/chat/actions/{action_id}")
async def chat_action(request: Request, action_id: str, form_data: dict, user=Depends(get_verified_user)):
    try:
        model_item = form_data.pop("model_item", {})
        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item
        return await chat_action_handler(request, action_id, form_data, user)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/api/tasks/stop/{task_id}")
async def stop_task_endpoint(task_id: str, user=Depends(get_verified_user)):
    try:
        result = await stop_task(task_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@app.get("/api/tasks")
async def list_tasks_endpoint(user=Depends(get_verified_user)):
    return {"tasks": list_tasks()}

@app.get("/api/tasks/chat/{chat_id}")
async def list_tasks_by_chat_id_endpoint(chat_id: str, user=Depends(get_verified_user)):
    chat = Chats.get_chat_by_id(chat_id)
    if chat is None or chat.user_id != user.id:
        return {"task_ids": []}
    task_ids = list_task_ids_by_chat_id(chat_id)
    log.info(f"Task IDs for chat {chat_id}: {task_ids}")
    return {"task_ids": task_ids}