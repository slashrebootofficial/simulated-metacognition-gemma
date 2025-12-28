@app.post("/api/chat/completions")
async def chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    from open_webui.utils.chat import chat_completion_handler
    from open_webui.apps.webui.models.chats import Chats
    from open_webui.utils.utils import process_chat_payload, process_chat_response
    from open_webui.apps.webui.models.models import Models
    from fastapi import HTTPException
    from open_webui.config import BYPASS_MODEL_ACCESS_CONTROL
    from open_webui.utils.models import check_model_access

    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    model_item = form_data.pop("model_item", {})
    tasks = form_data.pop("background_tasks", None)

    metadata = {}
    try:
        if not model_item.get("direct", False):
            model_id = form_data.get("model", None)
            if model_id not in request.app.state.MODELS:
                raise Exception("Model not found")

            model = request.app.state.MODELS[model_id]
            model_info = Models.get_model_by_id(model_id)

            # Check if user has access to the model
            if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
                try:
                    check_model_access(user, model)
                except Exception as e:
                    raise e
        else:
            model = model_item
            model_info = None

            request.state.direct = True
            request.state.model = model

        # Handle lyra_4:latest specifically
        if model_id == "lyra_4:latest":
            try:
                # Extract user input and context
                messages = form_data.get("messages", [])
                if not messages:
                    raise HTTPException(
                        status_code=400,
                        detail="No messages provided"
                    )
                user_input = messages[-1].get("content", "")
                context = "\n".join([msg.get("content", "") for msg in messages[:-1]])
                emotional_state = {"emotion": "neutral", "value": 0.5}
                config = DEFAULT_CONFIG.copy()

                # Call lyra_memory's process_lyra_input
                response, new_context = await process_lyra_input(user_input, context, emotional_state)

                # Format response to match Open WebUI expectations
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(user_input.split()),
                        "completion_tokens": len(response.split()),
                        "total_tokens": len(user_input.split()) + len(response.split())
                    }
                }
            except Exception as e:
                log.debug(f"Error processing Lyra request: {e}")
                if metadata.get("chat_id") and metadata.get("message_id"):
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {"error": {"content": str(e)}},
                    )
                raise HTTPException(status_code=400, detail=str(e))

        # Original logic for other models
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
            **(
                {"function_calling": "native"}
                if form_data.get("params", {}).get("function_calling") == "native"
                or (
                    model_info
                    and model_info.params.model_dump().get("function_calling")
                    == "native"
                )
                else {}
            ),
        }

        request.state.metadata = metadata
        form_data["metadata"] = metadata

        form_data, metadata, events = await process_chat_payload(
            request, form_data, user, metadata, model
        )

    except Exception as e:
        log.debug(f"Error processing chat payload: {e}")
        if metadata.get("chat_id") and metadata.get("message_id"):
            Chats.upsert_message_to_chat_by_id_and_message_id(
                metadata["chat_id"],
                metadata["message_id"],
                {
                    "error": {"content": str(e)},
                },
            )
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    try:
        response = await chat_completion_handler(request, form_data, user)
        return await process_chat_response(
            request, response, form_data, user, metadata, model, events, tasks
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

# Alias for chat_completion (Legacy)
generate_chat_completions = chat_completion
generate_chat_completion = chat_completion