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
from contextlib import asynccontextmanager
from urllib.parse import urlencode, parse_qs, urlparse
from pydantic import BaseModel
from sqlalchemy import text
from typing import Optional
from aiocache import cached
import aiohttp
import requests
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import Request, Depends, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer
from PIL import Image
import io
import uuid
from datetime import datetime
import pytz
import re
import spacy
import math
import torch
from typing import List, Dict, Optional, Any

class ChatCompletionForm(BaseModel):
    model: Optional[str] = None
    model_item: Optional[Dict[str, Any]] = {}
    chat_id: Optional[str] = None
    id: Optional[str] = None  # message_id
    session_id: Optional[str] = None
    messages: List[Dict[str, Any]]  # Messages with role/content
    tool_ids: Optional[List[str]] = None
    tool_servers: Optional[List[str]] = None
    files: Optional[List[str]] = None
    features: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    background_tasks: Optional[Any] = None

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    applications,
    BackgroundTasks,
)

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response, StreamingResponse


from open_webui.utils import logger
from open_webui.utils.audit import AuditLevel, AuditLoggingMiddleware
from open_webui.utils.logger import start_logger
from open_webui.socket.main import (
    app as socket_app,
    periodic_usage_pool_cleanup,
)
from open_webui.routers import (
    audio,
    images,
    ollama,
    openai,
    retrieval,
    pipelines,
    tasks,
    auths,
    channels,
    chats,
    folders,
    configs,
    groups,
    files,
    functions,
    memories,
    models,
    knowledge,
    prompts,
    evaluations,
    tools,
    users,
    utils,
)

from open_webui.routers.retrieval import (
    get_embedding_function,
    get_ef,
    get_rf,
)

from open_webui.internal.db import Session, engine

from open_webui.models.functions import Functions
from open_webui.models.models import Models
from open_webui.models.users import UserModel, Users
from open_webui.models.chats import Chats

from open_webui.config import (
    LICENSE_KEY,
    # Ollama
    ENABLE_OLLAMA_API,
    OLLAMA_BASE_URLS,
    OLLAMA_API_CONFIGS,
    # OpenAI
    ENABLE_OPENAI_API,
    ONEDRIVE_CLIENT_ID,
    ONEDRIVE_SHAREPOINT_URL,
    ONEDRIVE_SHAREPOINT_TENANT_ID,
    OPENAI_API_BASE_URLS,
    OPENAI_API_KEYS,
    OPENAI_API_CONFIGS,
    # Direct Connections
    ENABLE_DIRECT_CONNECTIONS,
    # Thread pool size for FastAPI/AnyIO
    THREAD_POOL_SIZE,
    # Tool Server Configs
    TOOL_SERVER_CONNECTIONS,
    # Code Execution
    ENABLE_CODE_EXECUTION,
    CODE_EXECUTION_ENGINE,
    CODE_EXECUTION_JUPYTER_URL,
    CODE_EXECUTION_JUPYTER_AUTH,
    CODE_EXECUTION_JUPYTER_AUTH_TOKEN,
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD,
    CODE_EXECUTION_JUPYTER_TIMEOUT,
    ENABLE_CODE_INTERPRETER,
    CODE_INTERPRETER_ENGINE,
    CODE_INTERPRETER_PROMPT_TEMPLATE,
    CODE_INTERPRETER_JUPYTER_URL,
    CODE_INTERPRETER_JUPYTER_AUTH,
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN,
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD,
    CODE_INTERPRETER_JUPYTER_TIMEOUT,
    # Image
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    COMFYUI_BASE_URL,
    COMFYUI_API_KEY,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
    ENABLE_IMAGE_GENERATION,
    ENABLE_IMAGE_PROMPT_GENERATION,
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGES_OPENAI_API_BASE_URL,
    IMAGES_OPENAI_API_KEY,
    IMAGES_GEMINI_API_BASE_URL,
    IMAGES_GEMINI_API_KEY,
    # Audio
    AUDIO_STT_ENGINE,
    AUDIO_STT_MODEL,
    AUDIO_STT_OPENAI_API_BASE_URL,
    AUDIO_STT_OPENAI_API_KEY,
    AUDIO_STT_AZURE_API_KEY,
    AUDIO_STT_AZURE_REGION,
    AUDIO_STT_AZURE_LOCALES,
    AUDIO_STT_AZURE_BASE_URL,
    AUDIO_STT_AZURE_MAX_SPEAKERS,
    AUDIO_TTS_API_KEY,
    AUDIO_TTS_ENGINE,
    AUDIO_TTS_MODEL,
    AUDIO_TTS_OPENAI_API_BASE_URL,
    AUDIO_TTS_OPENAI_API_KEY,
    AUDIO_TTS_SPLIT_ON,
    AUDIO_TTS_VOICE,
    AUDIO_TTS_AZURE_SPEECH_REGION,
    AUDIO_TTS_AZURE_SPEECH_BASE_URL,
    AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT,
    PLAYWRIGHT_WS_URL,
    PLAYWRIGHT_TIMEOUT,
    FIRECRAWL_API_BASE_URL,
    FIRECRAWL_API_KEY,
    WEB_LOADER_ENGINE,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
    WHISPER_LANGUAGE,
    DEEPGRAM_API_KEY,
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    # Retrieval
    RAG_TEMPLATE,
    DEFAULT_RAG_TEMPLATE,
    RAG_FULL_CONTEXT,
    BYPASS_EMBEDDING_AND_RETRIEVAL,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_ENGINE,
    RAG_RERANKING_MODEL,
    RAG_EXTERNAL_RERANKER_URL,
    RAG_EXTERNAL_RERANKER_API_KEY,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_RELEVANCE_THRESHOLD,
    RAG_ALLOWED_FILE_EXTENSIONS,
    RAG_FILE_MAX_COUNT,
    RAG_FILE_MAX_SIZE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_API_KEY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTENT_EXTRACTION_ENGINE,
    EXTERNAL_DOCUMENT_LOADER_URL,
    EXTERNAL_DOCUMENT_LOADER_API_KEY,
    TIKA_SERVER_URL,
    DOCLING_SERVER_URL,
    DOCLING_OCR_ENGINE,
    DOCLING_OCR_LANG,
    DOCLING_DO_PICTURE_DESCRIPTION,
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY,
    MISTRAL_OCR_API_KEY,
    RAG_TOP_K,
    RAG_TOP_K_RERANKER,
    RAG_TEXT_SPLITTER,
    TIKTOKEN_ENCODING_NAME,
    PDF_EXTRACT_IMAGES,
    YOUTUBE_LOADER_LANGUAGE,
    YOUTUBE_LOADER_PROXY_URL,
    # Retrieval (Web Search)
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_ENGINE,
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
    WEB_SEARCH_RESULT_COUNT,
    WEB_SEARCH_CONCURRENT_REQUESTS,
    WEB_SEARCH_TRUST_ENV,
    WEB_SEARCH_DOMAIN_FILTER_LIST,
    JINA_API_KEY,
    SEARCHAPI_API_KEY,
    SEARCHAPI_ENGINE,
    SERPAPI_API_KEY,
    SERPAPI_ENGINE,
    SEARXNG_QUERY_URL,
    YACY_QUERY_URL,
    YACY_USERNAME,
    YACY_PASSWORD,
    SERPER_API_KEY,
    SERPLY_API_KEY,
    SERPSTACK_API_KEY,
    SERPSTACK_HTTPS,
    TAVILY_API_KEY,
    TAVILY_EXTRACT_DEPTH,
    BING_SEARCH_V7_ENDPOINT,
    BING_SEARCH_V7_SUBSCRIPTION_KEY,
    BRAVE_SEARCH_API_KEY,
    EXA_API_KEY,
    PERPLEXITY_API_KEY,
    SOUGOU_API_SID,
    SOUGOU_API_SK,
    KAGI_SEARCH_API_KEY,
    MOJEEK_SEARCH_API_KEY,
    BOCHA_SEARCH_API_KEY,
    GOOGLE_PSE_API_KEY,
    GOOGLE_PSE_ENGINE_ID,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_API_KEY,
    ONEDRIVE_CLIENT_ID,
    ONEDRIVE_SHAREPOINT_URL,
    ONEDRIVE_SHAREPOINT_TENANT_ID,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    ENABLE_WEB_LOADER_SSL_VERIFICATION,
    ENABLE_GOOGLE_DRIVE_INTEGRATION,
    ENABLE_ONEDRIVE_INTEGRATION,
    UPLOAD_DIR,
    EXTERNAL_WEB_SEARCH_URL,
    EXTERNAL_WEB_SEARCH_API_KEY,
    EXTERNAL_WEB_LOADER_URL,
    EXTERNAL_WEB_LOADER_API_KEY,
    # WebUI
    WEBUI_AUTH,
    WEBUI_NAME,
    WEBUI_BANNERS,
    WEBHOOK_URL,
    ADMIN_EMAIL,
    SHOW_ADMIN_DETAILS,
    JWT_EXPIRES_IN,
    ENABLE_SIGNUP,
    ENABLE_LOGIN_FORM,
    ENABLE_API_KEY,
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS,
    API_KEY_ALLOWED_ENDPOINTS,
    ENABLE_CHANNELS,
    ENABLE_NOTES,
    ENABLE_COMMUNITY_SHARING,
    ENABLE_MESSAGE_RATING,
    ENABLE_USER_WEBHOOKS,
    ENABLE_EVALUATION_ARENA_MODELS,
    USER_PERMISSIONS,
    DEFAULT_USER_ROLE,
    PENDING_USER_OVERLAY_CONTENT,
    PENDING_USER_OVERLAY_TITLE,
    DEFAULT_PROMPT_SUGGESTIONS,
    DEFAULT_MODELS,
    DEFAULT_ARENA_MODEL,
    MODEL_ORDER_LIST,
    EVALUATION_ARENA_MODELS,
    # WebUI (OAuth)
    ENABLE_OAUTH_ROLE_MANAGEMENT,
    OAUTH_ROLES_CLAIM,
    OAUTH_EMAIL_CLAIM,
    OAUTH_PICTURE_CLAIM,
    OAUTH_USERNAME_CLAIM,
    OAUTH_ALLOWED_ROLES,
    OAUTH_ADMIN_ROLES,
    # WebUI (LDAP)
    ENABLE_LDAP,
    LDAP_SERVER_LABEL,
    LDAP_SERVER_HOST,
    LDAP_SERVER_PORT,
    LDAP_ATTRIBUTE_FOR_MAIL,
    LDAP_ATTRIBUTE_FOR_USERNAME,
    LDAP_SEARCH_FILTERS,
    LDAP_SEARCH_BASE,
    LDAP_APP_DN,
    LDAP_APP_PASSWORD,
    LDAP_USE_TLS,
    LDAP_CA_CERT_FILE,
    LDAP_VALIDATE_CERT,
    LDAP_CIPHERS,
    # Misc
    ENV,
    CACHE_DIR,
    STATIC_DIR,
    FRONTEND_BUILD_DIR,
    CORS_ALLOW_ORIGIN,
    DEFAULT_LOCALE,
    OAUTH_PROVIDERS,
    WEBUI_URL,
    RESPONSE_WATERMARK,
    # Admin
    ENABLE_ADMIN_CHAT_ACCESS,
    ENABLE_ADMIN_EXPORT,
    # Tasks
    TASK_MODEL,
    TASK_MODEL_EXTERNAL,
    ENABLE_TAGS_GENERATION,
    ENABLE_TITLE_GENERATION,
    ENABLE_SEARCH_QUERY_GENERATION,
    ENABLE_RETRIEVAL_QUERY_GENERATION,
    ENABLE_AUTOCOMPLETE_GENERATION,
    TITLE_GENERATION_PROMPT_TEMPLATE,
    TAGS_GENERATION_PROMPT_TEMPLATE,
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    QUERY_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
    AppConfig,
    reset_config,
)
from open_webui.env import (
    AUDIT_EXCLUDED_PATHS,
    AUDIT_LOG_LEVEL,
    CHANGELOG,
    REDIS_URL,
    REDIS_SENTINEL_HOSTS,
    REDIS_SENTINEL_PORT,
    GLOBAL_LOG_LEVEL,
    MAX_BODY_LOG_SIZE,
    SAFE_MODE,
    SRC_LOG_LEVELS,
    VERSION,
    WEBUI_BUILD_HASH,
    WEBUI_SECRET_KEY,
    WEBUI_SESSION_COOKIE_SAME_SITE,
    WEBUI_SESSION_COOKIE_SECURE,
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER,
    WEBUI_AUTH_TRUSTED_NAME_HEADER,
    WEBUI_AUTH_SIGNOUT_REDIRECT_URL,
    ENABLE_WEBSOCKET_SUPPORT,
    BYPASS_MODEL_ACCESS_CONTROL,
    RESET_CONFIG_ON_START,
    OFFLINE_MODE,
    ENABLE_OTEL,
    EXTERNAL_PWA_MANIFEST_URL,
    AIOHTTP_CLIENT_SESSION_SSL,
)


from open_webui.utils.models import (
    get_all_models,
    get_all_base_models,
    check_model_access,
)
from open_webui.utils.chat import (
    generate_chat_completion as chat_completion_handler,
    chat_completed as chat_completed_handler,
    chat_action as chat_action_handler,
)
from open_webui.utils.middleware import process_chat_payload, process_chat_response
from open_webui.utils.access_control import has_access

from open_webui.utils.auth import (
    get_license_data,
    get_http_authorization_cred,
    decode_token,
    get_admin_user,
    get_verified_user,
)
from open_webui.utils.plugin import install_tool_and_function_dependencies
from open_webui.utils.oauth import OAuthManager
from open_webui.utils.security_headers import SecurityHeadersMiddleware

from open_webui.tasks import (
    list_task_ids_by_chat_id,
    stop_task,
    list_tasks,
)  # Import from tasks.py

from open_webui.utils.redis import get_sentinels_from_env


if SAFE_MODE:
    print("SAFE MODE ENABLED")
    Functions.deactivate_all_functions()

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                if path.endswith(".js"):
                    # Return 404 for javascript files
                    raise ex
                else:
                    return await super().get_response("index.html", scope)
            else:
                raise ex


print(
    rf"""
 ██████╗ ██████╗ ███████╗███╗   ██╗    ██╗    ██╗███████╗██████╗ ██╗   ██╗██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██║    ██║██╔════╝██╔══██╗██║   ██║██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ██║ █╗ ██║█████╗  ██████╔╝██║   ██║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██║███╗██║██╔══╝  ██╔══██╗██║   ██║██║
╚██████╔╝██║     ███████╗██║ ╚████║    ╚███╔███╔╝███████╗██████╔╝╚██████╔╝██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝     ╚══╝╚══╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝
v{VERSION} - building the best open-source AI user interface.
{f"Commit: {WEBUI_BUILD_HASH}" if WEBUI_BUILD_HASH != "dev-build" else ""}
https://github.com/open-webui/open-webui
"""
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_logger()
    if RESET_CONFIG_ON_START:
        reset_config()
    if LICENSE_KEY:
        get_license_data(app, LICENSE_KEY)
    asyncio.create_task(periodic_usage_pool_cleanup())
    yield

app = FastAPI(
    title="Open WebUI",
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
    lifespan=lifespan,
)

oauth_manager = OAuthManager(app)

app.state.config = AppConfig(
    redis_url=REDIS_URL,
    redis_sentinels=get_sentinels_from_env(REDIS_SENTINEL_HOSTS, REDIS_SENTINEL_PORT),
)

app.state.WEBUI_NAME = WEBUI_NAME
app.state.LICENSE_METADATA = None

########################################
#
# OPENTELEMETRY
#
########################################

if ENABLE_OTEL:
    from open_webui.utils.telemetry.setup import setup as setup_opentelemetry
    setup_opentelemetry(app=app, db_engine=engine)
########################################
#
# OLLAMA
#
########################################


app.state.config.ENABLE_OLLAMA_API = ENABLE_OLLAMA_API
app.state.config.OLLAMA_BASE_URLS = OLLAMA_BASE_URLS
app.state.config.OLLAMA_API_CONFIGS = OLLAMA_API_CONFIGS

app.state.OLLAMA_MODELS = {}

########################################
#
# OPENAI
#
########################################

app.state.config.ENABLE_OPENAI_API = ENABLE_OPENAI_API
app.state.config.OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS
app.state.config.OPENAI_API_KEYS = OPENAI_API_KEYS
app.state.config.OPENAI_API_CONFIGS = OPENAI_API_CONFIGS

app.state.OPENAI_MODELS = {}

########################################
#
# TOOL SERVERS
#
########################################

app.state.config.TOOL_SERVER_CONNECTIONS = TOOL_SERVER_CONNECTIONS
app.state.TOOL_SERVERS = []

########################################
#
# DIRECT CONNECTIONS
#
########################################

app.state.config.ENABLE_DIRECT_CONNECTIONS = ENABLE_DIRECT_CONNECTIONS

########################################
#
# WEBUI
#
########################################

app.state.config.WEBUI_URL = WEBUI_URL
app.state.config.ENABLE_SIGNUP = ENABLE_SIGNUP
app.state.config.ENABLE_LOGIN_FORM = ENABLE_LOGIN_FORM

app.state.config.ENABLE_API_KEY = ENABLE_API_KEY
app.state.config.ENABLE_API_KEY_ENDPOINT_RESTRICTIONS = (
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS
)
app.state.config.API_KEY_ALLOWED_ENDPOINTS = API_KEY_ALLOWED_ENDPOINTS

app.state.config.JWT_EXPIRES_IN = JWT_EXPIRES_IN

app.state.config.SHOW_ADMIN_DETAILS = SHOW_ADMIN_DETAILS
app.state.config.ADMIN_EMAIL = ADMIN_EMAIL


app.state.config.DEFAULT_MODELS = DEFAULT_MODELS
app.state.config.DEFAULT_PROMPT_SUGGESTIONS = DEFAULT_PROMPT_SUGGESTIONS
app.state.config.DEFAULT_USER_ROLE = DEFAULT_USER_ROLE

app.state.config.PENDING_USER_OVERLAY_CONTENT = PENDING_USER_OVERLAY_CONTENT
app.state.config.PENDING_USER_OVERLAY_TITLE = PENDING_USER_OVERLAY_TITLE

app.state.config.RESPONSE_WATERMARK = RESPONSE_WATERMARK

app.state.config.USER_PERMISSIONS = USER_PERMISSIONS
app.state.config.WEBHOOK_URL = WEBHOOK_URL
app.state.config.BANNERS = WEBUI_BANNERS
app.state.config.MODEL_ORDER_LIST = MODEL_ORDER_LIST


app.state.config.ENABLE_CHANNELS = ENABLE_CHANNELS
app.state.config.ENABLE_NOTES = ENABLE_NOTES
app.state.config.ENABLE_COMMUNITY_SHARING = ENABLE_COMMUNITY_SHARING
app.state.config.ENABLE_MESSAGE_RATING = ENABLE_MESSAGE_RATING
app.state.config.ENABLE_USER_WEBHOOKS = ENABLE_USER_WEBHOOKS

app.state.config.ENABLE_EVALUATION_ARENA_MODELS = ENABLE_EVALUATION_ARENA_MODELS
app.state.config.EVALUATION_ARENA_MODELS = EVALUATION_ARENA_MODELS

app.state.config.OAUTH_USERNAME_CLAIM = OAUTH_USERNAME_CLAIM
app.state.config.OAUTH_PICTURE_CLAIM = OAUTH_PICTURE_CLAIM
app.state.config.OAUTH_EMAIL_CLAIM = OAUTH_EMAIL_CLAIM

app.state.config.ENABLE_OAUTH_ROLE_MANAGEMENT = ENABLE_OAUTH_ROLE_MANAGEMENT
app.state.config.OAUTH_ROLES_CLAIM = OAUTH_ROLES_CLAIM
app.state.config.OAUTH_ALLOWED_ROLES = OAUTH_ALLOWED_ROLES
app.state.config.OAUTH_ADMIN_ROLES = OAUTH_ADMIN_ROLES

app.state.config.ENABLE_LDAP = ENABLE_LDAP
app.state.config.LDAP_SERVER_LABEL = LDAP_SERVER_LABEL
app.state.config.LDAP_SERVER_HOST = LDAP_SERVER_HOST
app.state.config.LDAP_SERVER_PORT = LDAP_SERVER_PORT
app.state.config.LDAP_ATTRIBUTE_FOR_MAIL = LDAP_ATTRIBUTE_FOR_MAIL
app.state.config.LDAP_ATTRIBUTE_FOR_USERNAME = LDAP_ATTRIBUTE_FOR_USERNAME
app.state.config.LDAP_APP_DN = LDAP_APP_DN
app.state.config.LDAP_APP_PASSWORD = LDAP_APP_PASSWORD
app.state.config.LDAP_SEARCH_BASE = LDAP_SEARCH_BASE
app.state.config.LDAP_SEARCH_FILTERS = LDAP_SEARCH_FILTERS
app.state.config.LDAP_USE_TLS = LDAP_USE_TLS
app.state.config.LDAP_CA_CERT_FILE = LDAP_CA_CERT_FILE
app.state.config.LDAP_VALIDATE_CERT = LDAP_VALIDATE_CERT
app.state.config.LDAP_CIPHERS = LDAP_CIPHERS


app.state.AUTH_TRUSTED_EMAIL_HEADER = WEBUI_AUTH_TRUSTED_EMAIL_HEADER
app.state.AUTH_TRUSTED_NAME_HEADER = WEBUI_AUTH_TRUSTED_NAME_HEADER
app.state.WEBUI_AUTH_SIGNOUT_REDIRECT_URL = WEBUI_AUTH_SIGNOUT_REDIRECT_URL
app.state.EXTERNAL_PWA_MANIFEST_URL = EXTERNAL_PWA_MANIFEST_URL

app.state.USER_COUNT = None
app.state.TOOLS = {}
app.state.FUNCTIONS = {}

########################################
#
# RETRIEVAL
#
########################################


app.state.config.TOP_K = RAG_TOP_K
app.state.config.TOP_K_RERANKER = RAG_TOP_K_RERANKER
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.config.ALLOWED_FILE_EXTENSIONS = RAG_ALLOWED_FILE_EXTENSIONS
app.state.config.FILE_MAX_SIZE = RAG_FILE_MAX_SIZE
app.state.config.FILE_MAX_COUNT = RAG_FILE_MAX_COUNT


app.state.config.RAG_FULL_CONTEXT = RAG_FULL_CONTEXT
app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = BYPASS_EMBEDDING_AND_RETRIEVAL
app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION = ENABLE_WEB_LOADER_SSL_VERIFICATION

app.state.config.CONTENT_EXTRACTION_ENGINE = CONTENT_EXTRACTION_ENGINE
app.state.config.EXTERNAL_DOCUMENT_LOADER_URL = EXTERNAL_DOCUMENT_LOADER_URL
app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY = EXTERNAL_DOCUMENT_LOADER_API_KEY
app.state.config.TIKA_SERVER_URL = TIKA_SERVER_URL
app.state.config.DOCLING_SERVER_URL = DOCLING_SERVER_URL
app.state.config.DOCLING_OCR_ENGINE = DOCLING_OCR_ENGINE
app.state.config.DOCLING_OCR_LANG = DOCLING_OCR_LANG
app.state.config.DOCLING_DO_PICTURE_DESCRIPTION = DOCLING_DO_PICTURE_DESCRIPTION
app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = DOCUMENT_INTELLIGENCE_ENDPOINT
app.state.config.DOCUMENT_INTELLIGENCE_KEY = DOCUMENT_INTELLIGENCE_KEY
app.state.config.MISTRAL_OCR_API_KEY = MISTRAL_OCR_API_KEY

app.state.config.TEXT_SPLITTER = RAG_TEXT_SPLITTER
app.state.config.TIKTOKEN_ENCODING_NAME = TIKTOKEN_ENCODING_NAME

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_EMBEDDING_BATCH_SIZE = RAG_EMBEDDING_BATCH_SIZE

app.state.config.RAG_RERANKING_ENGINE = RAG_RERANKING_ENGINE
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_EXTERNAL_RERANKER_URL = RAG_EXTERNAL_RERANKER_URL
app.state.config.RAG_EXTERNAL_RERANKER_API_KEY = RAG_EXTERNAL_RERANKER_API_KEY

app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.RAG_OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.RAG_OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.RAG_OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL
app.state.config.RAG_OLLAMA_API_KEY = RAG_OLLAMA_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.config.YOUTUBE_LOADER_PROXY_URL = YOUTUBE_LOADER_PROXY_URL


app.state.config.ENABLE_WEB_SEARCH = ENABLE_WEB_SEARCH
app.state.config.WEB_SEARCH_ENGINE = WEB_SEARCH_ENGINE
app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST = WEB_SEARCH_DOMAIN_FILTER_LIST
app.state.config.WEB_SEARCH_RESULT_COUNT = WEB_SEARCH_RESULT_COUNT
app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS = WEB_SEARCH_CONCURRENT_REQUESTS
app.state.config.WEB_LOADER_ENGINE = WEB_LOADER_ENGINE
app.state.config.WEB_SEARCH_TRUST_ENV = WEB_SEARCH_TRUST_ENV
app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
)

app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = ENABLE_GOOGLE_DRIVE_INTEGRATION
app.state.config.ENABLE_ONEDRIVE_INTEGRATION = ENABLE_ONEDRIVE_INTEGRATION
app.state.config.SEARXNG_QUERY_URL = SEARXNG_QUERY_URL
app.state.config.YACY_QUERY_URL = YACY_QUERY_URL
app.state.config.YACY_USERNAME = YACY_USERNAME
app.state.config.YACY_PASSWORD = YACY_PASSWORD
app.state.config.GOOGLE_PSE_API_KEY = GOOGLE_PSE_API_KEY
app.state.config.GOOGLE_PSE_ENGINE_ID = GOOGLE_PSE_ENGINE_ID
app.state.config.BRAVE_SEARCH_API_KEY = BRAVE_SEARCH_API_KEY
app.state.config.KAGI_SEARCH_API_KEY = KAGI_SEARCH_API_KEY
app.state.config.MOJEEK_SEARCH_API_KEY = MOJEEK_SEARCH_API_KEY
app.state.config.BOCHA_SEARCH_API_KEY = BOCHA_SEARCH_API_KEY
app.state.config.SERPSTACK_API_KEY = SERPSTACK_API_KEY
app.state.config.SERPSTACK_HTTPS = SERPSTACK_HTTPS
app.state.config.SERPER_API_KEY = SERPER_API_KEY
app.state.config.SERPLY_API_KEY = SERPLY_API_KEY
app.state.config.TAVILY_API_KEY = TAVILY_API_KEY
app.state.config.SEARCHAPI_API_KEY = SEARCHAPI_API_KEY
app.state.config.SEARCHAPI_ENGINE = SEARCHAPI_ENGINE
app.state.config.SERPAPI_API_KEY = SERPAPI_API_KEY
app.state.config.SERPAPI_ENGINE = SERPAPI_ENGINE
app.state.config.JINA_API_KEY = JINA_API_KEY
app.state.config.BING_SEARCH_V7_ENDPOINT = BING_SEARCH_V7_ENDPOINT
app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = BING_SEARCH_V7_SUBSCRIPTION_KEY
app.state.config.EXA_API_KEY = EXA_API_KEY
app.state.config.PERPLEXITY_API_KEY = PERPLEXITY_API_KEY
app.state.config.SOUGOU_API_SID = SOUGOU_API_SID
app.state.config.SOUGOU_API_SK = SOUGOU_API_SK
app.state.config.EXTERNAL_WEB_SEARCH_URL = EXTERNAL_WEB_SEARCH_URL
app.state.config.EXTERNAL_WEB_SEARCH_API_KEY = EXTERNAL_WEB_SEARCH_API_KEY
app.state.config.EXTERNAL_WEB_LOADER_URL = EXTERNAL_WEB_LOADER_URL
app.state.config.EXTERNAL_WEB_LOADER_API_KEY = EXTERNAL_WEB_LOADER_API_KEY


app.state.config.PLAYWRIGHT_WS_URL = PLAYWRIGHT_WS_URL
app.state.config.PLAYWRIGHT_TIMEOUT = PLAYWRIGHT_TIMEOUT
app.state.config.FIRECRAWL_API_BASE_URL = FIRECRAWL_API_BASE_URL
app.state.config.FIRECRAWL_API_KEY = FIRECRAWL_API_KEY
app.state.config.TAVILY_EXTRACT_DEPTH = TAVILY_EXTRACT_DEPTH

app.state.EMBEDDING_FUNCTION = None
app.state.ef = None
app.state.rf = None

app.state.YOUTUBE_LOADER_TRANSLATION = None


try:
    app.state.ef = get_ef(
        app.state.config.RAG_EMBEDDING_ENGINE,
        app.state.config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    )

    app.state.rf = get_rf(
        app.state.config.RAG_RERANKING_ENGINE,
        app.state.config.RAG_RERANKING_MODEL,
        app.state.config.RAG_EXTERNAL_RERANKER_URL,
        app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        RAG_RERANKING_MODEL_AUTO_UPDATE,
    )
except Exception as e:
    log.error(f"Error updating models: {e}")
    pass


app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.config.RAG_EMBEDDING_ENGINE,
    app.state.config.RAG_EMBEDDING_MODEL,
    app.state.ef,
    (
        app.state.config.RAG_OPENAI_API_BASE_URL
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_BASE_URL
    ),
    (
        app.state.config.RAG_OPENAI_API_KEY
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_API_KEY
    ),
    app.state.config.RAG_EMBEDDING_BATCH_SIZE,
)

########################################
#
# CODE EXECUTION
#
########################################

app.state.config.ENABLE_CODE_EXECUTION = ENABLE_CODE_EXECUTION
app.state.config.CODE_EXECUTION_ENGINE = CODE_EXECUTION_ENGINE
app.state.config.CODE_EXECUTION_JUPYTER_URL = CODE_EXECUTION_JUPYTER_URL
app.state.config.CODE_EXECUTION_JUPYTER_AUTH = CODE_EXECUTION_JUPYTER_AUTH
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_TOKEN = CODE_EXECUTION_JUPYTER_AUTH_TOKEN
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_PASSWORD = (
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_EXECUTION_JUPYTER_TIMEOUT = CODE_EXECUTION_JUPYTER_TIMEOUT

app.state.config.ENABLE_CODE_INTERPRETER = ENABLE_CODE_INTERPRETER
app.state.config.CODE_INTERPRETER_ENGINE = CODE_INTERPRETER_ENGINE
app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE = CODE_INTERPRETER_PROMPT_TEMPLATE

app.state.config.CODE_INTERPRETER_JUPYTER_URL = CODE_INTERPRETER_JUPYTER_URL
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH = CODE_INTERPRETER_JUPYTER_AUTH
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN = (
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
)
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD = (
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT = CODE_INTERPRETER_JUPYTER_TIMEOUT

########################################
#
# IMAGES
#
########################################

app.state.config.IMAGE_GENERATION_ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLE_IMAGE_GENERATION = ENABLE_IMAGE_GENERATION
app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = ENABLE_IMAGE_PROMPT_GENERATION

app.state.config.IMAGES_OPENAI_API_BASE_URL = IMAGES_OPENAI_API_BASE_URL
app.state.config.IMAGES_OPENAI_API_KEY = IMAGES_OPENAI_API_KEY

app.state.config.IMAGES_GEMINI_API_BASE_URL = IMAGES_GEMINI_API_BASE_URL
app.state.config.IMAGES_GEMINI_API_KEY = IMAGES_GEMINI_API_KEY

app.state.config.IMAGE_GENERATION_MODEL = IMAGE_GENERATION_MODEL

app.state.config.AUTOMATIC1111_BASE_URL = AUTOMATIC1111_BASE_URL
app.state.config.AUTOMATIC1111_API_AUTH = AUTOMATIC1111_API_AUTH
app.state.config.AUTOMATIC1111_CFG_SCALE = AUTOMATIC1111_CFG_SCALE
app.state.config.AUTOMATIC1111_SAMPLER = AUTOMATIC1111_SAMPLER
app.state.config.AUTOMATIC1111_SCHEDULER = AUTOMATIC1111_SCHEDULER
app.state.config.COMFYUI_BASE_URL = COMFYUI_BASE_URL
app.state.config.COMFYUI_API_KEY = COMFYUI_API_KEY
app.state.config.COMFYUI_WORKFLOW = COMFYUI_WORKFLOW
app.state.config.COMFYUI_WORKFLOW_NODES = COMFYUI_WORKFLOW_NODES

app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS


########################################
#
# AUDIO
#
########################################

app.state.config.STT_OPENAI_API_BASE_URL = AUDIO_STT_OPENAI_API_BASE_URL
app.state.config.STT_OPENAI_API_KEY = AUDIO_STT_OPENAI_API_KEY
app.state.config.STT_ENGINE = AUDIO_STT_ENGINE
app.state.config.STT_MODEL = AUDIO_STT_MODEL

app.state.config.WHISPER_MODEL = WHISPER_MODEL
app.state.config.WHISPER_VAD_FILTER = WHISPER_VAD_FILTER
app.state.config.DEEPGRAM_API_KEY = DEEPGRAM_API_KEY

app.state.config.AUDIO_STT_AZURE_API_KEY = AUDIO_STT_AZURE_API_KEY
app.state.config.AUDIO_STT_AZURE_REGION = AUDIO_STT_AZURE_REGION
app.state.config.AUDIO_STT_AZURE_LOCALES = AUDIO_STT_AZURE_LOCALES
app.state.config.AUDIO_STT_AZURE_BASE_URL = AUDIO_STT_AZURE_BASE_URL
app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS = AUDIO_STT_AZURE_MAX_SPEAKERS

app.state.config.TTS_OPENAI_API_BASE_URL = AUDIO_TTS_OPENAI_API_BASE_URL
app.state.config.TTS_OPENAI_API_KEY = AUDIO_TTS_OPENAI_API_KEY
app.state.config.TTS_ENGINE = AUDIO_TTS_ENGINE
app.state.config.TTS_MODEL = AUDIO_TTS_MODEL
app.state.config.TTS_VOICE = AUDIO_TTS_VOICE
app.state.config.TTS_API_KEY = AUDIO_TTS_API_KEY
app.state.config.TTS_SPLIT_ON = AUDIO_TTS_SPLIT_ON


app.state.config.TTS_AZURE_SPEECH_REGION = AUDIO_TTS_AZURE_SPEECH_REGION
app.state.config.TTS_AZURE_SPEECH_BASE_URL = AUDIO_TTS_AZURE_SPEECH_BASE_URL
app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT


app.state.faster_whisper_model = None
app.state.speech_synthesiser = None
app.state.speech_speaker_embeddings_dataset = None


########################################
#
# TASKS
#
########################################


app.state.config.TASK_MODEL = TASK_MODEL
app.state.config.TASK_MODEL_EXTERNAL = TASK_MODEL_EXTERNAL


app.state.config.ENABLE_SEARCH_QUERY_GENERATION = ENABLE_SEARCH_QUERY_GENERATION
app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION = ENABLE_RETRIEVAL_QUERY_GENERATION
app.state.config.ENABLE_AUTOCOMPLETE_GENERATION = ENABLE_AUTOCOMPLETE_GENERATION
app.state.config.ENABLE_TAGS_GENERATION = ENABLE_TAGS_GENERATION
app.state.config.ENABLE_TITLE_GENERATION = ENABLE_TITLE_GENERATION


app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE = TITLE_GENERATION_PROMPT_TEMPLATE
app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE = TAGS_GENERATION_PROMPT_TEMPLATE
app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE = (
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
)

app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = (
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
)
app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE = QUERY_GENERATION_PROMPT_TEMPLATE
app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE = (
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE
)
app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH = (
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
)


########################################
#
# WEBUI
#
########################################

app.state.MODELS = {}

class RedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "GET":
            path = request.url.path
            query_params = dict(parse_qs(urlparse(str(request.url)).query))
            if path.endswith("/watch") and "v" in query_params:
                video_id = query_params["v"][0]
                encoded_video_id = urlencode({"youtube": video_id})
                redirect_url = f"/?{encoded_video_id}"
                return RedirectResponse(url=redirect_url)
        response = await call_next(request)
        return response

app.add_middleware(RedirectMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

@app.middleware("http")
async def commit_session_after_request(request: Request, call_next):
    response = await call_next(request)
    try:
        Session.commit()
    except Exception as e:
        log.error(f"Session commit failed: {e}")
        Session.rollback()
    return response

@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    request.state.token = get_http_authorization_cred(request.headers.get("Authorization"))
    request.state.enable_api_key = app.state.config.ENABLE_API_KEY
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def inspect_websocket(request: Request, call_next):
    if "/ws/socket.io" in request.url.path and request.query_params.get("transport") == "websocket":
        upgrade = (request.headers.get("Upgrade") or "").lower()
        connection = (request.headers.get("Connection") or "").lower().split(",")
        if upgrade != "websocket" or "upgrade" not in connection:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid WebSocket upgrade request"},
            )
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ws", socket_app)
app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
app.include_router(openai.router, prefix="/openai", tags=["openai"])
app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])
app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])
app.include_router(auths.router, prefix="/api/v1/auths", tags=["auths"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["folders"])
app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(functions.router, prefix="/api/v1/functions", tags=["functions"])
app.include_router(evaluations.router, prefix="/api/v1/evaluations", tags=["evaluations"])
app.include_router(utils.router, prefix="/api/v1/utils", tags=["utils"])

try:
    audit_level = AuditLevel(AUDIT_LOG_LEVEL)
except ValueError as e:
    logger.error(f"Invalid audit level: {AUDIT_LOG_LEVEL}. Error: {e}")
    audit_level = AuditLevel.NONE

if audit_level != AuditLevel.NONE:
    app.add_middleware(
        AuditLoggingMiddleware,
        audit_level=audit_level,
        excluded_paths=AUDIT_EXCLUDED_PATHS,
        max_body_size=MAX_BODY_LOG_SIZE,
    )

# Initialize models
start_time = time.time()
try:
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
        log.info("Text embedding model loaded on CUDA")
    except Exception as e:
        log.error(f"Failed to load text embedding model on CUDA: {e}")
        try:
            embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
            log.info("Fallback to CPU for text embedding model")
        except Exception as e:
            log.error(f"Failed to load text embedding model on CPU: {e}")
            embedding_model = None
    if embedding_model is None:
        raise RuntimeError("Failed to load SentenceTransformer model")

    try:
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda')
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        log.info("CLIP model loaded on CUDA")
    except Exception as e:
        log.error(f"Failed to load CLIP model on CUDA: {e}")
        try:
            clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cpu')
            clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            log.info("Fallback to CPU for CLIP model")
        except Exception as e:
            log.error(f"Failed to load CLIP model on CPU: {e}")
            clip_model = None
            clip_processor = None
    if clip_model is None or clip_processor is None:
        raise RuntimeError("Failed to load CLIP model or processor")

    try:
        blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to('cuda')
        blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        log.info("BLIP model loaded on CUDA")
    except Exception as e:
        log.error(f"Failed to load BLIP model on CUDA: {e}")
        try:
            blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to('cpu')
            blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            log.info("Fallback to CPU for BLIP model")
        except Exception as e:
            log.error(f"Failed to load BLIP model on CPU: {e}")
            blip_model = None
            blip_processor = None
    if blip_model is None or blip_processor is None:
        raise RuntimeError("Failed to load BLIP model or processor")

    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        log.info("Tokenizer for all-mpnet-base-v2 loaded")
    except Exception as e:
        log.error(f"Failed to load tokenizer: {e}")
        tokenizer = None
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer for all-mpnet-base-v2")

    try:
        nlp = spacy.load("en_core_web_lg")
        log.info("SpaCy en_core_web_lg loaded")
    except Exception as e:
        log.error(f"Failed to load SpaCy model: {e}")
        nlp = None
    if nlp is None:
        raise RuntimeError("Failed to load SpaCy model")
finally:
    log.info(f"Models loaded in {time.time() - start_time:.2f} seconds")

#ChromaDB startup event
@app.on_event("startup")
async def startup_event():
    global chroma_client
    chroma_host = app.state.config.CHROMA_HTTP_HOST or 'chromadb'
    chroma_port = app.state.config.CHROMA_HTTP_PORT or 8000

    async def test_chroma_connection(client):
        try:
            async with client as c:
                await c.heartbeat()
            return True
        except Exception as e:
            log.error(f"ChromaDB heartbeat failed: {e}")
            return False

    try:
        chroma_client = chromadb.AsyncHttpClient(
            host=chroma_host,
            port=chroma_port,
            tenant="default_tenant",
            database="default_database",
            settings=ChromaSettings()
        )
        if not await test_chroma_connection(chroma_client):
            raise RuntimeError("ChromaDB connection failed: Heartbeat check unsuccessful")
        log.info("ChromaDB client initialized")
    except Exception as e:
        log.error(f"Failed to initialize ChromaDB client: {e}")
        raise RuntimeError(f"ChromaDB connection failed: {e}")

# Utility functions for embedding, captioning, and entity extraction
async def embed_text(texts):
    try:
        embeddings = await asyncio.to_thread(embedding_model.encode, texts, convert_to_tensor=False)
        embeddings = embeddings.tolist()
        if not embeddings or len(embeddings[0]) != 768:
            raise ValueError(f"Invalid text embedding dimension: {len(embeddings[0]) if embeddings else 'None'}")
        return embeddings
    except Exception as e:
        log.error(f"Text embedding error: {e}")
        raise

async def embed_image(image):
    try:
        inputs = await asyncio.to_thread(clip_processor, images=image, return_tensors="pt")
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            embeddings = await asyncio.to_thread(clip_model.get_image_features, **inputs)
            embeddings = embeddings.cpu().numpy().tolist()[0]
        if len(embeddings) != 512:
            raise ValueError(f"Invalid image embedding dimension: {len(embeddings)}")
        return embeddings
    except Exception as e:
        log.error(f"Image embedding error: {e}")
        raise

async def generate_caption(image):
    try:
        inputs = await asyncio.to_thread(blip_processor, images=image, return_tensors="pt")
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            out = await asyncio.to_thread(blip_model.generate, **inputs)
        caption = await asyncio.to_thread(blip_processor.decode, out[0], skip_special_tokens=True)
        log.debug(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        log.error(f"Caption generation error: {e}")
        return "Caption unavailable"

async def extract_entities(text):
    if not nlp or not text.strip():
        return []
    try:
        doc = await asyncio.to_thread(nlp, text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    except Exception as e:
        log.error(f"Entity extraction error: {e}")
        return []

async def count_tokens(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    if not tokenizer:
        log.warning("Tokenizer unavailable, falling back to word count")
        return len(text.split())
    try:
        tokens = await asyncio.to_thread(tokenizer.encode, text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        log.error(f"Token counting error: {e}")
        return len(text.split())

# Process structured input, handling text and images
async def process_structured_input(input_data):
    if isinstance(input_data, list):
        filtered_text = ""
        image_file = None
        for item in input_data:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    filtered_text += item['text'] + " "
                elif item.get('type') == 'image_url' and 'image_url' in item and 'url' in item['image_url']:
                    filtered_text += "[IMAGE_PROVIDED] "
        filtered_text = filtered_text.strip()
        return filtered_text, image_file
    if isinstance(input_data, str):
        return input_data, None
    return str(input_data), None

# Function to split text by tokens
async def split_text_by_tokens(text: str, chunk_size: int = 384, chunk_overlap: int = 64) -> List[str]:
    if not text or not text.strip():
        log.debug("Empty input text, returning [TEXT_EMPTY]")
        return ["[TEXT_EMPTY]"]
    if not tokenizer:
        log.error("Tokenizer unavailable, falling back to single chunk")
        return ["[TEXT_EMPTY]"]
    try:
        tokens = await asyncio.to_thread(tokenizer.encode, text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = await asyncio.to_thread(tokenizer.decode, chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += chunk_size - chunk_overlap
        return chunks or ["[TEXT_EMPTY]"]
    except UnicodeEncodeError as e:
        log.error(f"Unicode error in token splitting: {e}")
        return ["[TEXT_EMPTY]"]
    except Exception as e:
        log.error(f"Token splitting error: {e}")
        return ["[TEXT_EMPTY]"]

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
            pass
        models.append(model)
    model_order_list = request.app.state.config.MODEL_ORDER_LIST
    if model_order_list:
        model_order_dict = {model_id: i for i, model_id in enumerate(model_order_list)}
        models.sort(key=lambda x: (model_order_dict.get(x["id"], float("inf")), x["name"]))
    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models = get_filtered_models(models, user)
    log.debug(f"/api/models returned filtered models accessible to the user: {json.dumps([model['id'] for model in models])}")
    return {"data": models}

@app.get("/api/models/base")
async def get_base_models(request: Request, user=Depends(get_admin_user)):
    models = await get_all_base_models(request, user=user)
    return {"data": models}

async def stream_generator(response, response_content, form_data, aiohttp_session=None):
    log.debug(f"Streaming response for model: {form_data.get('model')}")
    if not hasattr(response, 'body_iterator'):
        log.error("Response has no body_iterator")
        raise ValueError("Invalid response: no body_iterator")
    iterator = response.body_iterator
    if not hasattr(iterator, '__aiter__'):
        log.error("Body_iterator is not an async iterator")
        if asyncio.iscoroutine(iterator):
            log.info("Body_iterator is a coroutine, attempting to await it")
            iterator = await iterator
            if not hasattr(iterator, '__aiter__'):
                log.error("Awaited body_iterator is still not an async iterator")
                raise ValueError("Invalid response: body_iterator is not async iterable after awaiting")
    try:
        async for chunk in iterator:
            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
            log.info("Processed chunk successfully")
            if not chunk_str.strip():
                continue
            if chunk_str.startswith("data: "):
                chunk_str = chunk_str[len("data: "):].strip()
            if chunk_str == "[DONE]":
                log.info("Received [DONE] marker, ending stream")
                break
            try:
                chunk_data = json.loads(chunk_str)  # Replace with async parsing
                if "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        response_content[0] += content
                    elif chunk_data["choices"][0].get("finish_reason"):
                        pass
                else:
                    log.warning(f"Unexpected chunk format: {chunk_str}")
            except json.JSONDecodeError:
                log.error(f"Failed to parse chunk as JSON: {chunk_str}")
                continue
    except Exception as e:
        log.error(f"Error iterating over body_iterator: {e}")
        raise
    finally:
        if aiohttp_session is not None:
            log.info("Closing provided aiohttp session")
            await aiohttp_session.close()
        if hasattr(response, 'content') and hasattr(response.content, '_session'):
            log.info("Closing response.content._session")
            await response.content._session.close()

@app.post("/api/chat/completions")
async def chat_completion(
    request: Request,
    model: Optional[str] = Form(None),
    model_item: Optional[str] = Form("{}", description="JSON-encoded dict"),  # JSON string
    chat_id: Optional[str] = Form(None),
    id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    messages: str = Form(..., description="JSON-encoded list of dicts"),  # Required, JSON string
    tool_ids: Optional[str] = Form(None, description="JSON-encoded list of strings"),
    tool_servers: Optional[str] = Form(None, description="JSON-encoded list of strings"),
    files: Optional[str] = Form(None, description="JSON-encoded list of strings"),
    features: Optional[str] = Form(None, description="JSON-encoded dict"),
    variables: Optional[str] = Form(None, description="JSON-encoded dict"),
    params: Optional[str] = Form(None, description="JSON-encoded dict"),
    background_tasks: Optional[str] = Form(None, description="JSON-encoded object"),
    image_file: Optional[UploadFile] = File(None),
    user=Depends(get_verified_user)
):
    # Parse JSON-encoded fields
    try:
        form_data = ChatCompletionForm(
            model=model,
            model_item=json.loads(model_item) if model_item else {},
            chat_id=chat_id,
            id=id,
            session_id=session_id,
            messages=json.loads(messages),
            tool_ids=json.loads(tool_ids) if tool_ids else None,
            tool_servers=json.loads(tool_servers) if tool_servers else None,
            files=json.loads(files) if files else None,
            features=json.loads(features) if features else None,
            variables=json.loads(variables) if variables else None,
            params=json.loads(params) if params else None,
            background_tasks=json.loads(background_tasks) if background_tasks else None
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in form data: {str(e)}")

    if not request.app.state.MODELS:
        await get_all_models(request, user=user)
    metadata = {}
    loop = asyncio.get_event_loop()
    aiohttp_session = None
    try:
        model_item = form_data.model_item or {}
        tasks = form_data.background_tasks
        if not model_item.get("direct", False):
            model_id = form_data.model
            if model_id not in request.app.state.MODELS:
                raise Exception("Model not found")
            model = request.app.state.MODELS[model_id]
            model_info = Models.get_model_by_id(model_id)
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
        metadata = {
            "user_id": user.id,
            "chat_id": form_data.chat_id,
            "message_id": form_data.id,
            "session_id": form_data.session_id,
            "tool_ids": form_data.tool_ids,
            "tool_servers": form_data.tool_servers,
            "files": form_data.files,
            "features": form_data.features,
            "variables": form_data.variables,
            "model": model,
            "direct": model_item.get("direct", False),
            **(
                {"function_calling": "native"}
                if form_data.params.get("function_calling") == "native"
                or (
                    model_info
                    and model_info.params.model_dump().get("function_calling")
                    == "native"
                )
                else {}
            ),
        }
        request.state.metadata = metadata
        form_data_dict = form_data.dict(exclude_unset=True)
        form_data_dict["metadata"] = metadata
        form_data_dict["stream"] = False
        form_data_dict, metadata, events = await process_chat_payload(
            request, form_data_dict, user, metadata, model
        )
        # Process input and retrieve past interactions
        prompt_timestamp = time.time()
        prompt_time_str = datetime.fromtimestamp(prompt_timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
        filtered_user_input, uploaded_image = process_structured_input(form_data_dict["messages"][-1]["content"])
        conversation_id = form_data_dict.get("chat_id", str(uuid.uuid4()))
        interaction_id = str(uuid.uuid4())
        previous_interaction_id = conversation_state.get(conversation_id, None)
        metadata["conversation_id"] = conversation_id
        metadata["interaction_id"] = interaction_id
        metadata["previous_interaction_id"] = previous_interaction_id
        conversation_state[conversation_id] = interaction_id
        log.debug(f"Updated conversation_state: chat_id={conversation_id}, interaction_id={interaction_id}")
        context = "image_related" if image_file else "general"
        past_interactions = await get_past_interactions(filtered_user_input, context=context, conversation_id=conversation_id, request=request)
        log.info(f"ChromaDB past interactions: {past_interactions[:4096]}...")
        caption = None
        image_id = None
        if image_file:
            try:
                image_content = await image_file.read()
                image = Image.open(io.BytesIO(image_content))
                image.verify()
                image = Image.open(io.BytesIO(image_content))
            except Exception as e:
                log.error(f"Invalid image file: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            caption = generate_caption(image)
            caption_entities = extract_entities(caption)
            system_state = {
                "role": "system",
                "content": f"Image caption: {caption}\nEntities: {caption_entities}"
            }
            form_data_dict["messages"].insert(0, system_state)
            past_image_interactions = await get_past_image_interactions(
                image=image,
                context="image_related",
                conversation_id=conversation_id,
                request=request,
                user=user
            )
            if past_image_interactions:
                form_data_dict["messages"].insert(1, {
                    "role": "system",
                    "content": f"Past relevant image interactions: {past_image_interactions}"
                })
        if past_interactions and past_interactions != "No relevant past interactions found.":
            form_data_dict["messages"].insert(-1, {
                "role": "system",
                "content": f"Past relevant interactions:\n{past_interactions}"
            })
        form_data_dict["messages"].insert(-1, {
            "role": "system",
            "content": f"Current prompt timestamp: {prompt_time_str}"
        })
        aiohttp_session = aiohttp.ClientSession()
        response = await chat_completion_handler(request, form_data_dict, user)
        if isinstance(response, StreamingResponse):
            response_content = [""]
            try:
                await stream_generator(response, response_content, form_data_dict, aiohttp_session)
            except Exception as e:
                log.error(f"Error processing streaming response: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing streaming response: {str(e)}")
            return {
                "id": f"lyra_4:latest-{str(uuid.uuid4())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": form_data_dict.get("model", "lyra_4:latest"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content[0]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": count_tokens(form_data_dict["messages"][-1]["content"]),
                    "completion_tokens": count_tokens(response_content[0]),
                    "total_tokens": count_tokens(form_data_dict["messages"][-1]["content"]) + count_tokens(response_content[0])
                },
                "stream": False
            }
        else:
            try:
                response_content = response["choices"][0]["message"]["content"]
                filtered_response_content = response_content
                log.info(f"Non-streaming response content: {filtered_response_content[:4096]}...")
            except (TypeError, KeyError):
                log.error(f"Unexpected response format: {response}")
                raise HTTPException(status_code=500, detail="Unexpected response format")
            try:
                client = chroma_client
                if client is None:
                    log.error("ChromaDB client unavailable")
                    return await process_chat_response(
                        request, response, form_data_dict, user, metadata, model, events, tasks
                    )
                if image_file:
                    image_collection = await client.get_or_create_collection(name="image_memories")
                    image_embedding = embed_image(image)
                    log.debug(f"Image embedding length: {len(image_embedding) if image_embedding else None}")
                    image_metadata = {
                        "caption": caption,
                        "entities": caption_entities,
                        "timestamp": prompt_timestamp,
                        "time_str": prompt_time_str,
                        "context": "image_related",
                        "event_type": "image_shared",
                        "access_count": 0,
                        "user_id": user.id,
                        "conversation_id": conversation_id,
                        "interaction_id": interaction_id,
                        "previous_interaction_id": previous_interaction_id,
                        "image_id": image_id
                    }
                    if image_embedding is None or len(image_embedding) != 512:
                        log.error(f"Invalid image embedding dimension: {len(image_embedding) if image_embedding else 'None'}")
                    else:
                        try:
                            await image_collection.add(
                                embeddings=[image_embedding],
                                metadatas=[image_metadata],
                                ids=[str(uuid.uuid4())],
                                documents=[caption],
                            )
                        except chromadb.errors.InvalidDimensionException as e:
                            log.error(f"ChromaDB image dimension mismatch: {e}")
                        except chromadb.errors.UniqueConstraintError as e:
                            log.error(f"ChromaDB image unique constraint violation: {e}")
                        except Exception as e:
                            log.error(f"Failed to store image interaction: {e}")
                        else:
                            log.info(f"Stored image memory with timestamp: {prompt_timestamp} ({prompt_time_str})")
                text_collection = await client.get_or_create_collection(name="chat_memories")
                filtered_input = filtered_user_input.strip() or "[TEXT_EMPTY]"
                input_entities = extract_entities(filtered_input) if filtered_input != "[TEXT_EMPTY]" else []
                if filtered_input != "[TEXT_EMPTY]":
                    chunks = split_text_by_tokens(
                        filtered_input,
                        chunk_size=384,
                        chunk_overlap=64
                    )
                    if not chunks:
                        chunks = [filtered_input]
                    embeddings = embed_text(chunks)[0] if chunks else None
                else:
                    chunks = [filtered_input]
                    embeddings = None
                context = "image_related" if image_file else "general"
                event_type = "image_shared" if image_file else "text_message"
                for chunk, embedding in zip(chunks, embeddings or [None]):
                    log.debug(f"Text embedding length: {len(embedding) if embedding else None}")
                    metadata = {
                        "response": filtered_response_content,
                        "timestamp": prompt_timestamp,
                        "time_str": prompt_time_str,
                        "context": context,
                        "event_type": event_type,
                        "access_count": 0,
                        "user_id": user.id,
                        "caption": caption if image_file else None,
                        "entities": input_entities,
                        "interaction_id": interaction_id,
                        "conversation_id": conversation_id,
                        "previous_interaction_id": previous_interaction_id,
                        "image_id": image_id if image_file else None
                    }
                    if embedding and len(embedding) != 768:
                        log.error(f"Invalid text embedding dimension: {len(embedding)}")
                        embedding = None
                    try:
                        await text_collection.add(
                            embeddings=[embedding] if embedding else None,
                            metadatas=[metadata],
                            ids=[str(uuid.uuid4())],
                            documents=[chunk],
                        )
                        log.info(f"Stored text chunk with timestamp: {prompt_timestamp} ({prompt_time_str})")
                    except chromadb.errors.InvalidDimensionException as e:
                        log.error(f"ChromaDB dimension mismatch: {e}")
                    except chromadb.errors.UniqueConstraintError as e:
                        log.error(f"ChromaDB unique constraint violation: {e}")
                    except Exception as e:
                        log.error(f"Failed to store text chunk: {e}")
                conversation_state[conversation_id] = interaction_id
            except Exception as e:
                log.error("ChromaDB storage error: %s", e)
            return await process_chat_response(
                request, response, form_data_dict, user, metadata, model, events, tasks
            )
    except Exception as e:
        log.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if aiohttp_session is not None:
            log.info("Closing aiohttp session in chat_completion")
            await aiohttp_session.close()

async def get_past_interactions(user_input, context="general", conversation_id=None, use_cache=True, request=None):
    log.info(f"Retrieving past interactions for input: {user_input[:100]}... with context: {context}")
    prompt_timestamp = time.time()  # Define prompt_timestamp for scoring
    cache_key = f"{conversation_id}:{user_input}" if conversation_id else user_input
    if use_cache and conversation_id in memory_cache and cache_key in memory_cache[conversation_id]:
        log.info(f"Cache hit for {cache_key}")
        return memory_cache[conversation_id][cache_key]
    try:
        client = chroma_client
        if client is None:
            log.error("ChromaDB client unavailable")
            return "ChromaDB unavailable, relying on current context."
        collection = await client.get_or_create_collection(name="chat_memories")
        embeddings = embed_text([user_input])
        if embeddings is None or not embeddings or len(embeddings[0]) != 768:  # all-mpnet-base-v2 dimension
            log.error(f"Invalid text embedding: {len(embeddings[0]) if embeddings and embeddings[0] else 'None or empty'}")
            return "No relevant past interactions found."
        query_embedding = np.mean(embeddings[0], axis=0).tolist()
        where_clause = {"context": context, "user_id": user.id}
        if conversation_id:
            where_clause["conversation_id"] = conversation_id
        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where=where_clause
        )
        past_interactions = "Retrieved Memories:\n"
        if results["documents"]:
            log.info(f"Retrieved {len(results['documents'][0])} past interactions")
            interactions = {}
            for doc, meta, distance, id in zip(results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]):
                interaction_id = meta.get("interaction_id", id)
                if interaction_id not in interactions:
                    interactions[interaction_id] = {
                        "docs": [],
                        "metas": [],
                        "scores": [],
                        "distances": []
                    }
                interactions[interaction_id]["docs"].append(doc)
                interactions[interaction_id]["metas"].append(meta)
                interactions[interaction_id]["scores"].append(1.0 - distance)
                interactions[interaction_id]["distances"].append(distance)
            # Compute weighted scores
            scored_interactions = []
            for interaction_id, data in interactions.items():
                meta = data["metas"][0]
                semantic_score = max(data["scores"]) * 0.6
                access_count = int(meta.get('access_count', 0))
                timestamp = float(meta.get('timestamp', 0))
                age_in_days = (prompt_timestamp - timestamp) / (24 * 3600)
                decayed_access_score = (access_count * math.exp(-age_in_days / 30)) * 0.2
                recent_score = 0
                if conversation_id and meta.get("previous_interaction_id"):
                    recent_results = await collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        where={"conversation_id": conversation_id, "interaction_id": meta["previous_interaction_id"]}
                    )
                    if recent_results["distances"] and recent_results["distances"][0]:
                        recent_score = (1.0 - min(recent_results["distances"][0])) * 0.1
                query_entities = [e["text"] for e in extract_entities(user_input)]
                memory_entities = [e["text"] for e in meta.get("entities", [])]
                entity_boost = sum(0.2 for e in query_entities if e in memory_entities)
                total_score = semantic_score + decayed_access_score + recent_score + min(entity_boost, 0.2)
                scored_interactions.append((interaction_id, total_score, data))
            scored_interactions.sort(key=lambda x: x[1], reverse=True)
            for interaction_id, score, data in scored_interactions[:5]:
                meta = data["metas"][0]
                timestamp = float(meta.get('timestamp', 0))
                time_str = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
                caption = meta.get('caption', 'N/A')
                access_count = int(meta.get('access_count', 0))
                entities = meta.get('entities', [])
                entity_texts = [f"{e['text']} ({e['label']})" for e in entities]
                full_doc = " ".join(data["docs"])
                past_interactions += (
                    f"Text Interaction\n"
                    f"Text: User: {full_doc}\n"
                    f"Lyra: {meta['response']}\n"
                    f"Entities: {', '.join(entity_texts)}\n"
                    f"Timestamp: {time_str}\n"
                    f"Access Count: {access_count}\n\n"
                )
                ids_to_update = [id for id, m in zip(results["ids"][0], results["metadatas"][0]) if m.get("interaction_id", id) == interaction_id]
                metadatas_to_update = [dict(m, access_count=access_count + 1) for m in data["metas"]]
                try:
                    await collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
                    log.info(f"Updated access_count for {len(ids_to_update)} documents")
                except Exception as e:
                    log.error(f"Failed to update access_count: {e}")
        else:
            log.info("No past interactions found")
            past_interactions = "No relevant past interactions found."
        if use_cache and conversation_id:
            memory_cache[conversation_id][cache_key] = past_interactions
            log.debug(f"Cached result for {cache_key}")
        return past_interactions
    except chromadb.errors.InvalidDimensionException as e:
        log.error(f"ChromaDB dimension mismatch: {e}")
        return "Embedding dimension error in ChromaDB."
    except chromadb.errors.UniqueConstraintError as e:
        log.error(f"ChromaDB unique constraint violation: {e}")
        return "Duplicate entry error in ChromaDB."
    except Exception as e:
        log.error(f"ChromaDB retrieval failed: {e}")
        return "ChromaDB unavailable, relying on current context."

async def get_past_image_interactions(image=None, caption=None, context="image_related", conversation_id=None, use_cache=True, request=None, user=None):
    log.info(f"Retrieving past image interactions for context: {context}")
    if use_cache and hasattr(request.state, 'image_memory_cache') and context in request.state.image_memory_cache:
        log.info(f"Using cached image memories for context: {context}")
        return request.state.image_memory_cache[context]
    try:
        client = chroma_client
        if client is None:
            log.error("ChromaDB client unavailable")
            return "ChromaDB unavailable for images, relying on current context."
        collection = await client.get_or_create_collection(name="image_memories")
        query_embedding = None
        if image:
            query_embedding = embed_image(image)
        elif caption:
            query_embedding = np.mean(embed_text([caption])[0], axis=0).tolist()
        if query_embedding is None:
            log.error("Failed to generate query embedding")
            return "No relevant past image interactions found."
        where_clause = {"context": context, "user_id": user.id}
        if conversation_id:
            where_clause["conversation_id"] = conversation_id
        results = await collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where=where_clause
        )
        past_image_interactions = "Retrieved Image Memories:\n"
        if results["documents"]:
            log.info(f"Retrieved {len(results['documents'][0])} past image interactions")
            ids_to_update = results["ids"][0][:3]
            metadatas_to_update = []
            for meta in results["metadatas"][0][:3]:
                access_count = int(meta.get('access_count', 0)) + 1
                meta['access_count'] = access_count
                metadatas_to_update.append(meta)
            try:
                await collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
                log.info(f"Updated access_count for {len(ids_to_update)} image documents")
            except Exception as e:
                log.error(f"Failed to update image access_count: {e}")
            for doc, meta, distance in zip(results["documents"][0][:3], results["metadatas"][0][:3], results["distances"][0][:3]):
                timestamp = float(meta.get('timestamp', 0))
                time_str = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
                caption = meta.get('caption', 'N/A')
                access_count = int(meta.get('access_count', 0))
                similarity = 1.0 - distance
                entities = meta.get('entities', [])
                entity_texts = [f"{e['text']} ({e['label']})" for e in entities]
                past_image_interactions += (
                    f"Image Interaction\n"
                    f"Caption: {caption}\n"
                    f"Entities: {', '.join(entity_texts)}\n"
                    f"Timestamp: {time_str}\n"
                    f"Similarity: {similarity:.2f}\n"
                    f"Access Count: {access_count}\n\n"
                )
        else:
            log.info("No past image interactions found")
            past_image_interactions = "No relevant past image interactions found."
        if not hasattr(request.state, 'image_memory_cache'):
            request.state.image_memory_cache = {}
        request.state.image_memory_cache[context] = past_image_interactions
        return past_image_interactions
    except chromadb.errors.InvalidDimensionException as e:
        log.error(f"ChromaDB image dimension mismatch: {e}")
        return "Image embedding dimension error in ChromaDB."
    except chromadb.errors.UniqueConstraintError as e:
        log.error(f"ChromaDB image unique constraint violation: {e}")
        return "Duplicate image entry error in ChromaDB."
    except Exception as e:
        log.error(f"ChromaDB image retrieval failed: {e}")
        return "ChromaDB unavailable for images, relying on current context."

generate_chat_completions = chat_completion
generate_chat_completion = chat_completion

@app.post("/api/chat/completed")
async def chat_completed(request: Request, form_data: dict, user=Depends(get_verified_user)):
    try:
        user_input = form_data.get("user_input")
        response_content = form_data.get("response_content")
        if user_input and response_content:
            filtered_input, _ = process_structured_input(user_input)
            filtered_response = response_content
            document = filtered_input.strip() or "[TEXT_EMPTY]"
            if not document.strip():
                log.warning("No text content to store, skipping storage")
                return await chat_completed_handler(request, form_data, user)
            client = chroma_client
            if client is None:
                log.error("ChromaDB client unavailable")
                return await chat_completed_handler(request, form_data, user)
            collection = await client.get_or_create_collection(name="chat_memories")
            embeddings = embedding_model.encode([document], convert_to_tensor=False).tolist()
            if embeddings is None or not embeddings:
                log.error("Failed to generate text embeddings")
                return await chat_completed_handler(request, form_data, user)
            prompt_timestamp = time.time()
            prompt_time_str = datetime.fromtimestamp(prompt_timestamp, tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
            context = "general"
            event_type = "text_message"
            conversation_id = form_data.get("chat_id", str(uuid.uuid4()))
            interaction_id = str(uuid.uuid4())
            metadata = {
                "response": filtered_response,
                "timestamp": prompt_timestamp,
                "time_str": prompt_time_str,
                "context": context,
                "event_type": event_type,
                "access_count": 0,
                "user_id": user.id,
                "conversation_id": conversation_id,
                "interaction_id": interaction_id
            }
            try:
                await collection.add(
                    embeddings=[embeddings[0]],
                    metadatas=[metadata],
                    ids=[str(uuid.uuid4())],
                    documents=[document],
                )
                log.info(f"Stored text memory with timestamp: {prompt_timestamp} ({prompt_time_str})")
                conversation_state[conversation_id] = interaction_id
            except chromadb.errors.InvalidDimensionException as e:
                log.error(f"ChromaDB dimension mismatch: {e}")
            except chromadb.errors.UniqueConstraintError as e:
                log.error(f"ChromaDB unique constraint violation: {e}")
            except Exception as e:
                log.error(f"Failed to store interaction: {e}")
        return await chat_completed_handler(request, form_data, user)
    except Exception as e:
        log.error(f"Chat completed error: {e}")
        return await chat_completed_handler(request, form_data, user)

@app.post("/api/chat/actions/{action_id}")
async def chat_action(request: Request, action_id: str, form_data: dict, user=Depends(get_verified_user)):
    try:
        model_item = form_data.pop("model_item", {})
        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item
        return await chat_action_handler(request, action_id, form_data, user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@app.post("/api/tasks/stop/{task_id}")
async def stop_task_endpoint(task_id: str, user=Depends(get_verified_user)):
    try:
        result = await stop_task(task_id)
        return result
    except ValueError as e:
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
    print(f"Task IDs for chat {chat_id}: {task_ids}")
    return {"task_ids": task_ids}

@app.get("/api/config")
async def get_app_config(request: Request):
    user = None
    if "token" in request.cookies:
        token = request.cookies.get("token")
        try:
            data = decode_token(token)
        except Exception as e:
            log.debug(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        if data is not None and "id" in data:
            user = Users.get_user_by_id(data["id"])

    user_count = Users.get_num_users()
    onboarding = False

    if user is None:
        onboarding = user_count == 0

    return {
        **({"onboarding": True} if onboarding else {}),
        "status": True,
        "name": app.state.WEBUI_NAME,
        "version": VERSION,
        "default_locale": str(DEFAULT_LOCALE),
        "oauth": {
            "providers": {
                name: config.get("name", name)
                for name, config in OAUTH_PROVIDERS.items()
            }
        },
        "features": {
            "auth": WEBUI_AUTH,
            "auth_trusted_header": bool(app.state.AUTH_TRUSTED_EMAIL_HEADER),
            "enable_ldap": app.state.config.ENABLE_LDAP,
            "enable_api_key": app.state.config.ENABLE_API_KEY,
            "enable_signup": app.state.config.ENABLE_SIGNUP,
            "enable_login_form": app.state.config.ENABLE_LOGIN_FORM,
            "enable_websocket": ENABLE_WEBSOCKET_SUPPORT,
            **(
                {
                    "enable_direct_connections": app.state.config.ENABLE_DIRECT_CONNECTIONS,
                    "enable_channels": app.state.config.ENABLE_CHANNELS,
                    "enable_notes": app.state.config.ENABLE_NOTES,
                    "enable_web_search": app.state.config.ENABLE_WEB_SEARCH,
                    "enable_code_execution": app.state.config.ENABLE_CODE_EXECUTION,
                    "enable_code_interpreter": app.state.config.ENABLE_CODE_INTERPRETER,
                    "enable_image_generation": app.state.config.ENABLE_IMAGE_GENERATION,
                    "enable_autocomplete_generation": app.state.config.ENABLE_AUTOCOMPLETE_GENERATION,
                    "enable_community_sharing": app.state.config.ENABLE_COMMUNITY_SHARING,
                    "enable_message_rating": app.state.config.ENABLE_MESSAGE_RATING,
                    "enable_user_webhooks": app.state.config.ENABLE_USER_WEBHOOKS,
                    "enable_admin_export": ENABLE_ADMIN_EXPORT,
                    "enable_admin_chat_access": ENABLE_ADMIN_CHAT_ACCESS,
                    "enable_google_drive_integration": app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
                    "enable_onedrive_integration": app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
                }
                if user is not None
                else {}
            ),
        },
        **(
            {
                "default_models": app.state.config.DEFAULT_MODELS,
                "default_prompt_suggestions": app.state.config.DEFAULT_PROMPT_SUGGESTIONS,
                "user_count": user_count,
                "code": {
                    "engine": app.state.config.CODE_EXECUTION_ENGINE,
                },
                "audio": {
                    "tts": {
                        "engine": app.state.config.TTS_ENGINE,
                        "voice": app.state.config.TTS_VOICE,
                        "split_on": app.state.config.TTS_SPLIT_ON,
                    },
                    "stt": {
                        "engine": app.state.config.STT_ENGINE,
                    },
                },
                "file": {
                    "max_size": app.state.config.FILE_MAX_SIZE,
                    "max_count": app.state.config.FILE_MAX_COUNT,
                },
                "permissions": {**app.state.config.USER_PERMISSIONS},
                "google_drive": {
                    "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
                    "api_key": GOOGLE_DRIVE_API_KEY.value,
                },
                "onedrive": {
                    "client_id": ONEDRIVE_CLIENT_ID.value,
                    "sharepoint_url": ONEDRIVE_SHAREPOINT_URL.value,
                    "sharepoint_tenant_id": ONEDRIVE_SHAREPOINT_TENANT_ID.value,
                },
                "ui": {
                    "pending_user_overlay_title": app.state.config.PENDING_USER_OVERLAY_TITLE,
                    "pending_user_overlay_content": app.state.config.PENDING_USER_OVERLAY_CONTENT,
                    "response_watermark": app.state.config.RESPONSE_WATERMARK,
                },
                "license_metadata": app.state.LICENSE_METADATA,
                **(
                    {
                        "active_entries": app.state.USER_COUNT,
                    }
                    if user.role == "admin"
                    else {}
                ),
            }
            if user is not None
            else {}
        ),
    }

class UrlForm(BaseModel):
    url: str

@app.get("/api/webhook")
async def get_webhook_url(user=Depends(get_admin_user)):
    return {"url": app.state.config.WEBHOOK_URL}

@app.post("/api/webhook")
async def update_webhook_url(form_data: UrlForm, user=Depends(get_admin_user)):
    app.state.config.WEBHOOK_URL = form_data.url
    app.state.WEBHOOK_URL = app.state.config.WEBHOOK_URL
    return {"url": app.state.config.WEBHOOK_URL}

@app.get("/api/version")
async def get_app_version():
    return {"version": VERSION}

@app.get("/api/version/updates")
async def get_app_latest_release_version(user=Depends(get_verified_user)):
    if OFFLINE_MODE:
        log.debug(f"Offline mode is enabled, returning current version as latest version")
        return {"current": VERSION, "latest": VERSION}
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get("https://api.github.com/repos/open-webui/open-webui/releases/latest") as response:
                response.raise_for_status()
                data = await response.json()
                latest_version = data["tag_name"]
                return {"current": VERSION, "latest": latest_version[1:]}
    except Exception as e:
        log.debug(e)
        return {"current": VERSION, "latest": VERSION}

@app.get("/api/changelog")
async def get_app_changelog():
    return {key: CHANGELOG[key] for idx, key in enumerate(CHANGELOG) if idx < 5}

if len(OAUTH_PROVIDERS) > 0:
    app.add_middleware(
        SessionMiddleware,
        secret_key=WEBUI_SECRET_KEY,
        session_cookie="oui-session",
        same_site=WEBUI_SESSION_COOKIE_SAME_SITE,
        https_only=WEBUI_SESSION_COOKIE_SECURE,
    )

@app.get("/oauth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    return await oauth_manager.handle_login(request, provider)

@app.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, response: Response):
    return await oauth_manager.handle_callback(request, provider, response)

@app.get("/manifest.json")
async def get_manifest_json():
    if app.state.EXTERNAL_PWA_MANIFEST_URL:
        return requests.get(app.state.EXTERNAL_PWA_MANIFEST_URL).json()
    else:
        return {
            "name": app.state.WEBUI_NAME,
            "short_name": app.state.WEBUI_NAME,
            "description": "Open WebUI is an open, extensible, user-friendly interface for AI that adapts to your workflow.",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#343541",
            "orientation": "natural",
            "icons": [
                {
                    "src": "/static/logo.png",
                    "type": "image/png",
                    "sizes": "500x500",
                    "purpose": "any",
                },
                {
                    "src": "/static/logo.png",
                    "type": "image/png",
                    "sizes": "500x500",
                    "purpose": "maskable",
                },
            ],
        }

@app.get("/opensearch.xml")
async def get_opensearch_xml():
    xml_content = rf"""
    <OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/" xmlns:moz="http://www.mozilla.org/2006/browser/search/">
    <ShortName>{app.state.WEBUI_NAME}</ShortName>
    <Description>Search {app.state.WEBUI_NAME}</Description>
    <InputEncoding>UTF-8</InputEncoding>
    <Image width="16" height="16" type="image/x-icon">{app.state.config.WEBUI_URL}/static/favicon.png</Image>
    <Url type="text/html" method="get" template="{app.state.config.WEBUI_URL}/?q={"{searchTerms}"}"/>
    <moz:SearchForm>{app.state.config.WEBUI_URL}</moz:SearchForm>
    </OpenSearchDescription>
    """
    return Response(content=xml_content, media_type="application/xml")

@app.get("/health")
async def healthcheck():
    return {"status": True}

@app.get("/health/db")
async def healthcheck_with_db():
    Session.execute(text("SELECT 1;")).all()
    return {"status": True}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")

def swagger_ui_html(*args, **kwargs):
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
    )

applications.get_swagger_ui_html = swagger_ui_html  # Changed from swagger_ui