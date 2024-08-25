# # Run self-test before anything else
# import os
# if os.environ.get("SELF_TEST",None):
#     self_test_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"self-test.py")
#     import subprocess,sys
#     try:
#         subprocess.run([f"python {self_test_file}"],shell=True,check=True)
#     except subprocess.CalledProcessError as e:
#         sys.exit(1)
#     ## passed self-test

import os
os.environ["LOG_LEVEL"]="DEBUG"
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# WEB
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
import io

# GAI
from gai.lib.common.errors import *
from gai.lib.server.singleton_host import SingletonHost
from gai.lib.common.utils import free_mem
free_mem()

# Configure Dependencies
import dependencies
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_gen installed = {dependencies.SDK_VERSION}")

# Start FASTAPI
swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
    )
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()

host = None
gai_config = None

# STARTUP
from gai.lib.common.utils import get_gai_config
DEFAULT_GENERATOR=os.getenv("DEFAULT_GENERATOR")
async def startup_event():
    global host,gai_config

    # Perform initialization here
    try:
        gai_config = get_gai_config()
        
        # default generator
        DEFAULT_GENERATOR = gai_config["gen"]["default"]["tts"]
        if os.environ.get("DEFAULT_GENERATOR",None):
            DEFAULT_GENERATOR = os.environ.get("DEFAULT_GENERATOR")
        gai_config = gai_config["gen"][DEFAULT_GENERATOR]

        # host
        host = SingletonHost.GetInstanceFromConfig(gai_config)
        host.load()

        from gai.lib.common.color import Color
        color = Color()
        color.white(text=f"Default model loaded: {DEFAULT_GENERATOR}")
        free_mem()    
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

# SHUTDOWN
async def shutdown_event():
    # Perform cleanup here
    try:
        host.unload()
    except Exception as e:
        logger.error(f"Failed to unload default model: {e}")
        raise e
app.add_event_handler("shutdown", shutdown_event)

### ----------------- TTS ----------------- ###
class TextToSpeechRequest(BaseModel):
    model: Optional[str] = "xtts-2"
    input: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: Optional[bool] = False

@app.post("/gen/v1/audio/speech")
async def _text_to_speech(request: TextToSpeechRequest = Body(...)):
    try:
        response = host.generator.create(
            voice=request.voice,
            input=request.input
            )
        return StreamingResponse(io.BytesIO(response), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"_create: error={e}")
        return JSONResponse(
            content={"_create: error=": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12032, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )    
    server = uvicorn.Server(config=config)
    server.run()
