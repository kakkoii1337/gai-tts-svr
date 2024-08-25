import os

os.environ["LOG_LEVEL"]="DEBUG"
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

# GAI
from gai.lib.common.errors import *

# Router
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import APIRouter, Body, Depends
import uuid,io

router = APIRouter()
pyproject_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "..", "..", "..", "pyproject.toml")

### ----------------- TTS ----------------- ###
class TextToSpeechRequest(BaseModel):
    model: Optional[str] = "xtts-2"
    input: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: Optional[bool] = False

@router.post("/gen/v1/audio/speech")
async def _text_to_speech(request: TextToSpeechRequest = Body(...)):
    host = app.state.host
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

# __main__
if __name__ == "__main__":

    # Run self-test before anything else
    import os
    if os.environ.get("SELF_TEST",None):
        self_test_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),"self-test.py")
        import subprocess,sys
        try:
            subprocess.run([f"python {self_test_file}"],shell=True,check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(1)
        ## passed self-test

    import uvicorn
    from gai.lib.server import api_factory
    app = api_factory.create_app(pyproject_toml, category="tts")
    app.include_router(router, dependencies=[Depends(lambda: app.state.host)])
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
