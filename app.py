"""
Example algorithm inference server for Grand Challenge.

You should NOT need to edit this file. Instead, edit inference.py to implement
your algorithm's init_model() and run() functions.

Grand Challenge runs your algorithm as an HTTP server with two endpoints:

  GET  /health  — Called repeatedly after the container starts. Once this
                  returns HTTP 200 (OK), Grand Challenge considers
                  the server ready.
  POST /invoke  — Called once per case. Your algorithm should read its input
                  from /input, write its output to /output, and signal
                  completion by returning HTTP 201 (Created).

The lifecycle on Grand Challenge is:

  1. Container starts -> your server boots and loads the model
  2. /health is polled until it responds with HTTP 200 (timeout: ~5 minutes)
  3. Inputs are placed at /input
  4. POST /invoke is called
  5. Your code reads /input, does a forward pass, writes to /output,
     and finally, responds with HTTP 201
  6. Container stops and outputs are uploaded to Grand Challenge

Key points:
  - Load your model during startup (in the lifespan context), not during
    /invoke. The /invoke call has its own timeout, and spending that time
    loading a model may not leave sufficient time for processing.
  - The server must listen on port 4743.
  - /health must NOT redirect — Grand Challenge treats a redirect (HTTP 302) as a failure.
  - /invoke must respond with HTTP 201 (Created) when done. Any other response
    is treated as a failure.
  - Your container will have no internet access at runtime.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response, status
from uvicorn.config import LOGGING_CONFIG

import inference

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifespan: load the model at startup, release at shutdown."""
    global MODEL
    MODEL = inference.init_model()
    yield
    MODEL = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if MODEL is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invoke")
async def invoke():
    inference.run(MODEL)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    # Grand Challenge expects the server on port 4743 — do not change this.
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
