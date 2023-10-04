"""REST API."""

try:
    from deepparse.app.sentry import configure_sentry
    from deepparse.app.api import api, lifespan
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn


except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Ensure you installed the extra packages using: 'pip install deepparse[app]'") from e


configure_sentry()


app = FastAPI(lifespan=lifespan)

app.mount("/api", api)
app.mount("/", StaticFiles(directory="docs/_build/html", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
