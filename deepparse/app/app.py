"""REST API."""

try:
    from deepparse.app.sentry import configure_sentry
    from deepparse.app.api import api, lifespan
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    import os
    import warnings

except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Ensure you installed the extra packages using: 'pip install deepparse[app]'") from e


configure_sentry()


app = FastAPI(lifespan=lifespan)

app.mount("/api", api)
html_build_path = "docs/_build/html"
if os.path.exists(html_build_path):
    app.mount("/", StaticFiles(directory=html_build_path, html=True), name="static")
else:
    warnings.warn(f"Unable to mount static files, probably because the docs are not built yet: {html_build_path}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
