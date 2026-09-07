# Professional deployment

This bundle turns the existing Stage 10 + V2 model repository into a polished single-service FastAPI website.

## Local verification

From the repository root:

```cmd
pip install -r requirements_deploy.txt
python -m uvicorn run_app:app --host 127.0.0.1 --port 8000
```

Open:
- Website: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Railway

Commit and push these files to `full-raw-pipeline`.

Railway configuration:
- Build: Dockerfile
- Healthcheck: `/health`
- Public service port: Railway-provided `$PORT`

Recommended persistent storage:
- Add a Railway volume mounted at `/data`
- Set environment variable:
  `LANDSLIDE_RUNTIME_DIR=/data/runtime`

Without a volume, the prediction map and ML model still work because they are committed in Git, but newly submitted field reports, road updates, and acknowledgement status can reset on redeploy.

## Presentation URL

After a successful Railway deploy, generate a public domain in Railway Networking. Share that HTTPS URL with judges/team members.

## Scientific wording

Use:
- “AI-generated landslide risk probability”
- “~0.806 ROC-AUC under geographic cross-validation”
- “prototype operating threshold”

Do not call ROC-AUC accuracy and do not describe this as an official warning system.
