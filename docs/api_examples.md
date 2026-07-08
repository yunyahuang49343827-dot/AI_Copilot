# API Examples

Start the backend first:

```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```

Open interactive docs:

```text
http://127.0.0.1:8000/docs
```

## GET /health

```bash
curl http://127.0.0.1:8000/health
```

Returns service status, service name, and local demo mode.

## POST /qa

```bash
curl -X POST http://127.0.0.1:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"query":"公司還沒公告重大資訊，可以買股票嗎？","top_k":5}'
```

Returns a grounded answer, evidence quality note, citations, and disclaimer.

## POST /workflow-advice

```bash
curl -X POST http://127.0.0.1:8000/workflow-advice \
  -H "Content-Type: application/json" \
  -d '{"query":"關係人交易需要董事會核准嗎？","top_k":5}'
```

Returns grounded answer, risk level, risk category, reasoning, checklist, citations, evidence quality, and disclaimer.

## POST /cases

```bash
curl -X POST http://127.0.0.1:8000/cases \
  -H "Content-Type: application/json" \
  -d '{"query":"公司還沒公告重大資訊，可以買股票嗎？","requester":"demo_user","department":"Finance","top_k":5}'
```

Creates an in-memory mock compliance case for the current server runtime.

## GET /cases

```bash
curl http://127.0.0.1:8000/cases
```

Lists all mock cases created since the FastAPI server started.

## GET /cases/{case_id}

```bash
curl http://127.0.0.1:8000/cases/CASE-YYYYMMDD-HHMMSS-XXXX
```

Returns a single mock case if found. Missing cases return HTTP 404.
