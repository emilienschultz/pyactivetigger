# Fuzzy testing the API

This document describes how to fuzz the ActiveTigger API from the command line
with [Schemathesis](https://schemathesis.readthedocs.io/). Schemathesis reads
the OpenAPI schema that FastAPI generates automatically and produces
property-based tests for every endpoint: malformed payloads, boundary values,
wrong types, missing fields, unexpected methods. It then reports endpoints
that return 500 errors, violate their own response schema, or accept data
they should reject.


## ⚠️ Before you start

**Fuzzing makes real requests.** The fuzzer will actually create and delete
users, projects, schemes and annotations, hundreds of times.

- **Never** run it against a production instance or a database / `projects/`
  directory you care about.
- Use a disposable `config.yaml` (fresh SQLite database, scratch `path` and
  `path_models` directories) that you can delete afterwards.

## Prerequisites

- The API running locally (from `api/`):

  ```bash
  uv run python -m activetigger
  ```

  The schema is then available at `http://localhost:5000/api/openapi.json`.

- `uv` installed (Schemathesis itself needs no installation: `uvx` fetches it
  on the fly). Alternatively `uv add --dev schemathesis` to pin it.

## 1. Get an access token

Most routes require authentication. Without a token the fuzzer only ever
exercises the 401 path, which is useless. Grab a token for the root user:

```bash
TOKEN=$(curl -s -X POST http://localhost:5000/api/token \
  -d "username=root" -d "password=$ROOT_PASSWORD" | jq -r .access_token)
```

(`$ROOT_PASSWORD` is the same variable used by the test suite; see
`test/conftest.py` for the default dev value.)

## 2. Basic run

```bash
uvx schemathesis run http://localhost:5000/api/openapi.json \
  -H "Authorization: Bearer $TOKEN"
```

This runs all default checks (server errors, status code / content type /
response schema conformance, negative data rejection, …) on every operation
in the schema.

## 3. Exclude the expensive endpoints

Some routes enqueue long computations (BERT training, embeddings, BERTopic,
generative models). Hammering them with hundreds of generated requests will
fill the task queue and slow everything down. Exclude them:

Also exclude the routes that invalidate the fuzzer's own session: fuzzing
`POST /users/disconnect` revokes the token, and every request after it gets
401 (the run then reports "Missing authentication" on most operations).

```bash
uvx schemathesis run http://localhost:5000/api/openapi.json \
  -H "Authorization: Bearer $TOKEN" \
  --exclude-path-regex "train|predict|features/add|bertopic|generate|disconnect|changepwd|resetpwd" \
  -n 50
```

- `--exclude-path-regex` filters operations by path (can be repeated; the
  exact counterparts `--include-path`, `--exclude-path`,
  `--include-path-regex` also exist).
- `-n / --max-examples 50` caps the number of generated test cases per
  operation (default is higher; lower it for a quick pass, raise it for a
  deeper one).

To fuzz **only** one router while iterating on it:

```bash
uvx schemathesis run http://localhost:5000/api/openapi.json \
  -H "Authorization: Bearer $TOKEN" \
  --include-path-regex "^/users"
```

## 4. Useful options

| Option | Effect |
|---|---|
| `-n, --max-examples N` | Test cases per operation (quick pass: 20–50, deep pass: 200+) |
| `--max-failures N` | Stop after N failures instead of testing everything |
| `--max-time SECONDS` | Overall time budget for the run |
| `--seed N` | Reproducible runs (the seed of a run is printed in its output) |
| `--rate-limit 100/m` | Throttle requests if the dev machine struggles |
| `--phases examples,coverage,fuzzing` | Select test phases (add `stateful` for sequence testing, see below) |
| `-m, --mode negative` | Only send *invalid* data (default `all` sends valid + invalid) |
| `--report junit --report-dir fuzz-report` | Write a machine-readable report (also: `har`, `vcr`, `json`) |
| `--exclude-checks positive_data_acceptance` | Disable a specific check |

## 5. Reading the results

- **`not_a_server_error` failures (500s) are real bugs**: an unhandled
  exception in the backend, usually missing validation before a Pandas /
  file / database operation. Each failure is printed with a `curl` command
  that reproduces it exactly — replay it and check the API logs for the
  traceback.
- **`response_schema_conformance` failures** mean the actual response does
  not match the declared response model — either fix the endpoint or fix the
  model in `datamodels.py`.
- **4xx responses are fine.** Routes that need an existing `project_slug`
  will mostly get random slugs and answer 404/403: that is a valid, expected
  answer. It also means random fuzzing mainly exercises validation, not deep
  project logic (see next section).

Re-run with the printed `--seed` to reproduce a whole failing run.


## Typical session

```bash
# terminal 1 — API on scratch data
cd api && uv run python -m activetigger

# terminal 2 — fuzz
TOKEN=$(curl -s -X POST http://localhost:5000/api/token \
  -d "username=root" -d "password=$ROOT_PASSWORD" | jq -r .access_token)

uvx schemathesis run http://localhost:5000/api/openapi.json \
  -H "Authorization: Bearer $TOKEN" \
  --exclude-path-regex "train|predict|features/add|bertopic|generate|disconnect|changepwd|resetpwd" \
  -n 50 --max-failures 10
```

Fix the 500s it finds, then re-run with a higher `-n` and fewer exclusions.
