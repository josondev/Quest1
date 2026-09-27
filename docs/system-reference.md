# Quest1 — Complete System Reference

---

## What We Built

Quest1 is an AI-powered video dialogue detection engine. You give it a video URL and a line of dialogue. It returns the **exact timestamp**, **frame number**, **screenshot**, and **confidence score** of where that dialogue appears — whether spoken or shown as text on screen.

---

## The Problem We Solved Along the Way

YouTube blocks automated requests from cloud server IPs. We went through:

| Attempt | Result |
|---|---|
| Raw yt-dlp with no auth | ❌ Bot detection |
| Webshare free proxies | ❌ Datacenter IPs flagged |
| Azure default IP (no proxy) | ❌ Azure IPs flagged |
| IPRoyal free public proxies | ❌ Dead proxies |
| Android player_client hack | ❌ Didn't bypass auth check |
| Deno + yt-dlp EJS JS solver | ✅ Solves JS challenges |
| Fresh cookies + EJS | ✅ **Works** |

Root causes we found:
- **Old cookies** on blob were overwritten by yt-dlp with a stripped version (fixed by blob size validation)
- **Permission denied** on `temp_data/` (fixed via `.dockerignore` so cookies.txt from repo doesn't conflict)
- **Cookies expiry** — YouTube rotates session tokens periodically (automated via GitHub Actions workflow)

---

## Full System Flow

```
User (Browser)
    │
    │  HTTP POST /api/v1/jobs
    │  { url, target_text }
    ▼
quest1-ui  (NiceGUI frontend — port 8080)
    │  forwards to backend
    ▼
quest1-api  (FastAPI backend — port 8000)
    │
    │  1. Creates job_id
    │  2. Stores PROCESSING in Azure Table Storage
    │  3. Queues background task
    │  returns 202 immediately
    │
    ▼
PipelineOrchestrator.run()
    │
    ├─ Downloads cookies.txt from Azure Blob (if not cached)
    │
    ├─ probe_metadata(url)
    │   └─ yt-dlp + Deno/EJS solves YouTube JS challenge
    │   └─ returns fps, duration, stream_url, has_subtitles
    │
    ├─ TIER 0: Subtitle fuzzy match (similarity ≥ 85%)
    │   └─ ✓ DONE → extract frame → save result
    │
    ├─ TIER 1: Groq Whisper STT → word-level timestamp alignment
    │   └─ confidence ≥ 70% → TIER 3 dense OCR confirmation
    │
    ├─ TIER 2: Sparse OCR scan (Mistral) at 0.5 FPS
    │   └─ match ≥ 60% → TIER 3 dense OCR on ±2s window
    │
    ├─ TIER 4: NVIDIA NIM Llama 3.2 Vision arbiter
    │   └─ looks at candidate frames, picks best match
    │
    └─ Saves result to Azure Table + frame JPEG to Azure Blob

User polls GET /api/v1/jobs/{job_id} every 1.5s
    └─ Returns: timestamp, frame_number, confidence, tier, extracted_text, frame image URL
```

---

## Azure Concepts Used

### Resource Group (`quest1-rg`)
A logical container that groups all Azure resources for this project. Billing, access control, and deletion all operate at this level. Think of it as a project folder in Azure.

### Azure Container Registry — ACR (`quest1registry001`)
A private Docker image registry hosted on Azure. Instead of Docker Hub, we push our built images here. GitHub Actions builds the Docker image and pushes it here on every deploy.
- **Login server:** `quest1registry001.azurecr.io`
- **Repository:** `quest1` (holds all image tags)
- **SKU:** Basic (~$5/month from student credit)

### Azure Container Apps Environment (`quest1-env`)
The shared infrastructure layer that hosts all container apps. Handles networking, load balancing, scaling, and the Dapr sidecar. Multiple apps share one environment.
- **Location:** Central India
- **Default domain:** `ambitiousground-4a1b7fb4.centralindia.azurecontainerapps.io`

### Azure Container Apps (`quest1-api`, `quest1-ui`)
Serverless containers — Azure manages the underlying VMs. You just say "run this Docker image with these env vars". They scale to zero when idle (0 replicas = $0), scale up on traffic.
- **Revisions:** Every deploy or config change creates a new revision. Old ones stay until you delete them.
- **Scaling:** 0–2 replicas, 0.25 CPU, 0.5 Gi RAM per replica
- **Ingress:** Public HTTPS endpoint, Azure handles TLS automatically

### Azure Storage Account (`quest1storage001`)
One account, two services used:

| Service | What we use it for |
|---|---|
| **Blob Storage** | Stores `cookies.txt` + frame JPEG images per job |
| **Table Storage** | Stores job state (job_id → DetectionResult JSON) |

- **Container (Blob):** `quest1-artifacts` — like a bucket/folder
- **Table:** `quest1jobs` — NoSQL key-value store, PartitionKey + RowKey

### Azure NAT Gateway (not used — too expensive)
Would give a static outbound IP so YouTube doesn't block requests. ~$40/month. Not worth it on a $100 student credit.

---

## Every Azure CLI Command Explained

### Account & Setup
```powershell
az account show
```
Shows your current subscription name, ID, and login state. Used to confirm you're on "Azure for Students".

```powershell
az account show --query id -o tsv
```
`--query` filters the JSON response using JMESPath syntax. `id` extracts just the subscription ID. `-o tsv` outputs as plain text (no quotes) — useful for variables.

---

### Container Apps
```powershell
az containerapp list --resource-group quest1-rg --query "[].{Name:name, FQDN:...}" -o table
```
Lists all Container Apps in the resource group. `--query` renames JSON fields for readability. `-o table` formats as a readable table.

```powershell
az containerapp show --name quest1-api --resource-group quest1-rg --query "properties.template.containers[0].image" -o tsv
```
Gets the currently deployed Docker image SHA. We used this constantly to confirm the latest build was running. `properties.template.containers[0].image` drills into the nested JSON response.

```powershell
az containerapp update --name quest1-api --resource-group quest1-rg --set-env-vars KEY="value" --output none
```
Updates environment variables on a running app. Creates a new revision automatically. `--output none` suppresses the JSON response. We used this to:
- Remove the broken Webshare proxy (`--remove-env-vars YT_DLP_PROXY`)
- Force restarts with a dummy `COOKIES_REFRESH` variable

```powershell
az containerapp exec --name quest1-api --resource-group quest1-rg --command "deno --version"
```
Opens an interactive shell or runs a one-off command inside a running container replica. Used extensively to debug — checking Deno installation, inspecting `temp_data/`, verifying yt-dlp works from inside the container.

```powershell
az containerapp revision list --name quest1-api --resource-group quest1-rg -o table
```
Lists all revisions (deployments) for an app, with their active status and creation time. Used to confirm a new image was actually deployed vs cached.

---

### Container Registry
```powershell
az acr repository show-tags --name quest1registry001 --repository quest1 --orderby time_desc -o table
```
Lists all Docker image tags (Git SHAs) pushed to ACR, newest first. Used to confirm the CI/CD push worked.

---

### Blob Storage
```powershell
az storage blob upload --account-name quest1storage001 --container-name quest1-artifacts --name cookies.txt --file "C:\path\cookies.txt" --account-key "KEY" --overwrite
```
Uploads a local file to a blob container. `--overwrite` replaces the existing blob. `--account-key` authenticates directly with the storage key (no RBAC needed).

```powershell
az storage blob download --account-name quest1storage001 --container-name quest1-artifacts --name cookies.txt --file "C:\local\path.txt" --account-key "KEY"
```
Downloads a blob to a local file. Used to inspect what cookies were actually on the blob.

```powershell
az storage blob show --account-name quest1storage001 --container-name quest1-artifacts --name cookies.txt --account-key "KEY" --query "{size:properties.contentLength, lastModified:properties.lastModified}" -o json
```
Gets metadata about a blob without downloading it. We used this to check if cookies were fresh (by `lastModified`) and valid (by `size` — should be 3000+ bytes, not 1756 which was the stripped version).

```powershell
az storage blob list --account-name quest1storage001 --container-name quest1-artifacts --account-key "KEY" -o table
```
Lists all blobs in a container. Used to find the rogue `test_cookies.py` blob we accidentally uploaded.

```powershell
az storage blob delete --account-name quest1storage001 --container-name quest1-artifacts --name test_cookies.py --account-key "KEY"
```
Deletes a single blob. Used to clean up the test script we accidentally uploaded.

```powershell
az storage account keys list --account-name quest1storage001 --resource-group quest1-rg --query "[0].value" -o tsv
```
Gets the storage account access key. We needed this because the key in the summary was wrong — this fetched the correct one.

---

### Logs & Monitoring
```powershell
az containerapp logs show --name quest1-api --resource-group quest1-rg --tail 50
```
Streams the last 50 lines of container console logs. Unreliable on student plan (kept hitting `eventStreamEndpoint` error).

```powershell
az monitor log-analytics query -w WORKSPACE_ID --analytics-query "ContainerAppConsoleLogs_CL | where ..."
```
Queries logs stored in Log Analytics workspace using Kusto Query Language (KQL). More reliable than live streaming but has ~5 min ingestion delay. Used to filter for cookie-related logs.

---

## CI/CD Flow (GitHub Actions)

```
git push origin Rex
    │
    ├─ cookies.txt changed?
    │   └─ refresh-cookies.yaml runs:
    │       Upload cookies.txt → Azure Blob
    │       Restart quest1-api container
    │
    └─ Any other change?
        └─ ci-cd.yaml runs:
            1. pytest (82 tests) — blocks if any fail
            2. docker build + push to ACR
            3. az containerapp update (new image tag)
            4. deploy GitHub Pages (docs/)
```

**Secrets stored in GitHub (Settings → Secrets → Actions):**

| Secret | Used by | Purpose |
|---|---|---|
| `AZURE_CREDENTIALS` | ci-cd.yaml | Service principal JSON to `az login` |
| `ACR_USERNAME` | ci-cd.yaml | ACR admin username for docker login |
| `ACR_PASSWORD` | ci-cd.yaml | ACR admin password for docker login |
| `AZURE_STORAGE_KEY` | refresh-cookies.yaml | Storage account key for blob upload |

---

## Cookie Rotation — The Permanent Workflow

YouTube invalidates session cookies every few weeks. When you see "Sign in to confirm you're not a bot":

1. Export cookies from browser (logged into YouTube)
2. Save to `C:\AI_PROJECTS\Quest1\cookies.txt`
3. `git add cookies.txt && git commit -m "refresh cookies" && git push origin Rex`
4. GitHub Actions handles the rest automatically

**Why this can't be fully automated:** YouTube cookies are tied to a real browser session with a logged-in Google account. Any server-side automation trying to log in would hit CAPTCHA or 2FA. The human step (browser export) is unavoidable — but everything after that is now automated.
