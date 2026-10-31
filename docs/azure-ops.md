# Quest1 — Azure Operations Reference

Everything you need to monitor, debug, and maintain the Azure deployment.

---

## Infrastructure Map

```
quest1-rg  (Resource Group — Central India)
│
├── quest1-env                    Azure Container Apps Environment
│   ├── quest1-api                Backend  (FastAPI + pipeline)
│   └── quest1-ui                 Frontend (NiceGUI)
│
├── quest1registry001             Azure Container Registry (ACR)
│   └── quest1                    Docker image repository
│
└── quest1storage001              Azure Storage Account
    ├── Blob: quest1-artifacts    Frames + cookies.txt
    └── Table: quest1jobs         Job state store
```

**Live URLs**
| Service | URL |
|---|---|
| API | https://quest1-api.ambitiousground-4a1b7fb4.centralindia.azurecontainerapps.io |
| UI  | https://quest1-ui.ambitiousground-4a1b7fb4.centralindia.azurecontainerapps.io |
| Health | https://quest1-api.ambitiousground-4a1b7fb4.centralindia.azurecontainerapps.io/health |

---

## 1. Container Apps

### Check running image (confirm latest deploy)
```powershell
az containerapp show --name quest1-api --resource-group quest1-rg `
  --query "properties.template.containers[0].image" -o tsv
```

### List all revisions
```powershell
az containerapp revision list --name quest1-api --resource-group quest1-rg `
  --query "[].{Name:name, Active:properties.active, Created:properties.createdTime, State:properties.provisioningState}" `
  -o table
```

### Check current resource allocation
```powershell
az containerapp show --name quest1-api --resource-group quest1-rg `
  --query "{CPU:properties.template.containers[0].resources.cpu, Memory:properties.template.containers[0].resources.memory, MinReplicas:properties.template.scale.minReplicas, MaxReplicas:properties.template.scale.maxReplicas}" `
  -o json
```
> Current: 0.25 CPU · 0.5 Gi RAM · 0–2 replicas

### List all environment variables (names only, not values)
```powershell
az containerapp show --name quest1-api --resource-group quest1-rg `
  --query "properties.template.containers[0].env[].name" -o json
```

### Set / update an environment variable
```powershell
az containerapp update --name quest1-api --resource-group quest1-rg `
  --set-env-vars MY_VAR="value" --output none
```

### Remove an environment variable
```powershell
az containerapp update --name quest1-api --resource-group quest1-rg `
  --remove-env-vars MY_VAR --output none
```

### Force a fresh restart (new revision rolling out)
> Azure revision restart is unreliable on the student plan. Use a dummy env var bump instead:
```powershell
az containerapp update --name quest1-api --resource-group quest1-rg `
  --set-env-vars RESTART_TS="$(Get-Date -Format 'yyyyMMddHHmm')" --output none
```

### Shell into a running container
```powershell
az containerapp exec --name quest1-api --resource-group quest1-rg `
  --command "bash"
```

### Run a one-off command inside the container
```powershell
az containerapp exec --name quest1-api --resource-group quest1-rg `
  --command "deno --version"
```

---

## 2. Container Registry (ACR)

### List all images + tags
```powershell
az acr repository show-tags --name quest1registry001 --repository quest1 `
  --orderby time_desc -o table
```

### Check the latest image SHA
```powershell
az acr repository show --name quest1registry001 --image quest1:latest -o json
```

### Purge old images (keep last 5)
```powershell
az acr run --registry quest1registry001 --cmd `
  "acr purge --filter 'quest1:.*' --ago 30d --keep 5 --untagged" /dev/null
```

---

## 3. Blob Storage

### List blobs in quest1-artifacts
```powershell
az storage blob list `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --account-key "YOUR_ACCOUNT_KEY" `
  --query "[].{Name:name, Size:properties.contentLength, Modified:properties.lastModified}" `
  -o table
```

### Check cookies.txt is fresh (run this after re-exporting cookies)
```powershell
az storage blob show `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --name cookies.txt `
  --account-key "YOUR_ACCOUNT_KEY" `
  --query "{size:properties.contentLength, lastModified:properties.lastModified}" `
  -o json
```
> Expected: size ~3000+ bytes, lastModified = today

### Upload fresh cookies.txt to blob
```powershell
az storage blob upload `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --name cookies.txt `
  --file "C:\path\to\your\cookies.txt" `
  --account-key "YOUR_ACCOUNT_KEY" `
  --overwrite
```
> After uploading, force a container restart (see section 1) so the new cookies are picked up.

### Download cookies.txt from blob (to inspect locally)
```powershell
az storage blob download `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --name cookies.txt `
  --file "C:\AI_PROJECTS\Quest1\cookies_from_blob.txt" `
  --account-key "YOUR_ACCOUNT_KEY"
```

### Delete a blob
```powershell
az storage blob delete `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --name BLOB_NAME `
  --account-key "YOUR_ACCOUNT_KEY"
```

### Get the storage account key (if you need it)
```powershell
az storage account keys list `
  --account-name quest1storage001 `
  --resource-group quest1-rg `
  --query "[0].value" -o tsv
```

---

## 4. Table Storage (Job State)

### List recent jobs
```powershell
az storage entity query `
  --account-name quest1storage001 `
  --table-name quest1jobs `
  --account-key "YOUR_ACCOUNT_KEY" `
  --select "RowKey,status,tier_executed,timestamp_seconds" `
  -o table
```

### Get a specific job by ID
```powershell
az storage entity show `
  --account-name quest1storage001 `
  --table-name quest1jobs `
  --partition-key "job" `
  --row-key "job_YOURID" `
  --account-key "YOUR_ACCOUNT_KEY" `
  -o json
```

---

## 5. Logs

### Stream live logs (system logs)
```powershell
az containerapp logs show --name quest1-api --resource-group quest1-rg `
  --type system --tail 50
```

### Query logs via Log Analytics (last 15 min)
```powershell
$WS = "b34c0765-742b-4594-bb1d-469a69d2d7fd"
az monitor log-analytics query -w $WS --analytics-query `
  "ContainerAppConsoleLogs_CL
   | where ContainerAppName_s == 'quest1-api'
   | where TimeGenerated > ago(15m)
   | project TimeGenerated, Log_s
   | order by TimeGenerated desc
   | take 50" -o table
```

### Filter logs for errors only
```powershell
az monitor log-analytics query -w $WS --analytics-query `
  "ContainerAppConsoleLogs_CL
   | where ContainerAppName_s == 'quest1-api'
   | where TimeGenerated > ago(1h)
   | where Log_s contains 'ERROR' or Log_s contains 'WARNING'
   | project TimeGenerated, Log_s
   | order by TimeGenerated desc" -o table
```

### Filter logs for cookie activity
```powershell
az monitor log-analytics query -w $WS --analytics-query `
  "ContainerAppConsoleLogs_CL
   | where ContainerAppName_s == 'quest1-api'
   | where TimeGenerated > ago(1h)
   | where Log_s contains 'cookie' or Log_s contains 'Cookie'
   | project TimeGenerated, Log_s
   | order by TimeGenerated desc" -o table
```

---

## 6. CI/CD (GitHub Actions)

### Trigger a manual deploy (without a code change)
```powershell
cd C:\AI_PROJECTS\Quest1
git commit --allow-empty -m "Trigger redeploy"
git push origin Rex
```

### Check what's deployed vs what's in git
```powershell
# Git SHA of latest push
git rev-parse --short HEAD

# SHA running in Azure
az containerapp show --name quest1-api --resource-group quest1-rg `
  --query "properties.template.containers[0].image" -o tsv
# The SHA after the colon should match git HEAD
```

### GitHub Secrets needed by CI/CD
| Secret | Purpose |
|---|---|
| `AZURE_CREDENTIALS` | Service principal JSON for `az login` |
| `ACR_USERNAME` | ACR admin username |
| `ACR_PASSWORD` | ACR admin password |

---

## 7. YouTube Cookies Maintenance

YouTube rotates session cookies periodically (every few weeks). When you see:

```
"The provided YouTube account cookies are no longer valid"
```

**Steps to refresh:**

1. Install browser extension: [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
2. Go to `https://www.youtube.com` while **logged in**
3. Click the extension → Export → save as `cookies.txt`
4. Upload to blob:
```powershell
az storage blob upload `
  --account-name quest1storage001 `
  --container-name quest1-artifacts `
  --name cookies.txt `
  --file "C:\path\to\cookies.txt" `
  --account-key "YOUR_ACCOUNT_KEY" `
  --overwrite
```
5. Force container restart:
```powershell
az containerapp update --name quest1-api --resource-group quest1-rg `
  --set-env-vars COOKIES_REFRESH="$(Get-Date -Format 'yyyyMMddHHmm')" --output none
```

**How the app validates cookies:**
- On first request per container boot, downloads `cookies.txt` from blob
- Compares local cache size against blob size — if they differ, re-downloads
- This prevents yt-dlp from overwriting the cache with a stripped version

---

## 8. Quick Health Check (run all at once)

```powershell
# 1. API health endpoint
curl https://quest1-api.ambitiousground-4a1b7fb4.centralindia.azurecontainerapps.io/health

# 2. Container Apps status
az containerapp list --resource-group quest1-rg `
  --query "[].{Name:name, LatestRevision:properties.latestRevisionName}" -o table

# 3. Latest deployed image SHA vs git HEAD
Write-Output "Git HEAD: $(git -C C:\AI_PROJECTS\Quest1 rev-parse HEAD)"
az containerapp show --name quest1-api --resource-group quest1-rg `
  --query "properties.template.containers[0].image" -o tsv

# 4. Cookies blob freshness
az storage blob show `
  --account-name quest1storage001 --container-name quest1-artifacts `
  --name cookies.txt `
  --account-key "YOUR_ACCOUNT_KEY" `
  --query "{size:properties.contentLength, lastModified:properties.lastModified}" -o json
```

---

## 9. Student Subscription Limits

| Resource | Current | Free / Included |
|---|---|---|
| Container Apps | 0.25 CPU / 0.5 Gi | Free tier: 180,000 vCPU-s/month |
| ACR Basic | ~1 image | Costs ~$5/month from credit |
| Blob Storage | LRS | First 5 GB free |
| Table Storage | quest1jobs | First 10 GB free |
| **Total Azure credit** | **$100** | Renews annually while enrolled |

> NAT Gateway (~$40/month) would burn your $100 in ~2 months. Not recommended unless you need a static IP badly.
