# Azure NAT Gateway Setup for Static Outbound IP

## Overview

This gives your Container App a **permanent static IP** for outgoing YouTube requests. Azure NAT Gateway is fully managed - no maintenance needed.

**Cost:** ~$32/month base + ~$0.045/GB data = **~$40-50/month total**

---

## Step 1: Create VNet and Subnet (3 minutes)

```bash
# Create VNet with /16 address space
az network vnet create \
  --name quest1-vnet \
  --resource-group quest1-rg \
  --location eastus \
  --address-prefix 10.0.0.0/16 \
  --subnet-name aca-subnet \
  --subnet-prefix 10.0.0.0/23
```

**Why /23?** Container Apps require minimum 512 IPs for infrastructure.

---

## Step 2: Create Static Public IP (1 minute)

```bash
# Create permanent static IP
az network public-ip create \
  --name quest1-nat-ip \
  --resource-group quest1-rg \
  --location eastus \
  --sku Standard \
  --allocation-method Static
```

**Save this IP!** This is what you'll use forever:

```bash
# Get your static IP address
az network public-ip show \
  --name quest1-nat-ip \
  --resource-group quest1-rg \
  --query ipAddress -o tsv
```

---

## Step 3: Create NAT Gateway (1 minute)

```bash
# Create NAT Gateway with your static IP
az network nat gateway create \
  --name quest1-nat-gateway \
  --resource-group quest1-rg \
  --location eastus \
  --public-ip-addresses quest1-nat-ip \
  --idle-timeout 10
```

---

## Step 4: Attach NAT Gateway to Subnet (30 seconds)

```bash
# Connect NAT Gateway to subnet
az network vnet subnet update \
  --name aca-subnet \
  --vnet-name quest1-vnet \
  --resource-group quest1-rg \
  --nat-gateway quest1-nat-gateway
```

---

## Step 5: Migrate Container App to VNet (5 minutes)

**IMPORTANT:** This will cause **~5 minutes downtime** while migrating.

### Option A: Recreate Environment (Recommended - Clean)

```bash
# 1. Get your subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# 2. Delete old environment (will delete the app too)
az containerapp env delete \
  --name quest1-env \
  --resource-group quest1-rg \
  --yes

# 3. Create new environment with VNet
az containerapp env create \
  --name quest1-env \
  --resource-group quest1-rg \
  --location eastus \
  --infrastructure-subnet-resource-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/quest1-rg/providers/Microsoft.Network/virtualNetworks/quest1-vnet/subnets/aca-subnet"

# 4. Re-deploy your app (CI/CD will handle this)
# Just push to Rex branch or manually trigger GitHub Actions
```

### Option B: In-Place Migration (Complex - Not Recommended)

Azure doesn't support adding VNet to existing Container App environments. You **must** recreate.

---

## Step 6: Update CI/CD Pipeline (Optional)

Your existing CI/CD will work fine. The app just uses the new environment.

But if you want to ensure VNet setup in CI/CD:

```yaml
# In .github/workflows/ci-cd.yaml
# Replace the containerapp env create step with:
- name: Create Container App Environment with VNet
  run: |
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    az containerapp env create \
      --name quest1-env \
      --resource-group quest1-rg \
      --location eastus \
      --infrastructure-subnet-resource-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/quest1-rg/providers/Microsoft.Network/virtualNetworks/quest1-vnet/subnets/aca-subnet" || true
```

---

## Step 7: Remove Proxy from Environment (1 minute)

Since you now have Azure's static IP, you don't need external proxies:

**GitHub Secrets:**
1. Go to: https://github.com/josondev/Quest1/settings/secrets/actions
2. Delete secret `YT_DLP_PROXY` or set it to empty

**Local .env:**
```bash
# Remove or comment out
# YT_DLP_PROXY=
```

---

## Verification

### 1. Check Your Static IP

```bash
# This is your permanent outbound IP
az network public-ip show \
  --name quest1-nat-ip \
  --resource-group quest1-rg \
  --query ipAddress -o tsv
```

### 2. Test Container App Uses NAT Gateway

```bash
# Deploy app, then check logs
az containerapp logs show \
  --name quest1-app \
  --resource-group quest1-rg \
  --follow
```

Run a test job - if no bot detection errors, it's working!

### 3. Verify NAT Gateway Traffic

```bash
# Check NAT Gateway metrics
az monitor metrics list \
  --resource /subscriptions/$SUBSCRIPTION_ID/resourceGroups/quest1-rg/providers/Microsoft.Network/natGateways/quest1-nat-gateway \
  --metric "ByteCount" \
  --start-time 2026-09-27T00:00:00Z \
  --end-time 2026-09-27T23:59:59Z
```

---

## Cost Breakdown

| Component | Monthly Cost |
|-----------|-------------|
| NAT Gateway Base | ~$32.40 |
| Data Processing (100 GB) | ~$4.50 |
| Static Public IP | ~$3.65 |
| **Total** | **~$40-45/month** |

**Free alternative:** Bright Data (2 GB free/month) - see `brightdata-setup.md`

---

## Troubleshooting

### "Subnet is too small"
- Container Apps need /23 minimum (512 IPs)
- Use `10.0.0.0/23` as shown above

### "Downtime during migration"
- Yes, ~5 minutes downtime when recreating environment
- Schedule during maintenance window
- Or accept the downtime for permanent fix

### "Still getting bot detection"
- Make sure `YT_DLP_PROXY` is removed/empty
- Check app is using new environment: `az containerapp show --name quest1-app -g quest1-rg --query "properties.environmentId"`

### "NAT Gateway not working"
- Verify subnet association: `az network vnet subnet show -n aca-subnet --vnet-name quest1-vnet -g quest1-rg --query natGateway`
- Check NAT Gateway has public IP: `az network nat gateway show -n quest1-nat-gateway -g quest1-rg --query publicIpAddresses`

---

## Rollback Plan

If something breaks:

```bash
# 1. Delete VNet-enabled environment
az containerapp env delete --name quest1-env -g quest1-rg --yes

# 2. Recreate simple environment (no VNet)
az containerapp env create \
  --name quest1-env \
  --resource-group quest1-rg \
  --location eastus

# 3. Re-deploy app via CI/CD
git commit --allow-empty -m "Trigger redeploy"
git push origin Rex

# 4. Re-add proxy secret if needed
# GitHub Secrets → YT_DLP_PROXY
```

---

## Summary

**You're paying for:**
- Permanent static IP (never changes)
- Unlimited connections (64,000 concurrent)
- Azure infrastructure (reliable, fast)
- No bot detection (legitimate Azure IP)

**You're avoiding:**
- External proxy services
- IP rotation issues
- Bot detection errors
- Free tier limitations

**Worth it if:**
- You scrape playlists frequently (>2 GB/month)
- You need 100% reliability
- You value convenience over cost
