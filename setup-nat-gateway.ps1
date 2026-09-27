# Azure NAT Gateway Setup Script for Quest1
# This script sets up static outbound IP for your Container App
# Cost: ~$40-50/month

param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== Quest1 Azure NAT Gateway Setup ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Create VNet and subnet (/23 = 512 IPs)" -ForegroundColor Yellow
Write-Host "  2. Create static public IP (permanent)" -ForegroundColor Yellow
Write-Host "  3. Create NAT Gateway (~`$40-50/month)" -ForegroundColor Yellow
Write-Host "  4. Recreate Container App environment (5 min downtime)" -ForegroundColor Yellow
Write-Host ""

if (-not $DryRun) {
    $confirm = Read-Host "Continue? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 0
    }
}

# Configuration
$RESOURCE_GROUP = "quest1-rg"
$LOCATION = "eastus"
$VNET_NAME = "quest1-vnet"
$SUBNET_NAME = "aca-subnet"
$NAT_GATEWAY_NAME = "quest1-nat-gateway"
$PUBLIC_IP_NAME = "quest1-nat-ip"
$ENV_NAME = "quest1-env"

Write-Host ""
Write-Host "Step 1/6: Getting subscription ID..." -ForegroundColor Green
$SUBSCRIPTION_ID = az account show --query id -o tsv
Write-Host "  Subscription: $SUBSCRIPTION_ID" -ForegroundColor Gray

Write-Host ""
Write-Host "Step 2/6: Creating VNet and subnet (10.0.0.0/23)..." -ForegroundColor Green
if ($DryRun) {
    Write-Host "  [DRY RUN] Would create VNet: $VNET_NAME" -ForegroundColor Gray
} else {
    az network vnet create `
        --name $VNET_NAME `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --address-prefix 10.0.0.0/16 `
        --subnet-name $SUBNET_NAME `
        --subnet-prefix 10.0.0.0/23 `
        --output none
    Write-Host "  ✓ VNet created: $VNET_NAME" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Step 3/6: Creating static public IP..." -ForegroundColor Green
if ($DryRun) {
    Write-Host "  [DRY RUN] Would create public IP: $PUBLIC_IP_NAME" -ForegroundColor Gray
} else {
    az network public-ip create `
        --name $PUBLIC_IP_NAME `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --sku Standard `
        --allocation-method Static `
        --output none
    
    $STATIC_IP = az network public-ip show `
        --name $PUBLIC_IP_NAME `
        --resource-group $RESOURCE_GROUP `
        --query ipAddress -o tsv
    
    Write-Host "  ✓ Static IP created: $STATIC_IP" -ForegroundColor Cyan
    Write-Host "  ✓ This is your permanent outbound IP!" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Step 4/6: Creating NAT Gateway..." -ForegroundColor Green
if ($DryRun) {
    Write-Host "  [DRY RUN] Would create NAT Gateway: $NAT_GATEWAY_NAME" -ForegroundColor Gray
} else {
    az network nat gateway create `
        --name $NAT_GATEWAY_NAME `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --public-ip-addresses $PUBLIC_IP_NAME `
        --idle-timeout 10 `
        --output none
    Write-Host "  ✓ NAT Gateway created: $NAT_GATEWAY_NAME" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Step 5/6: Attaching NAT Gateway to subnet..." -ForegroundColor Green
if ($DryRun) {
    Write-Host "  [DRY RUN] Would attach NAT Gateway to subnet" -ForegroundColor Gray
} else {
    az network vnet subnet update `
        --name $SUBNET_NAME `
        --vnet-name $VNET_NAME `
        --resource-group $RESOURCE_GROUP `
        --nat-gateway $NAT_GATEWAY_NAME `
        --output none
    Write-Host "  ✓ NAT Gateway attached to subnet" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Step 6/6: Recreating Container App environment with VNet..." -ForegroundColor Green
Write-Host "  WARNING: This will cause ~5 minutes downtime!" -ForegroundColor Yellow

if (-not $DryRun) {
    $confirm2 = Read-Host "  Proceed with environment recreation? (yes/no)"
    if ($confirm2 -ne "yes") {
        Write-Host ""
        Write-Host "Environment recreation skipped." -ForegroundColor Yellow
        Write-Host "NAT Gateway is ready but not connected to Container App." -ForegroundColor Yellow
        Write-Host "Run this manually when ready:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  az containerapp env delete --name $ENV_NAME -g $RESOURCE_GROUP --yes" -ForegroundColor Cyan
        Write-Host "  az containerapp env create --name $ENV_NAME -g $RESOURCE_GROUP --location $LOCATION --infrastructure-subnet-resource-id /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/$VNET_NAME/subnets/$SUBNET_NAME" -ForegroundColor Cyan
        exit 0
    }
}

if ($DryRun) {
    Write-Host "  [DRY RUN] Would delete and recreate Container App environment" -ForegroundColor Gray
} else {
    Write-Host "  Deleting old environment..." -ForegroundColor Gray
    az containerapp env delete `
        --name $ENV_NAME `
        --resource-group $RESOURCE_GROUP `
        --yes `
        --output none 2>$null
    
    Write-Host "  Creating new environment with VNet..." -ForegroundColor Gray
    $SUBNET_ID = "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/virtualNetworks/$VNET_NAME/subnets/$SUBNET_NAME"
    
    az containerapp env create `
        --name $ENV_NAME `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --infrastructure-subnet-resource-id $SUBNET_ID `
        --output none
    
    Write-Host "  ✓ Environment recreated with VNet" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""

if (-not $DryRun) {
    Write-Host "Your static outbound IP: $STATIC_IP" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Remove/empty GitHub Secret: YT_DLP_PROXY" -ForegroundColor Yellow
Write-Host "  2. Push to Rex branch to redeploy app" -ForegroundColor Yellow
Write-Host "  3. Test YouTube playlist scraping" -ForegroundColor Yellow
Write-Host ""
Write-Host "Cost: ~`$40-50/month for NAT Gateway + data processing" -ForegroundColor Gray
Write-Host ""
Write-Host "To remove NAT Gateway setup, see: docs/azure-nat-gateway-setup.md" -ForegroundColor Gray
