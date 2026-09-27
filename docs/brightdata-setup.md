# Bright Data Free Proxy Setup Guide

## Why Bright Data?

Bright Data offers the best free tier for datacenter proxies:
- ✅ **15 datacenter proxy IPs** (vs Webshare's 10)
- ✅ **2 GB monthly traffic** (vs Webshare's 1 GB)
- ✅ **Better IP reputation** - matches their paid infrastructure quality
- ✅ **No expiration** on the free tier

## Setup Steps

### 1. Create Bright Data Account
1. Go to [https://brightdata.com/](https://brightdata.com/)
2. Sign up for a free account
3. Verify your email

### 2. Create a Datacenter Proxy Zone
1. Log in to [Bright Data Control Panel](https://brightdata.com/cp/zones)
2. Click **"Add Zone"** or **"Create Zone"**
3. Choose **"Datacenter"** as the proxy type
4. Configure:
   - **Zone name**: `youtube-scraper` (or any name)
   - **Country**: US (recommended for YouTube)
   - Leave other settings as default
5. Click **"Add Zone"** or **"Save"**

### 3. Get Your Proxy Credentials
1. In the Zones dashboard, find your newly created zone
2. Copy the **proxy endpoint** - it looks like:
   ```
   brd.superproxy.io:22225
   ```
3. Copy your **Username** (format: `brd-customer-<ID>-zone-<zone-name>`)
4. Copy your **Password**

### 4. Update Environment Variables

**Local Development (.env file):**
```bash
YT_DLP_PROXY=http://your-username:your-password@brd.superproxy.io:22225
```

**Azure Container Apps (GitHub Secrets):**
1. Go to your repository: **Settings → Secrets and variables → Actions**
2. Update the secret `YT_DLP_PROXY`:
   ```
   http://your-username:your-password@brd.superproxy.io:22225
   ```

### 5. Test Locally (Optional)
```bash
# Test the proxy works
python -c "import requests; r=requests.get('https://www.youtube.com', proxies={'http':'http://USER:PASS@brd.superproxy.io:22225','https':'http://USER:PASS@brd.superproxy.io:22225'}); print(r.status_code)"
```

### 6. Deploy to Azure
After updating the GitHub secret, push any change to trigger the CI/CD pipeline:
```bash
git add .
git commit -m "Switch to Bright Data proxies"
git push origin Rex
```

## Monitoring Usage

1. Go to [Bright Data Dashboard](https://brightdata.com/cp/zones)
2. Check your zone's **traffic usage**
3. Free tier provides **2 GB/month**
4. Usage resets monthly

## Troubleshooting

### "Authentication Failed"
- Double-check username and password
- Ensure no extra spaces in the credentials
- Verify the zone is active in Bright Data dashboard

### "Connection Timeout"
- Check if firewall is blocking port 22225
- Try alternative port: 22225 (default) or 33333

### "Quota Exceeded"
- You've used your 2 GB monthly quota
- Wait for next month's reset, or upgrade to paid tier

## Alternative Ports

Bright Data supports multiple ports:
- `22225` - Default HTTP/HTTPS
- `33333` - Alternative port
- `22125` - SOCKS5 (not needed for yt-dlp)

## Proxy Format Reference

```
http://USERNAME:PASSWORD@HOST:PORT
```

Example:
```
http://brd-customer-abc123-zone-youtube:xyzpass123@brd.superproxy.io:22225
```

## Cost Comparison

| Provider | Free IPs | Free Traffic | IP Type |
|----------|----------|--------------|---------|
| Bright Data | 15 | 2 GB/month | Datacenter |
| Webshare | 10 | 1 GB/month | Datacenter |
| Oxylabs | 5 | N/A | Datacenter (static) |

## Need More?

If 2 GB/month isn't enough:
- **Pay-as-you-go**: ~$0.80/GB for datacenter proxies
- **Residential proxies**: Better for YouTube, but more expensive
- **Contact support**: Sometimes they offer extended free trials
