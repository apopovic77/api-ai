# GCP Budget & Pub/Sub Alert Setup

Backup-Alerting für Gemini API Kosten via Google Cloud Platform.

**Wichtig:** Das Token-basierte Tracking in `cost_tracker.py` ist die primäre Methode (real-time).
GCP Budget Alerts sind ein Backup mit ~24h Verzögerung.

## Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL ALERT SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRIMARY (Real-time)              BACKUP (24h delay)            │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Token Tracking  │              │ GCP Budget API  │           │
│  │ in cost_tracker │              │ + Pub/Sub       │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │                                │                    │
│           ▼                                ▼                    │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ Telegram Alert  │              │ Cloud Function  │           │
│  │ (immediate)     │              │ → Telegram      │           │
│  └─────────────────┘              └─────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1. GCP Budget erstellen

### 1.1 Via Google Cloud Console

1. Öffne: https://console.cloud.google.com/billing
2. Wähle dein Billing Account
3. Gehe zu "Budgets & alerts"
4. Klicke "CREATE BUDGET"

### 1.2 Budget Konfiguration

```
Name: gemini-api-monthly-30eur
Amount: 30 EUR (oder $32 USD)
Scope:
  - Projects: [dein-projekt]
  - Services: Generative Language API

Thresholds:
  - 50% of budget (Email + Pub/Sub)
  - 80% of budget (Email + Pub/Sub)
  - 95% of budget (Email + Pub/Sub)
  - 100% of budget (Email + Pub/Sub)
```

### 1.3 Via gcloud CLI

```bash
# Budget erstellen
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="gemini-api-monthly-30eur" \
  --budget-amount=32USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.8 \
  --threshold-rule=percent=0.95 \
  --threshold-rule=percent=1.0 \
  --filter-services="services/aiplatform.googleapis.com"
```

## 2. Pub/Sub Topic erstellen

```bash
# Topic erstellen
gcloud pubsub topics create gemini-budget-alerts

# Subscription für Cloud Function
gcloud pubsub subscriptions create gemini-budget-alerts-sub \
  --topic=gemini-budget-alerts
```

## 3. Budget mit Pub/Sub verbinden

1. Gehe zu Budget Details
2. Under "Manage notifications"
3. Wähle "Connect a Pub/Sub topic to this budget"
4. Wähle Topic: `gemini-budget-alerts`

## 4. Cloud Function für Telegram

### 4.1 Function erstellen

Erstelle `main.py`:

```python
import base64
import json
import os
import functions_framework
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


@functions_framework.cloud_event
def budget_alert(cloud_event):
    """Handle budget alert from Pub/Sub."""

    # Decode Pub/Sub message
    data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    budget_data = json.loads(data)

    # Extract info
    budget_name = budget_data.get("budgetDisplayName", "Unknown")
    cost_amount = budget_data.get("costAmount", 0)
    budget_amount = budget_data.get("budgetAmount", 0)
    threshold = budget_data.get("alertThresholdExceeded", 0) * 100

    # Determine emoji and status
    if threshold >= 100:
        emoji = "🚨"
        status = "BUDGET EXCEEDED"
    elif threshold >= 95:
        emoji = "⚠️"
        status = "CRITICAL"
    elif threshold >= 80:
        emoji = "🔶"
        status = "WARNING"
    else:
        emoji = "📊"
        status = "INFO"

    # Format message
    message = f"""
{emoji} <b>GCP Budget Alert - {status}</b>

<b>Budget:</b> {budget_name}
<b>Threshold:</b> {threshold:.0f}% exceeded
<b>Current Cost:</b> ${cost_amount:.2f}
<b>Budget Amount:</b> ${budget_amount:.2f}

<b>Note:</b> This is from GCP Billing (24h delayed).
Check real-time status: https://api-ai.arkturian.com/ai/gemini/cost-status
"""

    # Send to Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message.strip(),
            "parse_mode": "HTML",
        }
        requests.post(url, json=payload, timeout=10)

    return "OK"
```

### 4.2 Requirements

`requirements.txt`:
```
functions-framework==3.*
requests==2.*
```

### 4.3 Deploy

```bash
# Deploy Cloud Function
gcloud functions deploy gemini-budget-telegram \
  --gen2 \
  --runtime=python311 \
  --region=europe-west1 \
  --source=. \
  --entry-point=budget_alert \
  --trigger-topic=gemini-budget-alerts \
  --set-env-vars="TELEGRAM_BOT_TOKEN=YOUR_TOKEN,TELEGRAM_CHAT_ID=YOUR_CHAT_ID"
```

## 5. Alternative: Webhook statt Cloud Function

Wenn du keine Cloud Function nutzen willst, kannst du auch einen Webhook Endpoint in api-ai erstellen:

### 5.1 Endpoint in api-ai

Füge zu `text_ai_routes.py` hinzu:

```python
@router.post("/gemini/gcp-budget-webhook")
async def gcp_budget_webhook(request: Request):
    """
    Webhook endpoint for GCP Budget Pub/Sub push notifications.

    Configure in GCP:
    1. Create Push Subscription auf gemini-budget-alerts Topic
    2. Endpoint URL: https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook
    """
    import base64
    import json
    from ..services.cost_tracker import cost_tracker

    body = await request.json()

    # Decode Pub/Sub message
    message_data = body.get("message", {}).get("data", "")
    if message_data:
        decoded = base64.b64decode(message_data).decode()
        budget_data = json.loads(decoded)

        budget_name = budget_data.get("budgetDisplayName", "Unknown")
        cost_amount = budget_data.get("costAmount", 0)
        budget_amount = budget_data.get("budgetAmount", 0)
        threshold = budget_data.get("alertThresholdExceeded", 0) * 100

        # Send Telegram alert
        if threshold >= 100:
            emoji = "🚨"
            status = "GCP: BUDGET EXCEEDED"
        elif threshold >= 80:
            emoji = "⚠️"
            status = "GCP: WARNING"
        else:
            emoji = "📊"
            status = "GCP: INFO"

        message = f"""
{emoji} <b>{status}</b>

<b>Budget:</b> {budget_name}
<b>Threshold:</b> {threshold:.0f}%
<b>GCP Cost:</b> ${cost_amount:.2f} / ${budget_amount:.2f}

<i>Note: GCP data has ~24h delay</i>
"""
        cost_tracker._send_telegram_message(message.strip())

    return {"status": "ok"}
```

### 5.2 Push Subscription erstellen

```bash
gcloud pubsub subscriptions create gemini-budget-push \
  --topic=gemini-budget-alerts \
  --push-endpoint=https://api-ai.arkturian.com/ai/gemini/gcp-budget-webhook \
  --push-auth-service-account=YOUR_SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com
```

## 6. Testen

### 6.1 Token Tracking testen (Primary)

```bash
# Einen Gemini Request machen
curl -X POST "https://api-ai.arkturian.com/ai/gemini" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, what is 2+2?"}'

# Status prüfen
curl "https://api-ai.arkturian.com/ai/gemini/cost-status"
```

### 6.2 Pub/Sub testen (Backup)

```bash
# Test-Nachricht publishen
gcloud pubsub topics publish gemini-budget-alerts \
  --message='{"budgetDisplayName":"Test","costAmount":15,"budgetAmount":30,"alertThresholdExceeded":0.5}'
```

## 7. Kosten

| Service | Kosten |
|---------|--------|
| Cloud Billing Budget | Kostenlos |
| Pub/Sub | ~$0.04/100k messages |
| Cloud Function | ~$0.40/million invocations |
| **Token Tracking** | **Kostenlos** (in api-ai) |

## 8. Zusammenfassung

| Feature | Token Tracking | GCP Budget |
|---------|----------------|------------|
| Delay | Real-time | ~24 Stunden |
| Genauigkeit | Sehr hoch | Hoch |
| Kosten | Kostenlos | ~$1/Monat |
| Blocking | Ja | Nein |
| Setup | Bereits aktiv | Manual |

**Empfehlung:** Token Tracking ist bereits aktiv und funktioniert. GCP Budget als zusätzliche Absicherung einrichten wenn gewünscht.
