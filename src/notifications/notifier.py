import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

def is_critical_event(event: dict) -> bool:
    critical_types = [
        "ENTRY_RESTRICTED_ZONE",
        "ENTRY_AFTER_HOURS",
        "BBS_FALL_DETECTED",
        "BBS_UNSAFE_PROXIMITY"
    ]
    event_type = event.get("type", "")
    return any(c_type in event_type for c_type in critical_types)

def send_realtime_alert(event: dict) -> None:
    try:
        tw_sid = os.getenv("TWILIO_ACCOUNT_SID")
        tw_tok = os.getenv("TWILIO_AUTH_TOKEN")
        tw_from = os.getenv("TWILIO_FROM_NUMBER")
        tw_to = os.getenv("TWILIO_TO_NUMBER")
        
        time_str = event.get("time") or event.get("timestamp") or "unknown time"

        message = f"⚠ GuardianVision Alert ({event.get('camera_id')}): {event.get('type')} at {time_str}, Risk {event.get('risk')}"
        
        if tw_sid and tw_tok and tw_from and tw_to:
            import requests
            auth = (tw_sid, tw_tok)
            data = {
                "To": tw_to,
                "From": tw_from,
                "Body": message
            }
            resp = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{tw_sid}/Messages.json", auth=auth, data=data)
            resp.raise_for_status()
        else:
            print(f"[ALERT-SIM] {message}")
    except Exception as e:
        print(f"[ALERT-ERROR] Failed to send realtime alert: {e}")

def send_email_report(subject: str, body: str) -> None:
    try:
        email_host = os.getenv("EMAIL_HOST")
        email_port = os.getenv("EMAIL_PORT")
        email_user = os.getenv("EMAIL_USER")
        email_pass = os.getenv("EMAIL_PASS")
        email_to = os.getenv("EMAIL_TO")
        
        if not all([email_host, email_port, email_user, email_pass, email_to]):
            print(f"[EMAIL-SIM] Would send: {subject}")
            return

        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = email_user
        msg['To'] = email_to
        
        with smtplib.SMTP(email_host, int(email_port)) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
    except Exception as e:
        print(f"[EMAIL-ERROR] Failed to send email report: {e}")
