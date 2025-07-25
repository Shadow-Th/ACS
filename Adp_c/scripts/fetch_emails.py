import imaplib, email
import pandas as pd
from datetime import datetime, timedelta
import base64

def extract_email_features(username, password, num_emails=50):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)
    mail.select("inbox")
    
    result, data = mail.search(None, "ALL")
    email_ids = data[0].split()[-num_emails:]
    rows = []

    for eid in email_ids:
        result, msg_data = mail.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = msg["subject"] or ""
        subject_len = len(subject)
        sender = msg["from"]
        sender_importance = 1 if "boss" in sender.lower() or "important" in sender.lower() else 0.5
        attachment_size = 0
        priority = 1 if "important" in subject.lower() or "urgent" in subject.lower() else 0

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == "multipart" or part.get("Content-Disposition") is None:
                    continue
                payload = part.get_payload(decode=True)
                if payload:
                    attachment_size += len(payload)

        time_since_access = 1  # Simulated (can integrate with metadata or IMAP flags)

        rows.append([subject_len, sender_importance, attachment_size / 1000, time_since_access, priority, -1])  # -1 for unknown label

    df = pd.DataFrame(rows, columns=["subject_len", "sender_importance", "attachment_size", "time_since_access", "priority", "label"])
    df.to_csv("data/emails.csv", index=False)
    print("✅ Email features extracted and saved to data/emails.csv")

