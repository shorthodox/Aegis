#!/usr/bin/env python3
"""Send a branded password reset email using Firebase Admin and SMTP.

Requires environment variables or a .env file:
- FIREBASE_SERVICE_ACCOUNT: path to Firebase service account JSON (optional if app default creds available)
- FROM_EMAIL: sender email
- CONTINUE_URL: (optional) your hosted reset page (e.g. https://example.com/reset-password)
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS for your SMTP provider

Usage:
  python scripts/send_reset_email.py --email user@example.com
"""
import argparse
import os
import logging
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

import firebase_admin
from firebase_admin import auth, credentials

logging.basicConfig(level=logging.INFO)


def init_firebase():
    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if sa_path and os.path.exists(sa_path):
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
        logging.info("Initialized Firebase with service account %s", sa_path)
    else:
        try:
            firebase_admin.initialize_app()
            logging.info("Initialized Firebase with default credentials")
        except Exception:
            logging.warning("Firebase not initialized — set FIREBASE_SERVICE_ACCOUNT to a valid JSON path.")


def generate_reset_link(email: str) -> str:
    continue_url = os.getenv("CONTINUE_URL")
    action_settings = None
    if continue_url:
        try:
            action_settings = auth.ActionCodeSettings(url=continue_url, handle_code_in_app=True)
        except Exception:
            action_settings = None
    if action_settings:
        return auth.generate_password_reset_link(email, action_settings)
    return auth.generate_password_reset_link(email)


def send_email(email: str, link: str):
    from_email = os.getenv("FROM_EMAIL", "noreply@example.com")

    html = f"""
    <html>
      <body style="font-family:Arial,Helvetica,sans-serif;color:#111;margin:0;padding:0;">
        <div style="max-width:600px;margin:0 auto;padding:24px;">
          <img src="https://example.com/logo.png" alt="Logo" style="height:48px;margin-bottom:12px;">
          <h2 style="color:#0b5cff;margin:8px 0;">Reset your password</h2>
          <p style="color:#333;">We received a request to reset the password for this account.</p>
          <p style="text-align:center;margin:28px 0;">
            <a href="{link}" style="background:#0b5cff;color:#fff;padding:12px 20px;border-radius:6px;text-decoration:none;display:inline-block;font-weight:600;">Reset password</a>
          </p>
          <p style="font-size:13px;color:#666;">If you didn't request this, you can safely ignore this email.</p>
          <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
          <p style="font-size:12px;color:#999;">Sent by Your Company</p>
        </div>
      </body>
    </html>
    """

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    use_ssl = os.getenv("SMTP_SSL", "false").lower() in ("1", "true", "yes")

    if not smtp_host:
        raise RuntimeError("SMTP_HOST not set. Cannot send email")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")
    use_ssl = os.getenv("SMTP_SSL", "false").lower() in ("1", "true", "yes")

    if not smtp_host:
        raise RuntimeError("SMTP_HOST not set. Cannot send email")

    msg = EmailMessage()
    msg["Subject"] = "Reset your password"
    msg["From"] = from_email
    msg["To"] = email
    msg.set_content("Reset your password: " + link)
    msg.add_alternative(html, subtype="html")

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if use_tls:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)

    logging.info("Email sent via SMTP host %s", smtp_host)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Recipient email address")
    args = parser.parse_args()

    init_firebase()
    link = generate_reset_link(args.email)
    logging.info("Generated reset link for %s", args.email)
    send_email(args.email, link)


if __name__ == "__main__":
    main()
