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
    company_name = os.getenv("BRAND_NAME", "Your Company")
    logo_url = os.getenv("LOGO_URL", "https://example.com/logo.png")
    support_email = os.getenv("SUPPORT_EMAIL", from_email)

    html = f"""
    <html>
      <body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f4f7fb;color:#33475b;">
        <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
          <tr>
            <td align="center" style="padding:24px 0;">
              <table width="600" cellpadding="0" cellspacing="0" role="presentation" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 20px 60px rgba(22, 48, 86, 0.09);">
                <tr>
                  <td style="padding:32px 40px 24px;text-align:center;background:#0b5cff;">
                    <img src="{logo_url}" alt="{company_name}" width="120" style="display:block;margin:0 auto 16px;" />
                    <h1 style="margin:0;font-size:24px;line-height:1.2;color:#ffffff;">Reset Your Password</h1>
                  </td>
                </tr>
                <tr>
                  <td style="padding:32px 40px 24px;">
                    <p style="margin:0 0 18px;font-size:16px;line-height:1.7;color:#33475b;">Hi,</p>
                    <p style="margin:0 0 24px;font-size:16px;line-height:1.7;color:#33475b;">We received a request to reset the password for your {company_name} account associated with <strong>{email}</strong>. Click the button below to choose a new password.</p>
                    <p style="text-align:center;margin:0 0 30px;">
                      <a href="{link}" style="background:#0b5cff;color:#ffffff;text-decoration:none;padding:14px 24px;border-radius:10px;display:inline-block;font-size:16px;font-weight:600;">Reset your password</a>
                    </p>
                    <p style="margin:0 0 20px;font-size:14px;line-height:1.7;color:#667085;">If the button does not work, copy and paste the link below into your browser:</p>
                    <p style="word-break:break-all;font-size:14px;color:#0b5cff;"><a href="{link}" style="color:#0b5cff;text-decoration:none;">{link}</a></p>
                    <p style="margin:24px 0 0;font-size:14px;line-height:1.7;color:#667085;">If you did not request a password reset, you can safely ignore this email. The link will expire shortly for your security.</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:0 40px 32px;border-top:1px solid #e9edf5;">
                    <p style="margin:0;font-size:13px;line-height:1.6;color:#8492a6;">Need help? Contact us at <a href="mailto:{support_email}" style="color:#0b5cff;text-decoration:none;">{support_email}</a>.</p>
                  </td>
                </tr>
              </table>
              <p style="margin:20px 0 0;font-size:13px;color:#8492a6;">{company_name} • Secure account notifications</p>
            </td>
          </tr>
        </table>
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

    msg = EmailMessage()
    msg["Subject"] = f"{company_name} password reset request"
    msg["From"] = from_email
    msg["To"] = email
    msg.set_content(
        f"Hi,\n\nWe received a request to reset the password for your {company_name} account. "
        f"Use the link below to choose a new password:\n\n{link}\n\n"
        "If you did not request this, you can ignore this email.\n\n"
        f"Need help? Contact {support_email}.\n"
        f"{company_name}"
    )
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
