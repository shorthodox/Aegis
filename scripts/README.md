Send branded password reset emails
===============================

This script generates a Firebase password reset link and sends a branded HTML email via SendGrid.

Setup
------
1. Create a Firebase service account JSON with the required permissions and save it somewhere safe (not in the repo).
2. Verify your sender domain or `FROM_EMAIL` in SendGrid.
3. Copy `.env.example` to `.env` and fill values for `SENDGRID_API_KEY`, `FIREBASE_SERVICE_ACCOUNT`, `FROM_EMAIL`, and `CONTINUE_URL`.
4. Install dependencies (preferably in a virtualenv):

```bash
pip install -r requirements.txt
```

Usage
------
Send a reset email to one address:

```bash
python scripts/send_reset_email.py --email user@example.com
```

Notes
------
- Do not commit `.env` or service account files. Add them to `.gitignore` if needed.
- `CONTINUE_URL` is optional; if set, users will be directed to that page where you can handle the `oobCode` to perform the password reset in-app.
- Customize the HTML in `scripts/send_reset_email.py` to match your brand (logo URL, colors, footer).

SMTP (Neo Work Mail) usage
-------------------------
You can send via your own SMTP server (e.g., Neo Work Mail) by setting the SMTP environment variables in `.env` instead of `SENDGRID_API_KEY`.

Required SMTP vars:
- `SMTP_HOST`, `SMTP_PORT`, and credentials `SMTP_USER`/`SMTP_PASS` if authentication is required.

If `SENDGRID_API_KEY` is set the script will prefer SendGrid; otherwise it will fall back to SMTP.
