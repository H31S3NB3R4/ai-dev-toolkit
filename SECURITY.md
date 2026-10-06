# Security Policy

The AI Dev Toolkit team takes security and user privacy seriously.

---

## Supported Versions

We provide security updates and patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability in AI Dev Toolkit (such as prompt injection bypasses, API key leaks, or unsafe parsing bugs), please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Send an email to `security@ai-dev-toolkit.org` with:
   - A description of the vulnerability and potential impact.
   - Minimal reproduction steps or proof-of-concept code.
   - Any proposed mitigations or fixes.
3. We will acknowledge receipt of your report within **48 hours** and provide regular status updates during our investigation and patch deployment.

---

## API Key and Secret Safety

- AI Dev Toolkit is designed to never print, log, or serialize API keys or secret credentials.
- Error messages and stack traces are automatically sanitized to prevent accidental credential leakage.
- Never commit `.env` files or secret keys to version control. Keep `.env` listed in `.gitignore`.
