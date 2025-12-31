# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the security team directly (or open a private security advisory on GitHub)
3. Include as much detail as possible:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 7 days
- **Resolution Timeline**: We aim to resolve critical issues within 30 days
- **Disclosure**: We will coordinate disclosure timing with you

### Security Best Practices for Contributors

When contributing to this project:

1. **Never commit secrets**
   - No API keys, passwords, or tokens in code
   - Use environment variables (see `env.template`)
   - Check commits before pushing

2. **Sanitize inputs**
   - Validate all user inputs
   - Use parameterized queries for database access
   - Sanitize data before logging

3. **Dependencies**
   - Keep dependencies updated
   - Review security advisories for dependencies
   - Pin versions in production

4. **Database Access**
   - Use least-privilege database accounts
   - Never expose database credentials
   - Use connection pooling appropriately

5. **API Security**
   - Implement proper authentication
   - Use HTTPS in production
   - Rate limit endpoints

### Security Configuration

The application supports several security-related configurations:

```bash
# Example secure configuration (env.template)
APP_ENV=production

# Use strong, unique passwords
BACKEND_DB_PASS=<strong-password>
DW_DB_PASS=<strong-password>

# Never commit real credentials
```

## Scope

This security policy applies to:
- The main application code
- Official Docker images
- Documentation containing security guidance

## Out of Scope

- Third-party dependencies (report to their maintainers)
- Self-hosted instances with custom modifications
- Social engineering attacks

---

Thank you for helping keep SOTI Stella Sentinel secure!

