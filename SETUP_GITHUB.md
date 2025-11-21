# GitHub Repository Setup Instructions

## Prerequisites

1. **Install Git** (if not already installed)
   - Download from: https://git-scm.com/download/win
   - Or use: `winget install Git.Git`

2. **GitHub Account**
   - Make sure you're logged into GitHub
   - Repository URL: https://github.com/Ghassenboussalem/OddoBhfComplianceCheck.git

## Step-by-Step Setup

### 1. Initialize Git Repository

Open a terminal in your project directory and run:

```bash
git init
```

### 2. Add Remote Repository

```bash
git remote add origin https://github.com/Ghassenboussalem/OddoBhfComplianceCheck.git
```

### 3. Stage All Files

```bash
git add .
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: AI-Enhanced Compliance Checker v1.0"
```

### 5. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

## Authentication

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your password)

### Creating a Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Copy and save the token
4. Use it as your password when pushing

## Verify

After pushing, visit:
https://github.com/Ghassenboussalem/OddoBhfComplianceCheck

You should see all your files!

## Files Included

✅ Core Python files (check.py, ai_engine.py, etc.)
✅ Configuration files (*.json)
✅ Documentation (*.md)
✅ Rules files (*.json)
✅ Example files (exemple.json, exemple_violations.json)
✅ .gitignore (protects sensitive data)
✅ .env.example (template for API keys)
✅ requirements.txt (dependencies)
✅ LICENSE

## Files Excluded (for security)

❌ .env (contains your API keys)
❌ prospectus.docx (sensitive document)
❌ *.xlsx files (sensitive data)
❌ registration.csv (sensitive data)

## Troubleshooting

### "Git not found"
Install Git from https://git-scm.com/download/win

### "Permission denied"
Use a Personal Access Token instead of password

### "Repository not found"
Make sure the repository exists on GitHub first

### "Failed to push"
Check if repository already has content. If so, pull first:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```
