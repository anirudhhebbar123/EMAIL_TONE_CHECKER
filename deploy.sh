#!/bin/bash

echo "🚀 Email Tone Checker - Render Deployment Script"
echo "================================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Add all files
echo ""
echo "📝 Adding files to Git..."
git add .

# Show status
echo ""
echo "📋 Files to be committed:"
git status --short

# Commit
echo ""
read -p "Enter commit message (default: 'Deploy to Render'): " commit_msg
commit_msg=${commit_msg:-"Deploy to Render"}
git commit -m "$commit_msg"
echo "✅ Files committed"

# Check if remote exists
if git remote | grep -q 'origin'; then
    echo ""
    echo "✅ Remote 'origin' already exists"
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Push to existing remote? (y/n): " push_existing
    if [ "$push_existing" = "y" ]; then
        git push origin main
        echo "✅ Pushed to GitHub"
    fi
else
    echo ""
    echo "⚠️  No remote repository found"
    echo ""
    read -p "Enter your GitHub repository URL (e.g., https://github.com/username/email-tone-checker.git): " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        git branch -M main
        git push -u origin main
        echo "✅ Pushed to GitHub"
    else
        echo "❌ No repository URL provided. Please push manually."
    fi
fi

echo ""
echo "================================================"
echo "✅ Deployment Preparation Complete!"
echo ""
echo "Next Steps:"
echo "1. Go to https://render.com"
echo "2. Sign in with GitHub"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Select your repository: email-tone-checker"
echo "5. Configure:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: gunicorn app:app --timeout 120 --workers 2"
echo "   - Plan: Free"
echo "6. Click 'Create Web Service'"
echo ""
echo "Your app will be live in 5-10 minutes!"
echo "================================================"