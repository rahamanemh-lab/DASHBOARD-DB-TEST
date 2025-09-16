#!/bin/bash
# Mise à jour vers Python 3.11

echo "🐍 Mise à jour Python 3.10 → 3.11"
echo "==================================="

# 1. Modifier .gitlab-ci.yml pour Python 3.11
echo "📝 Modification .gitlab-ci.yml..."
sed -i '' 's/python:3\.10/python:3.11/g' .gitlab-ci.yml

# 2. Modifier Dockerfile pour Python 3.11
if [ -f "Dockerfile" ]; then
    echo "🐳 Modification Dockerfile..."
    sed -i '' 's/python:3\.10/python:3.11/g' Dockerfile
fi

# 3. Vérifier les changements
echo ""
echo "📋 Versions Python mises à jour:"
grep -n "python:" .gitlab-ci.yml
if [ -f "Dockerfile" ]; then
    grep -n "python:" Dockerfile
fi

# 4. Vérifier la syntaxe YAML
if python3 -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))" 2>/dev/null; then
    echo "✅ Syntaxe YAML valide"
else
    echo "❌ Erreur syntaxe YAML"
    exit 1
fi

# 5. Commit et push
echo ""
echo "💾 Commit des changements..."
git add .gitlab-ci.yml
if [ -f "Dockerfile" ]; then
    git add Dockerfile
fi

git commit -m "fix: mise à jour Python 3.10 → 3.11 pour numpy==2.3.0"

echo "📤 Push vers GitLab..."
if git push origin main; then
    echo ""
    echo "🎉 SUCCÈS!"
    echo "✅ Python 3.11 configuré"
    echo "🚀 Pipeline va se relancer avec Python 3.11"
    echo "📦 numpy==2.3.0 va maintenant s'installer"
    echo ""
    echo "🌐 Surveillez: https://gitlab.com/datanaly-group/dashboard-souscriptions/-/pipelines"
else
    echo "❌ Échec du push"
    git status
fi
