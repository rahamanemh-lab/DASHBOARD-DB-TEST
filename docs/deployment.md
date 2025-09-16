### 4. `.gitlab-ci.yml` (à la racine)
```yaml
stages:
  - test
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/

test:
  stage: test
  image: python:3.9
  before_script:
    - pip install --upgrade pip
    - pip install -r requirements.txt
  script:
    - python -m py_compile app.py
    - python -c "import analyses; import utils"
    - echo "✅ Tests de syntaxe réussis"
  only:
    - merge_requests
    - main

deploy:
  stage: deploy
  image: ubuntu:22.04
  before_script:
    - apt-get update -qq && apt-get install -y -qq openssh-client rsync
    - eval $(ssh-agent -s)
    - echo "$SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add -
    - mkdir -p ~/.ssh
    - chmod 700 ~/.ssh
    - ssh-keyscan -H $SERVER_IP >> ~/.ssh/known_hosts
  script:
    - echo "🚀 Déploiement du dashboard"
    - rsync -avz --delete --exclude='.git' --exclude='streamlit_env' --exclude='__pycache__' ./ $SERVER_USER@$SERVER_IP:$APP_PATH/
    - ssh $SERVER_USER@$SERVER_IP "cd $APP_PATH && source venv/bin/activate && pip install -r requirements.txt"
    - ssh $SERVER_USER@$SERVER_IP "sudo systemctl restart dashboard-app"
    - echo "✅ Déploiement terminé"
  environment:
    name: production
    url: https://dashboard.votredomaine.com
  only:
    - main