#!/bin/bash
# Script de sauvegarde

BACKUP_DIR="/home/ubuntu/backups"
APP_DIR="/var/www/dashboard-souscriptions"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="dashboard-backup_$DATE.tar.gz"

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/$BACKUP_NAME -C $APP_DIR .
find $BACKUP_DIR -name "dashboard-backup_*.tar.gz" -type f -mtime +7 -delete

echo "✅ Sauvegarde créée : $BACKUP_DIR/$BACKUP_NAME"