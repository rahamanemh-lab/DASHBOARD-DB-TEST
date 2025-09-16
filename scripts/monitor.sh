#!/bin/bash
# Script de monitoring

LOG_FILE="/var/log/dashboard-monitor.log"
APP_URL="http://localhost:8501"
SERVICE_NAME="dashboard-app"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

if ! systemctl is-active --quiet $SERVICE_NAME; then
    log_message "ERREUR: Service $SERVICE_NAME inactif"
    sudo systemctl restart $SERVICE_NAME
    log_message "INFO: Redémarrage effectué"
    exit 1
fi

if ! curl -f -s $APP_URL > /dev/null; then
    log_message "ERREUR: Application ne répond pas"
    sudo systemctl restart $SERVICE_NAME
    log_message "INFO: Redémarrage effectué"
    exit 1
fi

log_message "INFO: Application opérationnelle"