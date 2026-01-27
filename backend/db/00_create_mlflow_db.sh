#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE lumina_mlflow;
    GRANT ALL PRIVILEGES ON DATABASE lumina_mlflow TO $POSTGRES_USER;
EOSQL
