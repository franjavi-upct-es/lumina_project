#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[MLflow] Creating MLflow database and schema...${NC}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create MLflow database if it doesn't exist
    SELECT 'CREATE DATABASE lumina_mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lumina_mlflow') \gexec

    -- Grant privileges to the main user
    GRANT ALL PRIVILEGES ON DATABASE lumina_mlflow TO "$POSTGRES_USER";

    -- Connect to lumina_mlflow and ensure proper schema permissions
    \c lumina_mlflow

    -- Grant defaults privileges for future tables
    ALTER DEFAULT PRIVILEGES FOR USER "$POSTGRES_USER" GRANT ALL ON TABLES TO "$POSTGRES_USER";
    ALTER DEFAULT PRIVILEGES FOR USER "$POSTGRES_USER" GRANT ALL ON SEQUENCES TO "$POSTGRES_USER";

    -- Create extension for UUID if needed
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

EOSQL

echo -e "${GREEN}[MLflow] Database initialization completed successfully${NC}"

