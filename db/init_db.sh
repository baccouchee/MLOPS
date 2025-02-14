#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE predictions;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "predictions" <<-EOSQL
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        input_data JSONB,
        prediction VARCHAR(50)
    );
EOSQL