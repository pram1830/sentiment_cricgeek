# MySQL Setup Guide

This project can run against MySQL by using a MySQL `DATABASE_URL` and the `PyMySQL` driver.

The repository already includes MySQL support in `database.py` and a MySQL-specific Docker Compose file.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

The MySQL driver dependency is `PyMySQL`.

If you are using `docker-compose.mysql.yml`, make sure `.env` contains both `DB_PASSWORD` and `MYSQL_ROOT_PASSWORD`.

## 2. Create the database and user

Run these commands in MySQL as an admin user:

```sql
CREATE DATABASE cricgeek_dev CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'cricgeek'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON cricgeek_dev.* TO 'cricgeek'@'%';
FLUSH PRIVILEGES;
```

## 3. Set the environment variable

Use this connection string in `.env`:

```env
DATABASE_URL=mysql+pymysql://cricgeek:password@localhost:3306/cricgeek_dev?charset=utf8mb4
```

Recommended `.env` values for the Docker Compose MySQL stack:

```env
ENVIRONMENT=development
DATABASE_URL=mysql+pymysql://cricgeek:change-this-password@mysql:3306/cricgeek_dev?charset=utf8mb4
DB_PASSWORD=change-this-password
MYSQL_ROOT_PASSWORD=change-this-root-password
JWT_SECRET_KEY=replace-with-a-long-random-secret
ENCRYPTION_KEY=replace-with-fernet-key
```

## 4. Initialize the schema

```bash
python database.py init
```

Optional sample data:

```bash
python database.py seed
```

## 5. Start the app

```bash
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
streamlit run app.py --server.port 8501
```

## Docker option

Use `docker-compose.mysql.yml` to bring up MySQL, the API, and the Streamlit app together.

```bash
docker compose -f docker-compose.mysql.yml up --build
```

If you already have a local MySQL server, only the `DATABASE_URL` and schema init steps are required.

## Troubleshooting

- If `docker compose` stops immediately, verify that `MYSQL_ROOT_PASSWORD` and `DB_PASSWORD` are set in `.env`.
- If the API cannot connect to the database, confirm that `DATABASE_URL` uses the `mysql+pymysql://` scheme.
- If you are running locally without Docker, start MySQL first, then run `python database.py init`.