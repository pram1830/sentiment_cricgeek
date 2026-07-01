# MySQL Quickstart

Use this guide if you want CricGeek to run against MySQL instead of the default SQLite development database.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure MySQL

Create a database and user in MySQL:

```sql
CREATE DATABASE cricgeek_dev CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'cricgeek'@'%' IDENTIFIED BY 'change-this-password';
GRANT ALL PRIVILEGES ON cricgeek_dev.* TO 'cricgeek'@'%';
FLUSH PRIVILEGES;
```

## 3. Set environment variables

Copy `.env.example` to `.env` and set these values:

```env
ENVIRONMENT=development
DATABASE_URL=mysql+pymysql://cricgeek:change-this-password@localhost:3306/cricgeek_dev?charset=utf8mb4
DB_PASSWORD=change-this-password
MYSQL_ROOT_PASSWORD=change-this-root-password
JWT_SECRET_KEY=replace-with-a-long-random-secret
ENCRYPTION_KEY=replace-with-fernet-key
```

If you use Docker Compose, keep the hostname as `mysql` instead of `localhost`:

```env
DATABASE_URL=mysql+pymysql://cricgeek:change-this-password@mysql:3306/cricgeek_dev?charset=utf8mb4
```

## 4. Initialize the schema

```bash
python database.py init
```

Optional seed data:

```bash
python database.py seed
```

## 5. Run with Docker

```bash
docker compose -f docker-compose.mysql.yml up --build
```

That compose file starts MySQL, creates the schema, and launches the API and Streamlit app.

On Windows, the helper batch file now prefers the local `mysql` client if it is installed. If not, it falls back to Docker Compose automatically:

```bat
setup_mysql.bat
```

For a local MySQL server, the script runs the SQL initializer for you. You can still run it manually if needed:

```bash
mysql -u root -p < mysql_init.sql
```

## 6. Run locally

Start MySQL first, then launch the services:

```bash
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
streamlit run app.py --server.port 8501
```

## Notes

- The app already supports MySQL through SQLAlchemy and `PyMySQL`.
- UUIDs are stored as portable 36-character strings, so the schema works on MySQL and SQLite.
- If you want production deployment, use a managed MySQL instance and keep secrets out of source control.