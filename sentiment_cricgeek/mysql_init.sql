-- CricGeek MySQL initialization script
-- Update the password values below before running against a local MySQL server.

CREATE DATABASE IF NOT EXISTS cricgeek_dev
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'cricgeek'@'%' IDENTIFIED BY 'change-this-password';
ALTER USER 'cricgeek'@'%' IDENTIFIED BY 'change-this-password';

GRANT ALL PRIVILEGES ON cricgeek_dev.* TO 'cricgeek'@'%';
FLUSH PRIVILEGES;

USE cricgeek_dev;