CREATE DATABASE IF NOT EXISTS career_copilot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE career_copilot;

-- Create user only if not exists
-- NOTE: In production, use a more secure method for user creation and password management.
CREATE USER IF NOT EXISTS 'system'@'%' IDENTIFIED BY 'blasterBoy@2112';
GRANT ALL PRIVILEGES ON career_copilot.* TO 'system'@'%';
FLUSH PRIVILEGES;

-- Drop tables in reverse dependency order to avoid foreign key constraint errors
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS learning_resources;
DROP TABLE IF EXISTS user_skills;
DROP TABLE IF EXISTS resumes;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS job_skills;
DROP TABLE IF EXISTS skills;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS companies;


-- Companies table
CREATE TABLE companies (
  company_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) UNIQUE,
  industry VARCHAR(255)
);

-- Jobs table
CREATE TABLE jobs (
  job_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255),
  company_id INT,
  location VARCHAR(255),
  posted_date DATE NULL,
  description MEDIUMTEXT,
  source VARCHAR(100),
  FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

-- Skills table
CREATE TABLE skills (
  skill_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) UNIQUE,
  embedding JSON NULL,
  canonical_name VARCHAR(100)
);

-- Job-Skills junction table
CREATE TABLE job_skills (
  job_id BIGINT,
  skill_id INT,
  weight DECIMAL(6,4) DEFAULT 1.0,
  PRIMARY KEY (job_id, skill_id),
  FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
  FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

-- Users table
CREATE TABLE users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255),
  email VARCHAR(255) UNIQUE
);

-- Resumes table
CREATE TABLE resumes (
  resume_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  filename VARCHAR(255),
  parsed_text MEDIUMTEXT,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- User-Skills mapping
CREATE TABLE user_skills (
  user_id INT,
  skill_id INT,
  confidence DECIMAL(5,4) DEFAULT 1.0,
  PRIMARY KEY (user_id, skill_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
  FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

-- Learning resources
CREATE TABLE learning_resources (
  resource_id INT AUTO_INCREMENT PRIMARY KEY,
  skill_id INT,
  title VARCHAR(255),
  url TEXT,
  difficulty VARCHAR(50),
  estimated_hours INT,
  FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE
);

-- Recommendations table
CREATE TABLE recommendations (
  rec_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  summary TEXT,
  details JSON,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

