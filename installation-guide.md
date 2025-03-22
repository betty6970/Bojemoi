# Metasploit Module Analyzer Service
## Installation and Deployment Guide

This guide provides detailed instructions for installing, configuring, and deploying the Metasploit Module Analyzer Service on Alpine Linux.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Service](#running-the-service)
5. [API Usage](#api-usage)
6. [Model Training](#model-training)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## Prerequisites

- Alpine Linux 3.14 or newer
- Python 3.6 or newer
- At least 2GB RAM (4GB recommended for model training)
- 500MB disk space for code and models

## Installation

### 1. Install Required System Packages

```bash
# Update package repository
apk update

# Install required packages
apk add python3 py3-pip py3-wheel py3-numpy py3-pandas py3-scikit-learn
apk add py3-flask
```

### 2. Create Directory Structure

```bash
# Create application directories
mkdir -p /opt/metasploit-analyzer/models
mkdir -p /opt/metasploit-modules
mkdir -p /var/log/metasploit-analyzer
```

### 3. Install Python Dependencies

Create a requirements.txt file with the following contents:

```
Flask==2.0.1
scikit-learn==1.0.2
pandas==1.3.5
numpy==1.21.5
waitress==2.0.0
```

Then install the dependencies:

```bash
pip3 install -r requirements.txt
```

### 4. Copy Application Files

Copy the application files to the installation directory:

```bash
# Copy application code
cp app.py /opt/metasploit-analyzer/
cp -r lib/ /opt/metasploit-analyzer/
```

### 5. Install Service Scripts

Copy the init script to the appropriate location:

```bash
cp metasploit-analyzer /etc/init.d/
chmod +x /etc/init.d/metasploit-analyzer

# Add to startup
rc-update add metasploit-analyzer default
```

## Configuration

### 1. Configure Application Settings

Edit the configuration file if needed:

```bash
# Edit configuration (if you created a config file)
nano /opt/metasploit-analyzer/config.json
```

### 2. Add Metasploit Modules

Copy Metasploit modules to the modules directory:

```bash
# If you have an existing Metasploit installation
cp -r /path/to/metasploit-framework/modules/* /opt/metasploit-modules/

# Or download from GitHub
cd /tmp
git clone --depth=1 https://github.com/rapid7/metasploit-framework.git
cp -r metasploit-framework/modules/* /opt/metasploit-modules/
rm -rf /tmp/metasploit-framework
```

## Running the Service

### 1. Start the Service

```bash
# Start the service
/etc/init.d/metasploit-analyzer start

# Check status
/etc/init.d/metasploit-analyzer status
```

### 2. Verify the Service is Running

```bash
# Check if the service is running
ps aux | grep metasploit-analyzer

# Check the log file
tail -f /var/log/metasploit-analyzer.log
```

### 3. Test API Access

```bash
# Test the health endpoint
curl http://localhost:5000/api/health
```

## API Usage

The Metasploit Module Analyzer Service provides a RESTful API with the following endpoints:

### API Documentation

- **GET /api**: List all available endpoints with descriptions and methods

### Health and Status

- **GET /api/health**: Check the health status of the service

### Module Management

- **GET /api/modules**: List all available modules
- **GET /api/module/{module_name}**: Get details for a specific module

### Analysis Features

- **POST /api/predict**: Predict the type of a module based on its content
- **GET /api/rules**: Get extracted rules and patterns
- **GET /api/stats**: Get statistics about loaded modules

### Question Answering

- **POST /api/ask**: Answer a question about Metasploit modules
- **GET /api/questions/log**: Generate and log sample questions

### Model Management

- **POST /api/train**: Start model training
- **POST /api/reload**: Reload modules from the directory

### Example Requests

#### Listing modules

```bash
curl -X GET http://localhost:5000/api/modules
```

#### Asking a question

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Sample Exploit?"}'
```

#### Starting model training

```bash
curl -X POST http://localhost:5000/api/train
```

## Model Training

The service includes a machine learning system that classifies Metasploit modules. To train the models:

### Initial Training

```bash
# Ensure modules are loaded
curl -X POST http://localhost:5000/api/reload

# Start training
curl -X POST http://localhost:5000/api/train
```

### Monitoring Training Progress

```bash
# Check training status
curl -X GET http://localhost:5000/api/health

# Check the log file for training details
tail -f /var/log/metasploit-analyzer.log
```

### Retraining Models

Models should be retrained when:
- New modules are added
- Existing modules are updated
- Model performance is unsatisfactory

## Troubleshooting

### Common Issues

#### Service Fails to Start

Check the logs for errors:

```bash
tail -f /var/log/metasploit-analyzer.log
```

Ensure Python and dependencies are installed correctly:

```bash
python3 -c "import sklearn, pandas, flask, waitress"
```

#### API Returns Errors

Check if models are loaded:

```bash
curl -X GET http://localhost:5000/api/health
```

Reload modules and retrain models if needed:

```bash
curl -X POST http://localhost:5000/api/reload
curl -X POST http://localhost:5000/api/train
```

#### Memory Issues

If the service runs out of memory during training:

```bash
# Increase swap space
dd if=/dev/zero of=/swapfile bs=1M count=2048
mkswap /swapfile
swapon /swapfile
```

Add to /etc/fstab for persistence:

```
/swapfile none swap defaults 0 0
```

## Maintenance

### Log Rotation

Setup log rotation by creating a file at `/etc/logrotate.d/metasploit-analyzer`:

```
/var/log/metasploit-analyzer.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
    create 0640 root root
}
```

### Backup Models

Backup trained models periodically:

```bash
tar -czf /backup/metasploit-models-$(date +%Y%m%d).tar.gz /opt/metasploit-analyzer/models/
```

### Updating the Service

To update the service:

1. Stop the service:
   ```bash
   /etc/init.d/metasploit-analyzer stop
   ```

2. Backup the current installation:
   ```bash
   cp -r /opt/metasploit-analyzer /opt/metasploit-analyzer.bak
   ```

3. Deploy the updated code:
   ```bash
   cp -r new_version/* /opt/metasploit-analyzer/
   ```

4. Start the service:
   ```bash
   /etc/init.d/metasploit-analyzer start
   ```

5. Verify the update:
   ```bash
   curl -X GET http://localhost:5000/api/health
   ```

## Performance Tuning

For better performance:

1. Increase the worker processes in waitress configuration:
   ```python
   serve(app, host=args.host, port=args.port, threads=16)
   ```

2. Use PyPy for faster execution:
   ```bash
   apk add pypy3 pypy3-pip
   pypy3 -m pip install -r requirements.txt
   ```

3. Optimize training parameters for your hardware:
   ```python
   # Reduce features for lower memory usage
   self.tfidf_vectorizer = TfidfVectorizer(max_features=3000)
   
   # Reduce estimators for faster training
   self.type_classifier = RandomForestClassifier(n_estimators=50)
   ```