# Metasploit Module Analyzer Service
## Troubleshooting Guide

This guide provides solutions for common issues you may encounter when running the Metasploit Module Analyzer Service.

## Table of Contents

1. [Service Startup Issues](#service-startup-issues)
2. [Module Loading Problems](#module-loading-problems)
3. [API Errors](#api-errors)
4. [Model Training Issues](#model-training-issues)
5. [Performance Problems](#performance-problems)
6. [Question-Answering Issues](#question-answering-issues)
7. [Log Analysis](#log-analysis)
8. [Advanced Debugging](#advanced-debugging)

## Service Startup Issues

### Service Fails to Start

**Symptoms:**
- Service doesn't respond to requests
- `/etc/init.d/metasploit-analyzer status` shows not running
- Error messages in logs

**Possible Causes and Solutions:**

1. **Python Dependencies Missing**

   Check if all dependencies are installed:
   ```bash
   python3 -c "import flask, sklearn, pandas, numpy, waitress"
   ```
   
   If any import fails, reinstall dependencies:
   ```bash
   pip3 install -r /opt/metasploit-analyzer/requirements.txt
   ```

2. **Port Already in Use**

   Check if port 5000 is already in use:
   ```bash
   netstat -tuln | grep 5000
   ```
   
   If another service is using port 5000, modify the port in the service configuration or stop the conflicting service.

3. **Permission Issues**

   Check file permissions:
   ```bash
   ls -la /opt/metasploit-analyzer/
   ls -la /var/log/metasploit-analyzer.log
   ```
   
   Fix permissions if needed:
   ```bash
   chown -R root:root /opt/metasploit-analyzer/
   chmod 644 /var/log/metasploit-analyzer.log
   chmod +x /opt/metasploit-analyzer/app.py
   ```

4. **Invalid Configuration**

   Check for syntax errors in the application:
   ```bash
   cd /opt/metasploit-analyzer/
   python3 -m py_compile app.py
   ```

### Service Starts But Crashes

**Symptoms:**
- Service starts but stops responding after some time
- Error messages in logs indicating crashes

**Possible Causes and Solutions:**

1. **Memory Issues**

   Check memory usage:
   ```bash
   free -m
   ```
   
   If memory is low, add swap space:
   ```bash
   dd if=/dev/zero of=/swapfile bs=1M count=2048
   mkswap /swapfile
   swapon /swapfile
   ```

2. **Python Errors**

   Check logs for Python exceptions:
   ```bash
   tail -n 100 /var/log/metasploit-analyzer.log | grep "Exception"
   ```
   
   Address specific errors as they appear.

3. **File Descriptor Limits**

   Check current limits:
   ```bash
   ulimit -n
   ```
   
   If too low, increase in `/etc/security/limits.conf`:
   ```
   root soft nofile 4096
   root hard nofile 8192
   ```

## Module Loading Problems

### Modules Not Found

**Symptoms:**
- API returns empty module list
- "No modules loaded" errors

**Possible Causes and Solutions:**

1. **Incorrect Module Directory**

   Verify the module directory path:
   ```bash
   ls -la /opt/metasploit-modules/
   ```
   
   If empty or not found, copy modules:
   ```bash
   mkdir -p /opt/metasploit-modules/
   cp -r /path/to/metasploit-framework/modules/* /opt/metasploit-modules/
   ```

2. **Permission Issues**

   Check file permissions:
   ```bash
   ls -la /opt/metasploit-modules/
   ```
   
   Fix if needed:
   ```bash
   chmod -R 644 /opt/metasploit-modules/
   find /opt/metasploit-modules/ -type d -exec chmod 755 {} \;
   ```

3. **Invalid Module Format**

   Check for malformed module files:
   ```bash
   find /opt/metasploit-modules/ -name "*.rb" -exec grep -l "class MetasploitModule" {} \; | wc -l
   ```
   
   If count is much lower than expected, modules may not follow expected format.

### Module Parsing Errors

**Symptoms:**
- Errors in logs during module loading
- Missing module information

**Possible Causes and Solutions:**

1. **Encoding Issues**

   Check for encoding errors in logs:
   ```bash
   grep -i "encoding" /var/log/metasploit-analyzer.log
   ```
   
   Modify parser to handle different encodings:
   ```python
   # Use this in the parse_file method
   with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
       content = f.read()
   ```

2. **Unexpected Module Structure**

   If modules have different structure than expected, update regex patterns in `MetasploitModuleParser` class.

3. **File System Issues**

   Check for file system errors:
   ```bash
   dmesg | grep -i "error"
   ```
   
   Run file system check if needed:
   ```bash
   e2fsck -f /dev/sdaX  # Replace with appropriate partition
   ```

## API Errors

### API Returns 500 Errors

**Symptoms:**
- HTTP 500 response from API endpoints
- Error messages in logs

**Possible Causes and Solutions:**

1. **Application Errors**

   Check logs for exceptions:
   ```bash
   tail -n 100 /var/log/metasploit-analyzer.log
   ```
   
   Fix specific errors as they appear.

2. **Database Issues**

   If using a database for module storage, check connection:
   ```bash
   python3 -c "import sqlite3; conn = sqlite3.connect('/opt/metasploit-analyzer/data/modules.db'); print('Connected')"
   ```

3. **Disk Space Issues**

   Check available disk space:
   ```bash
   df -h
   ```
   
   Free space if needed:
   ```bash
   apt-get clean  # On Debian-based systems
   apk cache clean # On Alpine
   ```

### API Returns 404 Errors

**Symptoms:**
- HTTP 404 response when accessing endpoints
- "Not Found" messages

**Possible Causes and Solutions:**

1. **Incorrect URL**

   Verify you're using the correct URL format.

2. **Blueprint Not Registered**

   Check if API blueprint is registered correctly in the application.

3. **Endpoint Implementation Missing**

   Verify all endpoints are properly implemented in the `MetasploitAnalyzerAPI` class.

## Model Training Issues

### Training Fails to Start

**Symptoms:**
- Training doesn't start after API request
- No training feedback in logs

**Possible Causes and Solutions:**

1. **No Modules Loaded**

   Check if modules are loaded:
   ```bash
   curl -X GET http://localhost:5000/api/modules
   ```
   
   If no modules, reload them:
   ```bash
   curl -X POST http://localhost:5000/api/reload
   ```

2. **Already Training**

   Check if training is already in progress:
   ```bash
   curl -X GET http://localhost:5000/api/health
   ```
   
   Wait for current training to complete.

3. **Threading Issues**

   Check for thread-related errors in logs.

### Training Crashes

**Symptoms:**
- Training starts but fails to complete
- Error messages in logs

**Possible Causes and Solutions:**

1. **Out of Memory**

   Check memory usage during training:
   ```bash
   watch -n 1 free -m
   ```
   
   Increase available memory or reduce model complexity.

2. **Insufficient Training Data**

   If not enough valid modules, error may occur. Check logs for:
   ```
   "Not enough valid modules for training"
   ```
   
   Add more module examples or reduce minimum sample requirement.

3. **Invalid Data**

   Check for data processing errors in logs.

## Performance Problems

### Slow API Response

**Symptoms:**
- API endpoints take long time to respond
- Timeouts in client applications

**Possible Causes and Solutions:**

1. **High CPU Load**

   Check CPU usage:
   ```bash
   top
   ```
   
   If consistently high, consider:
   - Increasing server resources
   - Optimizing code (e.g., caching results)
   - Using pypy for better performance

2. **Inefficient Queries**

   Add logging to measure endpoint response times:
   ```python
   import time
   
   def some_endpoint():
       start_time = time.time()
       # Endpoint logic
       duration = time.time() - start_time
       logger.info(f"Endpoint executed in {duration:.2f} seconds")
   ```

3. **I/O Bottlenecks**

   Check disk I/O performance:
   ```bash
   iostat -x 1
   ```
   
   If disk is bottleneck, consider moving to SSD or optimizing I/O operations.

### High Memory Usage

**Symptoms:**
- System runs out of memory
- Service crashes with OOM errors

**Possible Causes and Solutions:**

1. **Large Models in Memory**

   Reduce model complexity:
   ```python
   # Reduce TF-IDF features
   self.tfidf_vectorizer = TfidfVectorizer(max_features=2000)
   
   # Use smaller classifier
   self.type_classifier = RandomForestClassifier(n_estimators=50)
   ```

2. **Memory Leaks**

   Check for increasing memory usage over time. If found, implement periodic restarts:
   ```bash
   # Add to crontab
   0 3 * * * /etc/init.d/metasploit-analyzer restart
   ```

3. **Too Many Modules Loaded**

   If too many modules cause memory issues, implement lazy loading or pagination.

## Question-Answering Issues

### Poor Answer Quality

**Symptoms:**
- Irrelevant or incorrect answers to questions
- Low confidence scores

**Possible Causes and Solutions:**

1. **Insufficient Training Data**

   Generate more QA pairs:
   ```bash
   curl -X GET http://localhost:5000/api/questions/log
   ```
   
   Improve QA generation templates.

2. **Question Matching Logic**

   Enhance the question matching algorithm in `answer_question` method.

3. **Model Not Trained Well**

   Retrain models with better parameters:
   ```bash
   curl -X POST http://localhost:5000/api/train
   ```

### Questions Not Recognized

**Symptoms:**
- "I don't have enough information" responses
- Module names not recognized in questions

**Possible Causes and Solutions:**

1. **Question Format**

   Ensure questions follow expected patterns.

2. **Module Name Matching**

   Improve module name recognition with fuzzy matching:
   ```python
   # Add to requirements.txt
   # fuzzywuzzy
   
   from fuzzywuzzy import process
   
   def find_module_name(question, module_names):
       matches = process.extract(question, module_names, limit=1)
       if matches and matches[0][1] > 80:  # 80% similarity threshold
           return matches[0][0]
       return None
   ```

3. **Missing Context**

   Add more context to the Q&A system.

## Log Analysis

### Understanding Log Messages

**Common Log Patterns:**

1. **Info Messages**
   ```
   INFO - Starting Metasploit Module Analyzer Service
   INFO - Loaded 1500 Metasploit modules
   INFO - Training completed with accuracy: 0.89
   ```

2. **Warning Messages**
   ```
   WARNING - No models found, will train new ones
   WARNING - Some modules missing required fields
   ```

3. **Error Messages**
   ```
   ERROR - Failed to parse module: /path/to/module.rb
   ERROR - Training failed: Not enough valid modules
   ERROR - Out of memory during model training
   ```

### Log Monitoring

Set up log monitoring to catch issues early:

```bash
# Real-time log monitoring
tail -f /var/log/metasploit-analyzer.log | grep -i error

# Send email on critical errors
grep -i "ERROR" /var/log/metasploit-analyzer.log | mail -s "Metasploit Analyzer Critical Errors" admin@example.com
```

## Advanced Debugging

### Running in Debug Mode

Launch the application in debug mode for detailed output:

```bash
cd /opt/metasploit-analyzer
python3 app.py --debug --modules-dir /opt/metasploit-modules
```

### Interactive Debugging

Use Python's debugger for interactive troubleshooting:

```bash
cd /opt/metasploit-analyzer
python3 -m pdb app.py
```

### Profiling Performance

Identify performance bottlenecks:

```bash
# Install profiler
pip3 install cProfile

# Run with profiling
python3 -m cProfile -o profile.stats app.py

# Analyze results (install first: pip3 install snakeviz)
snakeviz profile.stats
```

### Database Inspection

If using SQLite for caching:

```bash
sqlite3 /opt/metasploit-analyzer/data/cache.db
.tables
SELECT * FROM modules LIMIT 10;
```

### Network Debugging

Check network connectivity and API accessibility:

```bash
# Test local connection
curl -v http://localhost:5000/api/health

# Check open ports
netstat -tulpn | grep python

# Monitor network traffic
tcpdump -i any port 5000 -n
```

### Memory Profiling

Track memory usage:

```bash
# Install memory profiler
pip3 install memory_profiler

# Add to your code
@profile
def memory_intensive_function():
    # Function code

# Run with profiling
python3 -m memory_profiler app.py
```

## Additional Resources

- [Flask Debugging Documentation](https://flask.palletsprojects.com/en/2.0.x/debugging/)
- [Scikit-learn Troubleshooting](https://scikit-learn.org/stable/developers/contributing.html#debugging)
- [TF-IDF Vectorization Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Waitress Production Server Documentation](https://docs.pylonsproject.org/projects/waitress/en/latest/)