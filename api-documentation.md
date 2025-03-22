# Metasploit Module Analyzer API Documentation

This document provides comprehensive documentation for the Metasploit Module Analyzer Service REST API, including all available endpoints, request formats, and response structures.

## API Overview

The Metasploit Module Analyzer Service exposes a REST API that provides functionality for analyzing Metasploit modules, including:

- Module listing and details
- Machine learning-based module type prediction
- Module statistics and pattern extraction
- Question answering about modules
- Model training and management

## Base URL

All API endpoints are relative to the base URL of the service:

```
http://<hostname>:5000
```

## Authentication

Currently, the API does not implement authentication. It is designed for local access only. If public access is required, appropriate security measures should be implemented.

## Endpoints

### API Documentation

#### List All Endpoints

Returns a list of all available API endpoints with their descriptions and supported methods.

- **URL**: `/api`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "service": "Metasploit Module Analyzer API",
      "version": "1.0.0",
      "total_endpoints": 11,
      "endpoints": [
        {
          "url": "/api",
          "endpoint": "list_endpoints",
          "description": "Liste tous les endpoints disponibles avec leur description et méthodes",
          "methods": ["GET", "HEAD", "OPTIONS"]
        },
        // More endpoints...
      ]
    }
    ```

### Health and Status

#### Health Check

Provides the current health status of the service.

- **URL**: `/api/health`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "status": "ok",
      "timestamp": "2023-03-21T14:30:45.123456",
      "models_loaded": true,
      "modules_loaded": 1500,
      "training_in_progress": false
    }
    ```

### Module Management

#### List Modules

Returns a list of all Metasploit modules loaded in the service.

- **URL**: `/api/modules`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "total": 1500,
      "modules": [
        {
          "name": "Sample Exploit",
          "file_name": "sample_exploit.rb",
          "module_type": "Exploit",
          "authors": ["skape"],
          "disclosure_date": "2020-12-30"
        },
        // More modules...
      ]
    }
    ```

#### Get Module Details

Returns detailed information about a specific Metasploit module.

- **URL**: `/api/module/<module_name>`
- **Method**: `GET`
- **URL Parameters**:
  - `module_name`: Name of the module to retrieve
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "name": "Sample Exploit",
      "file_name": "sample_exploit.rb",
      "file_path": "/opt/metasploit-modules/exploits/sample_exploit.rb",
      "module_type": "Exploit",
      "description": "This exploit module illustrates how a vulnerability could be exploited in an TCP server that has a parsing bug.",
      "license": "MSF_LICENSE",
      "authors": ["skape"],
      "references": [
        {"type": "OSVDB", "value": "12345"},
        {"type": "EDB", "value": "12345"},
        {"type": "URL", "value": "http://www.example.com"},
        {"type": "CVE", "value": "1978-1234"}
      ],
      "payload_info": "Space => 1000, BadChars => \"\\x00\"",
      "targets": ["Windows XP/Vista/7/8"],
      "disclosure_date": "2020-12-30",
      "includes": ["Exploit::Remote::Tcp"]
    }
    ```
- **Error Response**:
  - **Code**: 404
  - **Content**:
    ```json
    {
      "error": "Module 'Sample Exploit' not found"
    }
    ```

### Analysis Features

#### Predict Module Type

Predicts the type of a Metasploit module based on its content using machine learning.

- **URL**: `/api/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "content": "class MetasploitModule < Msf::Exploit::Remote\n  Rank = NormalRanking\n  include Exploit::Remote::Tcp\n  def initialize(info = {})\n    super(\n      update_info(\n        info,\n        'Name' => 'Sample Exploit',\n        'Description' => %q{\n          This exploit module illustrates how a vulnerability could be exploited\n          in an TCP server that has a parsing bug.\n        }..."
  }
  ```
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "predicted_type": "Exploit",
      "probabilities": {
        "Exploit": 0.92,
        "Auxiliary": 0.05,
        "Post": 0.02,
        "Payload": 0.01
      }
    }
    ```
- **Error Response**:
  - **Code**: 400
  - **Content**:
    ```json
    {
      "error": "Module content required"
    }
    ```

#### Get Rules

Returns patterns and rules extracted from the analyzed modules.

- **URL**: `/api/rules`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "module_type_distribution": {
        "Exploit": 750,
        "Auxiliary": 500,
        "Post": 200,
        "Payload": 50
      },
      "top_authors": {
        "hdm": 120,
        "egyp7": 85,
        "jduck": 75,
        "sinn3r": 65,
        "todb": 60
      },
      "reference_type_distribution": {
        "CVE": 985,
        "URL": 750,
        "EDB": 420,
        "OSVDB": 300
      },
      "top_target_platforms": {
        "Windows": 450,
        "Linux": 200,
        "Multi": 150,
        "Android": 75,
        "macOS": 50
      },
      "disclosure_date_stats": {
        "oldest": "2004-03-15",
        "newest": "2023-02-28",
        "by_year": {
          "2004": 15,
          "2005": 28,
          "2006": 45,
          "2007": 62,
          // More years...
        }
      },
      "top_includes": {
        "Exploit::Remote::Tcp": 320,
        "Exploit::Remote::Http": 280,
        "Msf::Exploit::Remote": 180,
        // More includes...
      }
    }
    ```

#### Get Statistics

Returns statistics about the loaded Metasploit modules.

- **URL**: `/api/stats`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "total_modules": 1500,
      "module_types": {
        "Exploit": 750,
        "Auxiliary": 500,
        "Post": 200,
        "Payload": 50
      },
      "total_authors": 850,
      "total_references": 4500,
      "modules_with_targets": 1200,
      "modules_with_payload_info": 800,
      "average_authors_per_module": 2.3,
      "average_references_per_module": 3.0
    }
    ```

### Question Answering

#### Ask Question

Answers a question about Metasploit modules using the trained models.

- **URL**: `/api/ask`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "question": "What is Sample Exploit?"
  }
  ```
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "answer": "This exploit module illustrates how a vulnerability could be exploited in an TCP server that has a parsing bug.",
      "confidence": 0.92,
      "module": "Sample Exploit"
    }
    ```
- **Error Response**:
  - **Code**: 400
  - **Content**:
    ```json
    {
      "error": "Question text required"
    }
    ```

#### Log Questions

Generates sample questions for a random module and logs them for analysis.

- **URL**: `/api/questions/log`
- **Method**: `GET`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "module": "Sample Exploit",
      "questions_generated": 7,
      "sample_questions": [
        "What is Sample Exploit?",
        "Who authored Sample Exploit?",
        "What type of module is Sample Exploit?",
        "What are the targets for Sample Exploit?",
        "When was Sample Exploit disclosed?"
      ]
    }
    ```

### Model Management

#### Start Training

Initiates the training of machine learning models based on the loaded modules.

- **URL**: `/api/train`
- **Method**: `POST`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "status": "training_started",
      "message": "Model training has been started in the background"
    }
    ```
- **Error Response**:
  - **Code**: 200 (still success, but with error information)
  - **Content**:
    ```json
    {
      "error": "Training already in progress",
      "status": "in_progress"
    }
    ```

#### Reload Modules

Reloads Metasploit modules from the configured directory.

- **URL**: `/api/reload`
- **Method**: `POST`
- **URL Parameters**: None
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "status": "success",
      "modules_loaded": 1500
    }
    ```
- **Error Response**:
  - **Code**: 500
  - **Content**:
    ```json
    {
      "error": "Error loading modules: Directory not found"
    }
    ```

## Request and Response Formats

### General Response Format

All API responses are returned in JSON format and generally follow this structure:

```json
{
  "status": "success|error",
  // Other response-specific fields
}
```

Error responses typically include an "error" field with a description of what went wrong:

```json
{
  "error": "Error description"
}
```

### Pagination

For endpoints that may return large amounts of data (e.g., `/api/modules`), pagination support may be added in future versions.

### Data Types

- **module_name**: String - The name of a Metasploit module.
- **module_type**: String - One of: "Exploit", "Auxiliary", "Post", "Payload", etc.
- **content**: String - Raw module content or description text.
- **question**: String - A natural language question about Metasploit modules.
- **confidence**: Float - A value between 0 and 1 indicating confidence in a prediction.

## Example API Usage

### Workflow 1: Module Analysis

1. **Check if modules are loaded**:
   ```bash
   curl -X GET http://localhost:5000/api/health
   ```

2. **Get module list**:
   ```bash
   curl -X GET http://localhost:5000/api/modules
   ```

3. **Get details about a specific module**:
   ```bash
   curl -X GET http://localhost:5000/api/module/Sample%20Exploit
   ```

4. **Get statistics about all modules**:
   ```bash
   curl -X GET http://localhost:5000/api/stats
   ```

### Workflow 2: Machine Learning Model Training

1. **Reload modules from directory**:
   ```bash
   curl -X POST http://localhost:5000/api/reload
   ```

2. **Start model training**:
   ```bash
   curl -X POST http://localhost:5000/api/train
   ```

3. **Check training status**:
   ```bash
   curl -X GET http://localhost:5000/api/health
   ```

### Workflow 3: Question Answering

1. **Ask a question about a module**:
   ```bash
   curl -X POST http://localhost:5000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the purpose of Sample Exploit?"}'
   ```

2. **Predict the type of a new module**:
   ```bash
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"content": "class MetasploitModule < Msf::Exploit::Remote\n..."}'
   ```

## Error Codes and Handling

The API uses standard HTTP response codes to indicate the success or failure of a request:

- **200 OK**: Request succeeded
- **400 Bad Request**: Request parameters are missing or invalid
- **404 Not Found**: Requested resource not found
- **500 Internal Server Error**: Server error occurred during request processing

## Rate Limiting

Currently, the API does not implement rate limiting. For production deployments, consider adding rate limiting to prevent abuse.

## Future Enhancements

Planned enhancements for future versions:

1. **Authentication and Authorization**: Add secure API key authentication
2. **Pagination**: Support for large result sets
3. **Filtering and Sorting**: Advanced query options for module listing
4. **Module Version Tracking**: Track changes to modules over time
5. **Advanced Question Answering**: Improved NLP models for more complex questions
6. **Real-time Module Monitoring**: Watch for module updates and trigger analysis
7. **Interactive Visualizations**: API endpoints for visualization data

## Support and Feedback

For issues, suggestions, or feedback regarding the API, please open an issue in the project repository or contact the maintainers directly.