# Enhanced thearm_uzi Client Tool

This tool extends the original thearm_uzi vulnerability scanner with enhanced features and REST API integration to a Flask server for centralized reporting and management.

## Overview

This tool automates vulnerability scanning using Metasploit Framework by:

1. Connecting to a PostgreSQL database (the Metasploit database)
2. Retrieving random hosts from the database
3. Testing various exploits against these hosts
4. Reporting results to both the local database and a centralized API server

## Features

- **REST API Integration**: Communicates with a Flask server for vulnerability reporting
- **Modular Architecture**: Well-organized code structure for maintainability
- **Flexible Configuration**: Environment variables and command-line options
- **Enhanced Logging**: Comprehensive logging system
- **Improved Error Handling**: Robust exception management
- **Exploit Recommendations**: Can use API-recommended exploits
- **Detailed Reporting**: Configurable reporting depth
- **Session Tracking**: Reports and tracks Metasploit sessions

## Requirements

- Python 3.6+
- Metasploit Framework with RPC enabled
- PostgreSQL database
- Python packages:
  - psycopg2
  - requests
  - pymetasploit3
  - psutil

## Configuration

The tool can be configured through environment variables or command-line arguments:

### PostgreSQL Configuration
- `PG_USER`: PostgreSQL username (default: "postgres")
- `PG_PASSWORD`: PostgreSQL password
- `PG_DBNAME`: PostgreSQL database name (default: "msf")
- `PG_HOST`: PostgreSQL host (default: "postgres")

### Metasploit Configuration
- `MSF_RPC_HOST`: Metasploit RPC host (default: "localhost")
- `MSF_RPC_PORT`: Metasploit RPC port (default: 55553)
- `MSF_RPC_PASSWORD`: Metasploit RPC password
- `MSF_RPC_SSL`: Use SSL for RPC connection (default: true)

### API Configuration
- `API_SERVER_URL`: URL of the Flask API server (default: "http://localhost:5000")
- `API_KEY`: API key for authentication
- `USE_API_EXPLOITS`: Use exploits recommended by the API (default: true)
- `REPORT_TO_API`: Send scan results to the API server (default: true)

### Scanning Configuration
- `MODE_RUN`: Execution mode (0 = dry run, 1 = with payloads) (default: 0)
- `TARGET_OS`: Target operating system (default: "Linux")
- `MIN_EXPLOIT_DATE`: Minimum disclosure date for exploits (default: "2021-01-01")
- `SCAN_INTERVAL`: Seconds between host scans (default: 60)
- `MAX_EXPLOITS_PER_HOST`: Maximum number of exploits to try (default: 100)
- `REPORTING_DEPTH`: Reporting detail level (basic/standard/full) (default: "standard")

## Usage

```bash
# Basic usage
python enhanced_thearm_uzi.py

# Using command-line options
python enhanced_thearm_uzi.py --target-os Linux --api-server http://analyzer.local:5000 --api-key YOUR_API_KEY

# Using environment variables
export PG_PASSWORD="your_pg_password"
export MSF_RPC_PASSWORD="your_msf_password"
export API_KEY="your_api_key"
python enhanced_thearm_uzi.py
```

## API Integration

The tool integrates with a Flask-based API server (metasploit-analyzer) using the following endpoints:

### Host Management
- `GET /api/hosts` - List all registered hosts
- `POST /api/hosts` - Register a new host
- `GET /api/hosts/{host_id}` - Get host details
- `PUT /api/hosts/{host_id}` - Update host information

### Scan Management
- `GET /api/scans` - List all scans
- `POST /api/scans` - Start a new scan
- `GET /api/scans/{scan_id}` - Get scan details
- `PUT /api/scans/{scan_id}` - Update scan status

### Finding/Vulnerability Reporting
- `GET /api/findings` - List all findings
- `POST /api/findings` - Report a new finding
- `GET /api/findings/{finding_id}` - Get finding details

### Session Reporting
- `GET /api/sessions` - List all sessions
- `POST /api/sessions` - Report a new session
- `PUT /api/sessions/{session_id}` - Update session status

### Exploit Recommendations
- `GET /api/exploits/recommended` - Get recommended exploits for a target

## Data Flow

1. The tool retrieves a random host from the PostgreSQL database
2. It registers the host with the API server (if not already registered)
3. It starts a new scan in the API server
4. Optionally retrieves recommended exploits from the API server
5. Runs exploits against the target host
6. Reports findings, vulnerabilities, and sessions to the API server
7. Updates scan status to completed when finished

## Command-Line Options

```
--dry-run           Run in dry-run mode (no exploit execution)
--target-os OS      Target OS type (Linux, Windows, etc.)
--api-server URL    API server URL
--api-key KEY       API key for authentication
--msf-host HOST     Metasploit RPC host
--msf-port PORT     Metasploit RPC port
--msf-password PWD  Metasploit RPC password
--pg-host HOST      PostgreSQL host
--pg-user USER      PostgreSQL user
--pg-password PWD   PostgreSQL password
--pg-dbname NAME    PostgreSQL database name
--report-depth LEVEL Reporting depth (basic/standard/full)
--max-exploits NUM  Maximum exploits per host
--scan-interval SEC Seconds between host scans
