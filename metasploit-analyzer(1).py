import os
import re
import json
import logging
import datetime
import threading
import pickle
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, jsonify, request, Blueprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/metasploit-analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('metasploit-analyzer')

class MetasploitModuleParser:
    """
    Parser for Metasploit module files (.rb)
    Extracts metadata and content for analysis
    """
    
    def __init__(self):
        self.initialize_pattern = re.compile(r'def\s+initialize\s*\(\s*info\s*=\s*\{\s*\}\s*\)(.*?)(?:def|end)', re.DOTALL)
        self.name_pattern = re.compile(r"'Name'\s*=>\s*['\"](.*?)['\"]", re.DOTALL)
        self.desc_pattern = re.compile(r"'Description'\s*=>\s*%q{(.*?)}", re.DOTALL)
        self.license_pattern = re.compile(r"'License'\s*=>\s*(.*?),", re.DOTALL)
        self.author_pattern = re.compile(r"'Author'\s*=>\s*\[(.*?)\]", re.DOTALL)
        self.references_pattern = re.compile(r"'References'\s*=>\s*\[(.*?)\]", re.DOTALL)
        self.payload_pattern = re.compile(r"'Payload'\s*=>\s*{(.*?)}", re.DOTALL)
        self.targets_pattern = re.compile(r"'Targets'\s*=>\s*\[(.*?)\]", re.DOTALL)
        self.disclosure_pattern = re.compile(r"'DisclosureDate'\s*=>\s*['\"](.*?)['\"]", re.DOTALL)
        self.module_type_pattern = re.compile(r"class\s+MetasploitModule\s*<\s*Msf::(\w+)::", re.DOTALL)
        self.include_pattern = re.compile(r"include\s+(.*?)$", re.MULTILINE)
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Metasploit module file and extract relevant information
        
        Args:
            file_path: Path to the Ruby (.rb) file
            
        Returns:
            Dictionary with extracted module information
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            module_info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'raw_content': content,
                'module_type': None,
                'name': None,
                'description': None,
                'license': None,
                'authors': [],
                'references': [],
                'payload_info': None,
                'targets': [],
                'disclosure_date': None,
                'includes': []
            }
            
            # Extract module type
            module_type_match = self.module_type_pattern.search(content)
            if module_type_match:
                module_info['module_type'] = module_type_match.group(1)
                
            # Extract included modules
            include_matches = self.include_pattern.finditer(content)
            if include_matches:
                for match in include_matches:
                    module_info['includes'].append(match.group(1).strip())
            
            # Extract initialize block
            init_match = self.initialize_pattern.search(content)
            if init_match:
                initialize_content = init_match.group(1)
                
                # Extract name
                name_match = self.name_pattern.search(initialize_content)
                if name_match:
                    module_info['name'] = name_match.group(1).strip()
                
                # Extract description
                desc_match = self.desc_pattern.search(initialize_content)
                if desc_match:
                    module_info['description'] = desc_match.group(1).strip()
                
                # Extract license
                license_match = self.license_pattern.search(initialize_content)
                if license_match:
                    module_info['license'] = license_match.group(1).strip()
                
                # Extract authors
                authors_match = self.author_pattern.search(initialize_content)
                if authors_match:
                    authors_str = authors_match.group(1)
                    # Extract individual authors
                    author_entries = re.findall(r"['\"](.*?)['\"]", authors_str)
                    module_info['authors'] = [a.strip() for a in author_entries]
                
                # Extract references
                refs_match = self.references_pattern.search(initialize_content)
                if refs_match:
                    refs_str = refs_match.group(1)
                    # Extract reference entries
                    ref_entries = re.findall(r'\[\s*[\'"]([^\'"]*)[\'"]\s*,\s*[\'"]([^\'"]*)[\'"]\s*\]', refs_str)
                    module_info['references'] = [{'type': r[0].strip(), 'value': r[1].strip()} for r in ref_entries]
                
                # Extract payload information
                payload_match = self.payload_pattern.search(initialize_content)
                if payload_match:
                    module_info['payload_info'] = payload_match.group(1).strip()
                
                # Extract targets
                targets_match = self.targets_pattern.search(initialize_content)
                if targets_match:
                    targets_str = targets_match.group(1)
                    # This is a simplification - actual target parsing would be more complex
                    target_entries = re.findall(r'\[\s*[\'"]([^\'"]*)[\'"]', targets_str)
                    module_info['targets'] = [t.strip() for t in target_entries]
                
                # Extract disclosure date
                disclosure_match = self.disclosure_pattern.search(initialize_content)
                if disclosure_match:
                    module_info['disclosure_date'] = disclosure_match.group(1).strip()
            
            return module_info
        except Exception as e:
            logger.error(f"Error parsing module file {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'error': str(e)
            }

class MetasploitModuleAnalyzer:
    """
    Analyzes Metasploit modules using machine learning
    """
    
    def __init__(self, models_dir: str = '/opt/metasploit-analyzer/models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.modules_data = []
        self.parser = MetasploitModuleParser()
        
        # ML models
        self.tfidf_vectorizer = None
        self.type_classifier = None
        self.question_classifier = None
        
        # Track if models are loaded
        self.models_loaded = False
        
        # Load models if available
        self._load_latest_models()
    
    def _load_latest_models(self) -> bool:
        """
        Load the latest models from the models directory
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            # Find latest model directory
            model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
            if not model_dirs:
                logger.info("No model directories found. Will train new models.")
                return False
            
            # Sort by directory name (timestamp)
            latest_model_dir = sorted(model_dirs)[-1]
            
            # Load vectorizer
            vectorizer_path = latest_model_dir / 'tfidf_vectorizer.pkl'
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            
            # Load type classifier
            type_classifier_path = latest_model_dir / 'type_classifier.pkl'
            if type_classifier_path.exists():
                with open(type_classifier_path, 'rb') as f:
                    self.type_classifier = pickle.load(f)
            
            # Load question classifier
            question_classifier_path = latest_model_dir / 'question_classifier.pkl'
            if question_classifier_path.exists():
                with open(question_classifier_path, 'rb') as f:
                    self.question_classifier = pickle.load(f)
            
            if self.tfidf_vectorizer and self.type_classifier:
                logger.info(f"Successfully loaded models from {latest_model_dir}")
                self.models_loaded = True
                return True
            else:
                logger.warning(f"Some models missing in {latest_model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _save_models(self) -> str:
        """
        Save the current models to a timestamped directory
        
        Returns:
            Path to the saved models directory
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = self.models_dir / timestamp
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        with open(model_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save type classifier
        with open(model_dir / 'type_classifier.pkl', 'wb') as f:
            pickle.dump(self.type_classifier, f)
        
        # Save question classifier if exists
        if self.question_classifier:
            with open(model_dir / 'question_classifier.pkl', 'wb') as f:
                pickle.dump(self.question_classifier, f)
        
        # Create symlink to latest
        latest_link = self.models_dir / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(model_dir.name)
        
        logger.info(f"Models saved to {model_dir}")
        return str(model_dir)
    
    def load_modules(self, modules_dir: str) -> int:
        """
        Load and parse Metasploit modules from the specified directory
        
        Args:
            modules_dir: Directory containing Metasploit module files
            
        Returns:
            Number of modules loaded
        """
        self.modules_data = []
        count = 0
        
        for root, _, files in os.walk(modules_dir):
            for file in files:
                if file.endswith('.rb'):
                    file_path = os.path.join(root, file)
                    module_info = self.parser.parse_file(file_path)
                    if 'error' not in module_info:
                        self.modules_data.append(module_info)
                        count += 1
        
        logger.info(f"Loaded {count} Metasploit modules")
        return count
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train machine learning models on the loaded modules
        
        Returns:
            Dictionary with training results and metrics
        """
        if not self.modules_data:
            logger.error("No modules loaded. Cannot train models.")
            return {"error": "No modules loaded"}
        
        # Create a DataFrame for analysis
        df = pd.DataFrame(self.modules_data)
        
        # Prepare features and labels for type classification
        # Combine relevant text fields for feature extraction
        df['text_features'] = df.apply(
            lambda row: f"{row.get('name', '')} {row.get('description', '')} {' '.join(row.get('authors', []))}",
            axis=1
        )
        
        # Filter out rows without module_type
        df_valid = df.dropna(subset=['module_type'])
        
        if len(df_valid) < 10:
            logger.error(f"Not enough valid modules ({len(df_valid)}) for training. Need at least 10.")
            return {"error": "Not enough valid modules for training"}
        
        # Split data for training and testing
        X = df_valid['text_features'].values
        y = df_valid['module_type'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Train Random Forest classifier for module type prediction
        self.type_classifier = Pipeline([
            ('tfidf', self.tfidf_vectorizer),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.type_classifier.fit(X_train, y_train)
        
        # Evaluate the model
        train_accuracy = self.type_classifier.score(X_train, y_train)
        test_accuracy = self.type_classifier.score(X_test, y_test)
        
        y_pred = self.type_classifier.predict(X_test)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = cross_val_score(self.type_classifier, X, y, cv=5)
        
        # Save the models
        models_dir = self._save_models()
        
        # Set models as loaded
        self.models_loaded = True
        
        # Return training results
        return {
            "models_dir": models_dir,
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "cross_validation_scores": cv_scores.tolist(),
            "cross_validation_mean": float(cv_scores.mean()),
            "classification_report": classification_rep,
            "num_modules_trained": len(df_valid)
        }
    
    def predict_module_type(self, module_content: str) -> Dict[str, Any]:
        """
        Predict the type of a module based on its content
        
        Args:
            module_content: Content of the module to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        try:
            # Extract features from the module content
            # For a real input, we'd parse it first, but here we'll use it directly
            prediction = self.type_classifier.predict([module_content])[0]
            probabilities = self.type_classifier.predict_proba([module_content])[0]
            
            # Map probabilities to class names
            classes = self.type_classifier.classes_
            probs_dict = {class_name: float(prob) for class_name, prob in zip(classes, probabilities)}
            
            return {
                "predicted_type": prediction,
                "probabilities": probs_dict
            }
        except Exception as e:
            logger.error(f"Error predicting module type: {str(e)}")
            return {"error": str(e)}
    
    def extract_rules(self) -> Dict[str, Any]:
        """
        Extract patterns and rules from the analyzed modules
        
        Returns:
            Dictionary with extracted rules and patterns
        """
        if not self.modules_data:
            return {"error": "No modules loaded"}
        
        # Create a DataFrame for analysis
        df = pd.DataFrame(self.modules_data)
        
        # Common patterns analysis
        rules = {}
        
        # 1. Module type distribution
        rules["module_type_distribution"] = df['module_type'].value_counts().to_dict()
        
        # 2. Common authors
        all_authors = []
        for authors in df['authors']:
            if isinstance(authors, list):
                all_authors.extend(authors)
        
        if all_authors:
            top_authors = pd.Series(all_authors).value_counts().head(10).to_dict()
            rules["top_authors"] = top_authors
        
        # 3. Reference types distribution
        ref_types = []
        for refs in df['references']:
            if isinstance(refs, list):
                for ref in refs:
                    if isinstance(ref, dict) and 'type' in ref:
                        ref_types.append(ref['type'])
        
        if ref_types:
            ref_type_dist = pd.Series(ref_types).value_counts().to_dict()
            rules["reference_type_distribution"] = ref_type_dist
        
        # 4. Common target platforms
        all_targets = []
        for targets in df['targets']:
            if isinstance(targets, list):
                all_targets.extend(targets)
        
        if all_targets:
            top_targets = pd.Series(all_targets).value_counts().head(10).to_dict()
            rules["top_target_platforms"] = top_targets
        
        # 5. Disclosure date analysis
        if 'disclosure_date' in df.columns:
            # Convert to datetime
            valid_dates = []
            for date_str in df['disclosure_date'].dropna():
                try:
                    valid_dates.append(pd.to_datetime(date_str))
                except:
                    pass
            
            if valid_dates:
                date_series = pd.Series(valid_dates)
                rules["disclosure_date_stats"] = {
                    "oldest": date_series.min().strftime('%Y-%m-%d'),
                    "newest": date_series.max().strftime('%Y-%m-%d'),
                    "by_year": date_series.dt.year.value_counts().sort_index().to_dict()
                }
        
        # 6. Common includes analysis
        all_includes = []
        for includes in df['includes']:
            if isinstance(includes, list):
                all_includes.extend(includes)
        
        if all_includes:
            top_includes = pd.Series(all_includes).value_counts().head(10).to_dict()
            rules["top_includes"] = top_includes
        
        return rules
    
    def generate_stats(self) -> Dict[str, Any]:
        """
        Generate statistics about the analyzed modules
        
        Returns:
            Dictionary with module statistics
        """
        if not self.modules_data:
            return {"error": "No modules loaded"}
        
        # Basic stats
        total_modules = len(self.modules_data)
        module_types = {}
        authors_count = 0
        references_count = 0
        with_targets = 0
        with_payload = 0
        
        for module in self.modules_data:
            # Count module types
            module_type = module.get('module_type')
            if module_type:
                module_types[module_type] = module_types.get(module_type, 0) + 1
            
            # Count authors
            authors = module.get('authors', [])
            authors_count += len(authors)
            
            # Count references
            references = module.get('references', [])
            references_count += len(references)
            
            # Count targets
            if module.get('targets') and len(module.get('targets', [])) > 0:
                with_targets += 1
            
            # Count payload info
            if module.get('payload_info'):
                with_payload += 1
        
        return {
            "total_modules": total_modules,
            "module_types": module_types,
            "total_authors": authors_count,
            "total_references": references_count,
            "modules_with_targets": with_targets,
            "modules_with_payload_info": with_payload,
            "average_authors_per_module": authors_count / total_modules if total_modules > 0 else 0,
            "average_references_per_module": references_count / total_modules if total_modules > 0 else 0
        }
    
    def generate_qa_pairs(self, module_name: str = None) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs for a module or all modules
        
        Args:
            module_name: Optional name of specific module
            
        Returns:
            List of question-answer pairs
        """
        qa_pairs = []
        
        # Filter modules if module_name is specified
        modules_to_process = []
        if module_name:
            for module in self.modules_data:
                if module.get('name') == module_name:
                    modules_to_process.append(module)
                    break
        else:
            modules_to_process = self.modules_data
        
        for module in modules_to_process:
            # Basic information QA pairs
            name = module.get('name')
            if name:
                qa_pairs.append({
                    "question": f"What is {name}?",
                    "answer": module.get('description', "No description available.")
                })
                
                qa_pairs.append({
                    "question": f"Who authored {name}?",
                    "answer": ", ".join(module.get('authors', ["Unknown"]))
                })
                
                module_type = module.get('module_type')
                if module_type:
                    qa_pairs.append({
                        "question": f"What type of module is {name}?",
                        "answer": f"{name} is a {module_type} module."
                    })
                
                targets = module.get('targets', [])
                if targets:
                    qa_pairs.append({
                        "question": f"What are the targets for {name}?",
                        "answer": ", ".join(targets)
                    })
                
                disclosure_date = module.get('disclosure_date')
                if disclosure_date:
                    qa_pairs.append({
                        "question": f"When was {name} disclosed?",
                        "answer": f"The {name} module was disclosed on {disclosure_date}."
                    })
                
                references = module.get('references', [])
                if references:
                    refs_text = "; ".join([f"{ref.get('type', 'Reference')}: {ref.get('value', '')}" 
                                          for ref in references if isinstance(ref, dict)])
                    qa_pairs.append({
                        "question": f"What are the references for {name}?",
                        "answer": refs_text
                    })
        
        return qa_pairs
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about modules using the trained models
        
        Args:
            question: Question text
            
        Returns:
            Dictionary with answer and confidence
        """
        if not self.modules_data:
            return {"error": "No modules loaded"}
        
        # This is a simplified implementation - a real QA system would be more complex
        # Here, we'll use simple keyword matching
        
        question_lower = question.lower()
        
        # Try to identify what module the question is about
        module_name = None
        for module in self.modules_data:
            name = module.get('name')
            if name and name.lower() in question_lower:
                module_name = name
                break
        
        if not module_name:
            # General question about all modules
            if 'how many' in question_lower and 'modules' in question_lower:
                return {
                    "answer": f"There are {len(self.modules_data)} Metasploit modules loaded.",
                    "confidence": 0.9
                }
            elif 'what types' in question_lower and 'modules' in question_lower:
                module_types = {}
                for module in self.modules_data:
                    module_type = module.get('module_type')
                    if module_type:
                        module_types[module_type] = module_types.get(module_type, 0) + 1
                
                types_text = ", ".join([f"{k} ({v})" for k, v in module_types.items()])
                return {
                    "answer": f"The loaded modules include these types: {types_text}",
                    "confidence": 0.8
                }
            else:
                return {
                    "answer": "I don't have enough information to answer that question.",
                    "confidence": 0.2
                }
        else:
            # Question about a specific module
            # Generate QA pairs for this module
            qa_pairs = self.generate_qa_pairs(module_name)
            
            # Find the best matching question
            best_match = None
            best_score = 0
            
            for qa in qa_pairs:
                # Simple word overlap score
                q_words = set(qa['question'].lower().split())
                question_words = set(question_lower.split())
                overlap = len(q_words.intersection(question_words))
                score = overlap / len(q_words)
                
                if score > best_score:
                    best_score = score
                    best_match = qa
            
            if best_match and best_score > 0.4:
                return {
                    "answer": best_match['answer'],
                    "confidence": best_score,
                    "module": module_name
                }
            else:
                # Find the module and provide basic info
                for module in self.modules_data:
                    if module.get('name') == module_name:
                        return {
                            "answer": f"{module_name}: {module.get('description', 'No description available.')}",
                            "confidence": 0.5,
                            "module": module_name
                        }
            
            return {
                "answer": f"I found a module named {module_name}, but I don't have enough information to answer your specific question about it.",
                "confidence": 0.3,
                "module": module_name
            }


# Flask API implementation
api_bp = Blueprint('api', __name__, url_prefix='/api')

class MetasploitAnalyzerAPI:
    """
    Flask Blueprint implementation for the Metasploit Module Analyzer API
    """
    
    def __init__(self, analyzer: MetasploitModuleAnalyzer, modules_dir: str):
        self.analyzer = analyzer
        self.modules_dir = modules_dir
        self.training_lock = threading.Lock()
        self.is_training = False
        
    def register_routes(self, blueprint: Blueprint):
        """
        Register API routes to the blueprint
        
        Args:
            blueprint: Flask blueprint to register routes on
        """
        # API documentation
        blueprint.route('/', methods=['GET', 'OPTIONS', 'HEAD'])(self.list_endpoints)
        
        # Health check
        blueprint.route('/health', methods=['GET', 'OPTIONS', 'HEAD'])(self.health_check)
        
        # Module endpoints
        blueprint.route('/modules', methods=['GET', 'OPTIONS', 'HEAD'])(self.list_modules)
        blueprint.route('/module/<module_name>', methods=['GET', 'OPTIONS', 'HEAD'])(self.get_module)
        
        # Analysis endpoints
        blueprint.route('/predict', methods=['POST', 'OPTIONS'])(self.predict_module_type)
        blueprint.route('/rules', methods=['GET', 'OPTIONS', 'HEAD'])(self.get_rules)
        blueprint.route('/stats', methods=['GET', 'OPTIONS', 'HEAD'])(self.get_stats)
        
        # Q&A endpoints
        blueprint.route('/ask', methods=['POST', 'OPTIONS'])(self.ask_question)
        blueprint.route('/questions/log', methods=['GET', 'OPTIONS', 'HEAD'])(self.log_questions)
        
        # Model management
        blueprint.route('/train', methods=['POST', 'OPTIONS'])(self.start_training)
        blueprint.route('/reload', methods=['POST', 'OPTIONS'])(self.reload_modules)
    
    def list_endpoints(self):
        """List all available API endpoints"""
        endpoints = [
            {
                "url": "/api",
                "endpoint": "list_endpoints",
                "description": "Liste tous les endpoints disponibles avec leur description et méthodes",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/health",
                "endpoint": "health_check",
                "description": "Vérification de l'état du service",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/modules",
                "endpoint": "list_modules",
                "description": "Lister les modules disponibles",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/module/<module_name>",
                "endpoint": "get_module",
                "description": "Obtenir les détails d'un module spécifique",
                "methods": ["GET", "HEAD", "OPTIONS"],
                "parameters": ["module_name"]
            },
            {
                "url": "/api/predict",
                "endpoint": "predict_module_type",
                "description": "Prédire le type d'un module à partir de son contenu",
                "methods": ["POST", "OPTIONS"]
            },
            {
                "url": "/api/rules",
                "endpoint": "get_rules",
                "description": "Obtenir les règles extraites",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/stats",
                "endpoint": "get_stats",
                "description": "Obtenir des statistiques sur les modules",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/ask",
                "endpoint": "ask_question",
                "description": "Poser une question sur les modules",
                "methods": ["POST", "OPTIONS"]
            },
            {
                "url": "/api/questions/log",
                "endpoint": "log_questions",
                "description": "Affiche les questions générées dans les logs",
                "methods": ["GET", "HEAD", "OPTIONS"]
            },
            {
                "url": "/api/train",
                "endpoint": "start_training",
                "description": "Déclencher un entraînement du modèle",
                "methods": ["POST", "OPTIONS"]
            },
            {
                "url": "/api/reload",
                "endpoint": "reload_modules",
                "description": "Recharger les modules depuis le répertoire",
                "methods": ["POST", "OPTIONS"]
            }
        ]
        
        return jsonify({
            "service": "Metasploit Module Analyzer API",
            "version": "1.0.0",
            "total_endpoints": len(endpoints),
            "endpoints": endpoints
        })
    
    def health_check(self):
        """Check the health status of the service"""
        status = {
            "status": "ok",
            "timestamp": datetime.datetime.now().isoformat(),
            "models_loaded": self.analyzer.models_loaded,
            "modules_loaded": len(self.analyzer.modules_data),
            "training_in_progress": self.is_training
        }
        return jsonify(status)
    
    def list_modules(self):
        """List all available modules"""
        if not self.analyzer.modules_data:
            return jsonify({
                "error": "No modules loaded",
                "modules": []
            })
        
        modules = []
        for module in self.analyzer.modules_data:
            modules.append({
                "name": module.get('name'),
                "file_name": module.get('file_name'),
                "module_type": module.get('module_type'),
                "authors": module.get('authors'),
                "disclosure_date": module.get('disclosure_date')
            })
        
        return jsonify({
            "total": len(modules),
            "modules": modules
        })
    
    def get_module(self, module_name):
        """Get details for a specific module"""
        for module in self.analyzer.modules_data:
            if module.get('name') == module_name:
                # Remove raw_content to keep response size reasonable
                module_copy = module.copy()
                if 'raw_content' in module_copy:
                    del module_copy['raw_content']
                return jsonify(module_copy)
        
        return jsonify({
            "error": f"Module '{module_name}' not found"
        }), 404
    
    def predict_module_type(self):
        """Predict the type of a module based on its content"""
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400
        
        content = request.json.get('content')
        if not content:
            return jsonify({
                "error": "Module content required"
            }), 400
        
        result = self.analyzer.predict_module_type(content)
        return jsonify(result)
    
    def get_rules(self):
        """Get extracted rules and patterns"""
        rules = self.analyzer.extract_rules()
        return jsonify(rules)
    
    def get_stats(self):
        """Get statistics about loaded modules"""
        stats = self.analyzer.generate_stats()
        return jsonify(stats)
    
    def ask_question(self):
        """Answer a question about Metasploit modules"""
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400
        
        question = request.json.get('question')
        if not question:
            return jsonify({
                "error": "Question text required"
            }), 400
        
        answer = self.analyzer.answer_question(question)
        return jsonify(answer)
    
    def log_questions(self):
        """Generate and log sample questions"""
        # Generate QA pairs for a random module or first if available
        if not self.analyzer.modules_data:
            return jsonify({
                "error": "No modules loaded"
            })
        
        # Pick a random module
        import random
        module = random.choice(self.analyzer.modules_data)
        module_name = module.get('name')
        
        qa_pairs = self.analyzer.generate_qa_pairs(module_name)
        
        # Log questions
        for qa in qa_pairs:
            logger.info(f"Generated Q&A: {qa['question']} -> {qa['answer']}")
        
        return jsonify({
            "module": module_name,
            "questions_generated": len(qa_pairs),
            "sample_questions": [qa['question'] for qa in qa_pairs[:5]]
        })
    
    def start_training(self):
        """Start model training in a background thread"""
        if self.is_training:
            return jsonify({
                "error": "Training already in progress",
                "status": "in_progress"
            })
        
        def train_task():
            try:
                with self.training_lock:
                    self.is_training = True
                    logger.info("Starting model training")
                    results = self.analyzer.train_models()
                    logger.info(f"Training completed: {results}")
                    self.is_training = False
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                self.is_training = False
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_task)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "training_started",
            "message": "Model training has been started in the background"
        })
    
    def reload_modules(self):
        """Reload modules from the specified directory"""
        try:
            count = self.analyzer.load_modules(self.modules_dir)
            return jsonify({
                "status": "success",
                "modules_loaded": count
            })
        except Exception as e:
            logger.error(f"Error reloading modules: {str(e)}")
            return jsonify({
                "error": str(e)
            }), 500


def create_app(modules_dir: str = '/opt/metasploit-modules', 
               models_dir: str = '/opt/metasploit-analyzer/models'):
    """
    Create and configure the Flask application
    
    Args:
        modules_dir: Directory containing Metasploit modules
        models_dir: Directory for storing trained models
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Create analyzer instance
    analyzer = MetasploitModuleAnalyzer(models_dir=models_dir)
    
    # Load modules if directory exists
    if os.path.exists(modules_dir):
        analyzer.load_modules(modules_dir)
    else:
        logger.warning(f"Modules directory {modules_dir} does not exist")
    
    # Register API routes
    api_handler = MetasploitAnalyzerAPI(analyzer, modules_dir)
    api_handler.register_routes(api_bp)
    app.register_blueprint(api_bp)
    
    # Add basic route for root
    @app.route('/')
    def index():
        return jsonify({
            "service": "Metasploit Module Analyzer",
            "description": "API for analyzing Metasploit modules using machine learning",
            "api_docs": "/api"
        })
    
    return app


# Main entry point for running the application
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Metasploit Module Analyzer Service')
    parser.add_argument('--modules-dir', type=str, default='/opt/metasploit-modules',
                        help='Directory containing Metasploit modules')
    parser.add_argument('--models-dir', type=str, default='/opt/metasploit-analyzer/models',
                        help='Directory for storing trained models')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind the server')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create and run the application
    app = create_app(args.modules_dir, args.models_dir)
    
    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        # Use waitress for production deployment
        from waitress import serve
        logger.info(f"Starting production server on {args.host}:{args.port}")
        serve(app, host=args.host, port=args.port)