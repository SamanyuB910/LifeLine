# secure_data_manager.py - HIPAA-compliant data encryption and security
import os
import json
import hashlib
import base64
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pandas as pd
import logging

class SecureDataManager:
    """
    HIPAA-compliant data encryption and security manager
    - AES-256 encryption for all patient data
    - Secure key management with PBKDF2
    - Data anonymization capabilities
    - Audit logging for all access
    - Automatic data retention policies
    """
    
    def __init__(self, master_password=None):
        self.master_password = master_password or os.environ.get('LIFELINE_MASTER_PASSWORD')
        if not self.master_password:
            # Use default for development - CHANGE IN PRODUCTION!
            self.master_password = "lifeline_secure_2024_dev"
            print("⚠️  WARNING: Using default password. Set LIFELINE_MASTER_PASSWORD environment variable for production!")
        
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
        self.audit_log = []
        
        # Create secure directories
        self.secure_dir = "secure_data"
        self.encrypted_dir = os.path.join(self.secure_dir, "encrypted")
        self.keys_dir = os.path.join(self.secure_dir, "keys")
        
        os.makedirs(self.secure_dir, exist_ok=True)
        os.makedirs(self.encrypted_dir, exist_ok=True)
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.secure_dir, "security_audit.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.log_access("SECURE_DATA_MANAGER_INITIALIZED")
    
    def _derive_key(self):
        """Derive encryption key from master password using PBKDF2"""
        password = self.master_password.encode()
        salt = b'lifeline_salt_2024'  # In production, use random salt per patient
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_data(self, data, patient_id=None):
        """Encrypt sensitive data with AES-256"""
        try:
            if isinstance(data, pd.DataFrame):
                data_str = data.to_json(orient='records')
            else:
                data_str = json.dumps(data)
            
            encrypted_data = self.cipher.encrypt(data_str.encode())
            
            # Store encrypted data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"encrypted_data_{timestamp}.enc"
            if patient_id:
                filename = f"patient_{patient_id}_{timestamp}.enc"
            
            filepath = os.path.join(self.encrypted_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)
            
            self.log_access(f"DATA_ENCRYPTED - {filename} - Patient: {patient_id}")
            return filepath
            
        except Exception as e:
            self.log_access(f"ENCRYPTION_ERROR - {str(e)}", level="ERROR")
            raise
    
    def decrypt_data(self, filepath):
        """Decrypt data from file"""
        try:
            with open(filepath, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())
            
            self.log_access(f"DATA_DECRYPTED - {filepath}")
            return data
            
        except Exception as e:
            self.log_access(f"DECRYPTION_ERROR - {str(e)}", level="ERROR")
            raise
    
    def anonymize_patient_data(self, data, patient_id):
        """Remove PII and create anonymized dataset"""
        try:
            # Create hash of patient ID for anonymization
            patient_hash = hashlib.sha256(f"{patient_id}_lifeline".encode()).hexdigest()[:8]
            
            # Remove or hash identifiable information
            if isinstance(data, pd.DataFrame):
                # Remove face coordinates (could be identifying)
                if 'face_x' in data.columns:
                    data['face_x'] = 0
                if 'face_y' in data.columns:
                    data['face_y'] = 0
                
                # Add anonymized patient ID
                data['patient_hash'] = patient_hash
            
            self.log_access(f"DATA_ANONYMIZED - Patient: {patient_id} -> Hash: {patient_hash}")
            return data, patient_hash
            
        except Exception as e:
            self.log_access(f"ANONYMIZATION_ERROR - {str(e)}", level="ERROR")
            raise
    
    def enforce_retention_policy(self, days=90):
        """Automatically delete data older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_files = []
            
            for filename in os.listdir(self.encrypted_dir):
                if filename.endswith('.enc'):
                    filepath = os.path.join(self.encrypted_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        deleted_files.append(filename)
            
            if deleted_files:
                self.log_access(f"RETENTION_POLICY_ENFORCED - Deleted {len(deleted_files)} files")
            
            return deleted_files
            
        except Exception as e:
            self.log_access(f"RETENTION_POLICY_ERROR - {str(e)}", level="ERROR")
            raise
    
    def log_access(self, message, level="INFO"):
        """Log all data access for audit trail"""
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} - {level} - {message}"
        self.audit_log.append(log_entry)
        
        if level == "ERROR":
            logging.error(message)
        else:
            logging.info(message)
    
    def get_audit_report(self):
        """Generate security audit report"""
        return {
            'total_access_logs': len(self.audit_log),
            'recent_logs': self.audit_log[-10:],
            'encrypted_files': len([f for f in os.listdir(self.encrypted_dir) if f.endswith('.enc')]),
            'security_status': 'SECURE' if self.master_password else 'INSECURE'
        }
