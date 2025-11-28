"""
Utilities for pushing generated reports to the external S3 microservice and
recording each operation inside the MySQL `file_operations` table.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional

import requests

# Try to import RenderS3Client from s3_functions
try:
    from s3_fucntions import RenderS3Client, create_direct_mysql_client
    RENDER_S3_AVAILABLE = True
except ImportError:
    RENDER_S3_AVAILABLE = False
    RenderS3Client = None
    create_direct_mysql_client = None


class S3ReportStorageRender:
    """
    Wrapper adapter for RenderS3Client that maintains the same interface as S3ReportStorage.
    Uses RenderS3Client from s3_functions.py which integrates with the S3 microservice and MySQL.
    """
    def __init__(
        self,
        api_base_url: str,
        db_manager=None,
        default_user_id: str = "system-report",
        default_module: str = "cashflow-report",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.default_user_id = default_user_id
        self.default_module = default_module
        
        # Build MySQL config from db_manager if available
        mysql_config = None
        if db_manager:
            mysql_config = {
                'host': db_manager.host,
                'port': db_manager.port,
                'user': db_manager.user,
                'password': db_manager.password,
                'database': db_manager.database,
                'autocommit': True,
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci'
            }
        
        # Initialize RenderS3Client
        try:
            self.render_client = RenderS3Client(
                api_base_url=self.api_base_url,
                mysql_config=mysql_config
            )
            print(f"✅ RenderS3Client initialized with {self.api_base_url}")
        except Exception as e:
            print(f"⚠️ Failed to initialize RenderS3Client: {e}")
            # Fallback to client without MySQL
            try:
                self.render_client = RenderS3Client(
                    api_base_url=self.api_base_url,
                    mysql_config=None
                )
                print(f"⚠️ RenderS3Client initialized without MySQL (fallback mode)")
            except Exception as fallback_error:
                print(f"❌ RenderS3Client initialization failed: {fallback_error}")
                self.render_client = None

    def store_json_report(
        self,
        report_payload: Dict[str, Any],
        user_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store JSON report using RenderS3Client"""
        if not self.render_client:
            print("⚠️ RenderS3Client not available, skipping storage")
            return
        
        try:
            # Convert to JSON bytes
            content = json.dumps(report_payload, indent=2).encode("utf-8")
            filename = self._build_filename("json")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Upload using RenderS3Client
                resolved_user_id = user_id or self.default_user_id
                resolved_module = module or self.default_module
                
                result = self.render_client.upload(
                    file_path=temp_file_path,
                    user_id=resolved_user_id,
                    custom_file_name=filename,
                    module=resolved_module
                )
                
                if result.get('success'):
                    print(f"✅ JSON report stored successfully: {filename}")
                else:
                    print(f"⚠️ JSON report storage failed: {result.get('error', 'Unknown error')}")
            finally:
                # Always delete temp file
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"⚠️ S3 storage error (non-critical): {e}")
            import traceback
            traceback.print_exc()

    def store_pdf_report(
        self,
        pdf_bytes: bytes,
        user_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store PDF report using RenderS3Client"""
        if not self.render_client:
            print("⚠️ RenderS3Client not available, skipping storage")
            return
        
        try:
            filename = self._build_filename("pdf")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Upload using RenderS3Client
                resolved_user_id = user_id or self.default_user_id
                resolved_module = module or self.default_module
                
                result = self.render_client.upload(
                    file_path=temp_file_path,
                    user_id=resolved_user_id,
                    custom_file_name=filename,
                    module=resolved_module
                )
                
                if result.get('success'):
                    print(f"✅ PDF report stored successfully: {filename}")
                else:
                    print(f"⚠️ PDF report storage failed: {result.get('error', 'Unknown error')}")
            finally:
                # Always delete temp file
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"⚠️ S3 storage error (non-critical): {e}")
            import traceback
            traceback.print_exc()

    def _build_filename(self, extension: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"cashflow_report_{timestamp}.{extension.lstrip('.')}"


class S3ReportStorage:
    def __init__(
        self,
        api_base_url: str,
        db_manager=None,
        default_user_id: str = "system-report",
        default_module: str = "cashflow-report",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.db_manager = db_manager
        self.default_user_id = default_user_id
        self.default_module = default_module
        self._table_initialized = False
        if self.db_manager:
            self._ensure_table()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def store_json_report(
        self,
        report_payload: Dict[str, Any],
        user_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            content = json.dumps(report_payload, indent=2).encode("utf-8")
            filename = self._build_filename("json")
            self._upload_bytes(
                content=content,
                file_name=filename,
                content_type="application/json",
                user_id=user_id,
                module=module,
                metadata=metadata or {"kind": "report_json"},
            )
        except Exception as e:
            # Don't raise - just log and continue
            print(f"⚠️ S3 storage error (non-critical): {e}")
            import traceback
            traceback.print_exc()

    def store_pdf_report(
        self,
        pdf_bytes: bytes,
        user_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        filename = self._build_filename("pdf")
        self._upload_bytes(
            content=pdf_bytes,
            file_name=filename,
            content_type="application/pdf",
            user_id=user_id,
            module=module,
            metadata=metadata or {"kind": "report_pdf"},
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_filename(self, extension: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"cashflow_report_{timestamp}.{extension.lstrip('.')}"

    def _upload_bytes(
        self,
        content: bytes,
        file_name: str,
        content_type: str,
        user_id: Optional[str],
        module: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ):
        if not content:
            return

        resolved_user_id = user_id or self.default_user_id
        resolved_module = module or self.default_module
        metadata = metadata or {}

        # Try to log operation, but don't fail if it doesn't work
        operation_id = None
        try:
            operation_id = self._insert_operation_record(
                operation_type="upload",
                user_id=resolved_user_id,
                module=resolved_module,
                file_name=file_name,
                content_type=content_type,
                file_size=len(content),
                metadata=metadata,
            )
        except Exception as db_error:
            print(f"⚠️ Failed to log operation (continuing anyway): {db_error}")

        try:
            url = f"{self.api_base_url}/api/upload/{resolved_user_id}/{file_name}"
            files = {
                "file": (file_name, BytesIO(content), content_type or "application/octet-stream")
            }
            # ✅ Reduce timeout to prevent hanging (10 seconds max)
            response = requests.post(url, files=files, timeout=10)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            # Try to update operation record, but don't fail if it doesn't work
            if operation_id:
                try:
                    self._update_operation_record(
                        operation_id,
                        {
                            "status": "failed",
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass  # Don't fail if update fails
            print(f"⚠️ Report upload failed (non-critical): {exc}")
            return

        if not result.get("success"):
            error_message = result.get("error", "Unknown upload error")
            if operation_id:
                try:
                    self._update_operation_record(
                        operation_id,
                        {
                            "status": "failed",
                            "error": error_message,
                            "metadata": {"response": result},
                        },
                    )
                except Exception:
                    pass  # Don't fail if update fails
            print(f"⚠️ Report upload unsuccessful: {error_message}")
            return

        stored_info = result.get("file") or result
        if operation_id:
            try:
                self._update_operation_record(
                    operation_id,
                    {
                        "status": "completed",
                        "stored_name": stored_info.get("storedName") or stored_info.get("fileName") or file_name,
                        "s3_url": stored_info.get("url") or stored_info.get("downloadUrl"),
                        "s3_key": stored_info.get("s3Key"),
                        "s3_bucket": stored_info.get("bucket"),
                        "metadata": {
                            **metadata,
                            "upload_response": stored_info,
                        },
                    },
                )
            except Exception:
                pass  # Don't fail if update fails

    # ------------------------------------------------------------------ #
    # Database bookkeeping
    # ------------------------------------------------------------------ #

    def _ensure_table(self):
        if self._table_initialized or not self.db_manager:
            return

        conn = None
        cursor = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS file_operations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    operation_type ENUM('upload','download','export') NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    file_name VARCHAR(500) NOT NULL,
                    original_name VARCHAR(500),
                    stored_name VARCHAR(500),
                    s3_url TEXT,
                    s3_key VARCHAR(1000),
                    s3_bucket VARCHAR(255),
                    file_type VARCHAR(50),
                    file_size BIGINT,
                    content_type VARCHAR(255),
                    export_format VARCHAR(20),
                    record_count INT,
                    status ENUM('pending','processing','completed','failed') DEFAULT 'pending',
                    error TEXT,
                    metadata JSON,
                    platform VARCHAR(50) DEFAULT 'Direct',
                    module VARCHAR(100) DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    INDEX idx_user_id (user_id),
                    INDEX idx_operation_type (operation_type),
                    INDEX idx_module (module),
                    INDEX idx_status (status),
                    INDEX idx_created_at (created_at),
                    INDEX idx_file_type (file_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            conn.commit()  # ✅ Commit table creation
            self._table_initialized = True
            print("✅ file_operations table ensured successfully")
        except Exception as exc:
            print(f"⚠️ Unable to ensure file_operations table: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            # ✅ Always close cursor
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            # Note: Don't close connection - db_manager handles connection pool

    def _insert_operation_record(
        self,
        operation_type: str,
        user_id: str,
        module: str,
        file_name: str,
        content_type: str,
        file_size: int,
        metadata: Dict[str, Any],
    ) -> Optional[int]:
        if not self.db_manager:
            return None

        conn = None
        cursor = None
        try:
            # Try to get connection
            try:
                conn = self.db_manager.get_connection()
                if not conn:
                    print("⚠️ Could not get database connection for report storage")
                    return None
                
                # Check if connection is still valid (mysql.connector has is_connected())
                if hasattr(conn, 'is_connected') and not conn.is_connected():
                    # Connection lost, get a new one
                    conn = self.db_manager.get_connection()
            except Exception as conn_error:
                print(f"⚠️ Database connection error, skipping storage logging: {conn_error}")
                return None
            
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO file_operations
                (operation_type, user_id, file_name, original_name, status,
                 file_type, file_size, content_type, metadata, module)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    operation_type,
                    user_id,
                    file_name,
                    file_name,
                    "pending",
                    os.path.splitext(file_name)[1].lstrip("."),
                    file_size,
                    content_type,
                    json.dumps(metadata or {}),
                    module,
                ),
            )
            conn.commit()
            operation_id = cursor.lastrowid
            cursor.close()
            return operation_id
        except Exception as exc:
            # Log but don't fail - storage is optional
            print(f"⚠️ Failed to insert file_operations record: {exc}")
            print("📊 Report will still be generated, but storage tracking skipped")
            return None
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def _update_operation_record(self, operation_id: Optional[int], updates: Dict[str, Any]):
        if not self.db_manager or not operation_id:
            return

        conn = None
        cursor = None
        try:
            # Try to get connection
            try:
                conn = self.db_manager.get_connection()
                if not conn:
                    return
                
                # Check if connection is still valid (mysql.connector has is_connected())
                if hasattr(conn, 'is_connected') and not conn.is_connected():
                    # Connection lost, get a new one
                    conn = self.db_manager.get_connection()
            except Exception as conn_error:
                print(f"⚠️ Database connection error, skipping storage update: {conn_error}")
                return

            set_clauses = []
            values = []
            for column, value in updates.items():
                if column == "metadata" and value is not None:
                    set_clauses.append("metadata = %s")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{column} = %s")
                    values.append(value)

            set_clauses.append("updated_at = %s")
            values.append(datetime.utcnow())
            if updates.get("status") == "completed":
                set_clauses.append("completed_at = %s")
                values.append(datetime.utcnow())

            values.append(operation_id)

            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE file_operations SET {', '.join(set_clauses)} WHERE id = %s",
                tuple(values),
            )
            conn.commit()
        except Exception as exc:
            # Log but don't fail - storage is optional
            print(f"⚠️ Failed to update file_operations record {operation_id}: {exc}")
        finally:
            # Always close cursor and connection
            try:
                if cursor:
                    cursor.close()
            except Exception:
                pass
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

