import logging

import duckdb
import threading
from typing import Any, Dict
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DuckDBIO:
    """
    DuckDB IO operations for persistent benchmark caching.
    
    Provides high-performance DuckDB operations with connection management
    and optimized queries for benchmark inference caching.
    DuckDB is an embedded database that doesn't require a separate server.
    """
    
    def __init__(self, 
                 db_path: str = "cache/benchmark_cache.duckdb",
                 read_only: bool = False):
        """
        Initialize DuckDB connection.
        
        Args:
            db_path: Path to the DuckDB database file
            read_only: Whether to open database in read-only mode
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DuckDB connection per thread
        self._local = threading.local()
        
        try:
            # Test connection
            with self._get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            logger.info(f"✅ Connected to DuckDB at {self.db_path}")
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to DuckDB: {e}")
    
    @contextmanager
    def _get_connection(self):
        """Get a DuckDB connection (one per thread)."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = duckdb.connect(str(self.db_path), read_only=self.read_only)
        
        try:
            yield self._local.connection
        except Exception:
            # Close connection on error to get a fresh one next time
            if hasattr(self._local, 'connection') and self._local.connection:
                self._local.connection.close()
                self._local.connection = None
            raise
    
    def execute(self, sql: str, params: list = []) -> Any:
        """
        Execute a SQL statement.
        
        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement
            
        Returns:
            Query result
        """
        try:
            with self._get_connection() as conn:
                if params:
                    return conn.execute(sql, params)
                else:
                    return conn.execute(sql)
        except Exception as e:
            logger.info(f"Error executing SQL: {e}")
            raise
    
    def fetchone(self, sql: str, params: list = []) -> Any:
        """
        Execute SQL and fetch one result.
        
        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement
            
        Returns:
            Single row result or None
        """
        try:
            with self._get_connection() as conn:
                if params:
                    result = conn.execute(sql, params).fetchone()
                else:
                    result = conn.execute(sql).fetchone()
                return result
        except Exception as e:
            logger.info(f"Error fetching one result: {e}")
            return None
    
    def fetchall(self, sql: str, params: list = []) -> list:
        """
        Execute SQL and fetch all results.
        
        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement
            
        Returns:
            List of rows
        """
        try:
            with self._get_connection() as conn:
                if params:
                    results = conn.execute(sql, params).fetchall()
                else:
                    results = conn.execute(sql).fetchall()
                return results
        except Exception as e:
            logger.info(f"Error fetching all results: {e}")
            return []
    
    def executemany(self, sql: str, params_list: list) -> bool:
        """
        Execute SQL with multiple parameter sets.
        
        Args:
            sql: SQL statement to execute
            params_list: List of parameter lists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                conn.executemany(sql, params_list)
                return True
        except Exception as e:
            logger.info(f"Error executing many: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get generic database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            db_size_mb = db_size / (1024 * 1024)
            
            return {
                'backend': 'duckdb',
                'database_file': str(self.db_path),
                'database_size_mb': round(db_size_mb, 2)
            }
        except Exception as e:
            logger.info(f"Error getting database stats: {e}")
            return {
                'backend': 'duckdb',
                'database_file': str(self.db_path),
                'database_size_mb': 0
            }
    
    def close_connections(self):
        """Close all connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None 