import os
import boto3
import time
from typing import Any, Dict, Optional, List



class DynamoDBCacheIO:
    """
    Simplified cache IO using DynamoDB with direct operations.
    
    Architecture:
    - DynamoDB: Serverless NoSQL database for all cache data
    - Direct operations: Fast key-value lookups and batch operations
    - Pay-per-request: No capacity planning needed
    
    Features:
    - Fast key-value reads and writes 
    - Batch operations for efficiency
    - Auto-scaling and managed infrastructure
    """
    
    def __init__(self):
        """Initialize DynamoDB cache IO."""
        # Initialize DynamoDB
        self.table_name = "parrotlet_omni_benchmark_cache"
        self.region_name = os.getenv("AWS_REGION")
        
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region_name)
        self.client = boto3.client('dynamodb', region_name=self.region_name)
        
        # Initialize tables
        self._init_tables()
        print(f"✅ DynamoDB connected - Table: {self.table_name}, Region: {self.region_name}")
    
    def _init_tables(self):
        """Initialize DynamoDB tables for benchmark caching."""
        
        self.table = self.dynamodb.Table(self.table_name)
        self.table.table_status  # This will raise an exception if table doesn't exist
        
        # Check if metadata table exists
        metadata_table_name = f"{self.table_name}_run_metadata"
        self.metadata_table = self.dynamodb.Table(metadata_table_name)
        self.metadata_table.table_status
    
    
    def get_inference_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get a cached inference result by cache key from DynamoDB."""
        try:
            response = self.table.get_item(
                Key={'cache_key': cache_key}
            )
            
            if 'Item' in response:
                item = response['Item']
                return item
            return None
        except Exception as e:
            print(f"Error getting inference result: {e}")
            return None
    
    def batch_get_inference_results(self, cache_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple cached inference results by cache keys from DynamoDB."""
        if not cache_keys:
            return {}
        
        results = {}
        
        try:
            # DynamoDB batch_get_item has a limit of 100 items
            for i in range(0, len(cache_keys), 100):
                batch_keys = cache_keys[i:i+100]
                keys = [{'cache_key': key} for key in batch_keys]
                
                response = self.dynamodb.batch_get_item(
                    RequestItems={
                        self.table_name: {
                            'Keys': keys
                        }
                    }
                )
                
                if self.table_name in response['Responses']:
                    for item in response['Responses'][self.table_name]:
                        results[item['cache_key']] = item
            
            # Add None entries for missing keys
            for cache_key in cache_keys:
                if cache_key not in results:
                    results[cache_key] = None
            
            return results
            
        except Exception as e:
            print(f"Error batch getting inference results: {e}")
            # Return empty results for all keys on error
            return {key: None for key in cache_keys}  # type: ignore
    
    def batch_save_inference_results(self, inference_data_list: List[Dict[str, Any]]) -> bool:
        """Save multiple inference results to DynamoDB."""
        
        try:
            # Use DynamoDB batch writer for efficient batch operations
            with self.table.batch_writer() as batch:
                for data in inference_data_list:
                    data['created_at'] = int(time.time())
                    batch.put_item(Item=data)
            
            return True
            
        except Exception as e:
            print(f"Error batch saving inference results: {e}")
            return False
    
    def save_run_metadata(self, run_data: Dict[str, Any]) -> bool:
        """Save run metadata to DynamoDB."""
        try:
            run_data['created_at'] = int(time.time())
            self.metadata_table.put_item(
                Item={
                    **run_data
                }
            )
            return True
        except Exception as e:
            print(f"Error saving run metadata: {e}")
            return False
    
    def close_connections(self):
        """Close DynamoDB connections."""
        print("✅ DynamoDB connections closed")
