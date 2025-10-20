"""
MongoDB storage implementation for Crypto Futures Trading System.

This module provides database operations for storing and retrieving
OHLCV data, features, labels, and predictions.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import logging

from ...config.settings import DatabaseConfig

logger = logging.getLogger(__name__)


class MongoDBStorage:
    """
    MongoDB storage class for crypto trading data.
    
    Handles all database operations including data storage,
    retrieval, and management for the trading system.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize MongoDB storage.
        
        Args:
            config: Database configuration settings
        """
        self.config = config
        self.client: Optional[MongoClient] = None
        self.db = None
        self._collections = {}
    
    def _ensure_connected(self) -> bool:
        """Ensure there is an active DB connection and initialized collections."""
        if self.db is not None and self._collections:
            return True
        return self.connect()
    
    def connect(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username if self.config.username else None,
                password=self.config.password if self.config.password else None,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            self.client.server_info()
            self.db = self.client[self.config.database]
            
            # Initialize collections
            self._initialize_collections()
            
            logger.info(f"Connected to MongoDB: {self.config.host}:{self.config.port}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def _initialize_collections(self):
        """Initialize database collections with proper indexes."""
        collections_config = {
            'ohlcv_data': {
                'indexes': [
                    [('symbol', ASCENDING), ('timeframe', ASCENDING), ('timestamp', ASCENDING)],
                    [('timestamp', ASCENDING)]
                ]
            },
            'features': {
                'indexes': [
                    [('symbol', ASCENDING), ('timeframe', ASCENDING), ('timestamp', ASCENDING)],
                    [('timestamp', ASCENDING)]
                ]
            },
            'labels': {
                'indexes': [
                    [('symbol', ASCENDING), ('timeframe', ASCENDING), ('timestamp', ASCENDING)],
                    [('timestamp', ASCENDING)]
                ]
            },
            'predictions': {
                'indexes': [
                    [('symbol', ASCENDING), ('timeframe', ASCENDING), ('timestamp', ASCENDING)],
                    [('timestamp', ASCENDING)]
                ]
            },
            'trades': {
                'indexes': [
                    [('timestamp', ASCENDING)],
                    [('symbol', ASCENDING), ('timestamp', ASCENDING)]
                ]
            },
            'performance': {
                'indexes': [
                    [('timestamp', ASCENDING)],
                    [('strategy', ASCENDING), ('timestamp', ASCENDING)]
                ]
            }
        }
        
        for collection_name, config in collections_config.items():
            collection = self.db[collection_name]
            self._collections[collection_name] = collection
            
            # Create indexes
            for index_fields in config['indexes']:
                try:
                    collection.create_index(index_fields)
                    logger.info(f"Created index for {collection_name}: {index_fields}")
                except OperationFailure as e:
                    logger.warning(f"Failed to create index for {collection_name}: {e}")
    
    def store_ohlcv_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Store OHLCV data in MongoDB.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '30m', '1h', '4h')
            data: DataFrame with OHLCV data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prefer pre-mocked collection if provided by tests
            if 'ohlcv_data' in self._collections:
                collection = self._collections['ohlcv_data']
            else:
                # Ensure connection and collections are ready
                if not self._ensure_connected():
                    return False
                collection = self._collections['ohlcv_data']
            
            # Prepare data for storage
            records = []
            for idx, row in data.iterrows():
                # Prefer explicit 'timestamp' column if present; otherwise use index
                ts_value = row['timestamp'] if 'timestamp' in data.columns else idx
                record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': ts_value,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'created_at': datetime.now()
                }
                records.append(record)
            
            # Insert data
            result = collection.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} OHLCV records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {e}")
            return False
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data from MongoDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            if not self._ensure_connected():
                return pd.DataFrame()
            collection = self._collections['ohlcv_data']
            
            # Build query
            query = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.drop(['_id', 'symbol', 'timeframe', 'created_at'], axis=1, inplace=True)
            
            logger.info(f"Retrieved {len(df)} OHLCV records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve OHLCV data: {e}")
            return pd.DataFrame()
    
    def store_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Store OHLCV data (wrapper for store_ohlcv_data).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV DataFrame
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.store_ohlcv_data(symbol, timeframe, data)
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """
        Get OHLCV data (wrapper for get_ohlcv_data).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Maximum number of records
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        return self.get_ohlcv_data(symbol, timeframe, limit=limit)
    
    def store_features(self, symbol: str, timeframe: str, features: pd.DataFrame) -> bool:
        """
        Store technical features in MongoDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            features: DataFrame with technical features
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._ensure_connected():
                return False
            collection = self._collections['features']
            
            # Prepare data for storage
            records = []
            for timestamp, row in features.iterrows():
                record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'features': row.to_dict(),
                    'created_at': datetime.now()
                }
                records.append(record)
            
            # Insert data
            result = collection.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} feature records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def get_features(self, symbol: str, timeframe: str,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve technical features from MongoDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            pd.DataFrame: Technical features
        """
        try:
            if not self._ensure_connected():
                return pd.DataFrame()
            collection = self._collections['features']
            
            # Build query
            query = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                return pd.DataFrame()
            
            # Extract features
            features_data = []
            timestamps = []
            for record in data:
                timestamps.append(record['timestamp'])
                features_data.append(record['features'])
            
            df = pd.DataFrame(features_data, index=timestamps)
            
            logger.info(f"Retrieved {len(df)} feature records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return pd.DataFrame()
    
    def store_labels(self, symbol: str, timeframe: str, labels: pd.Series) -> bool:
        """
        Store market regime labels in MongoDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            labels: Series with market regime labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._ensure_connected():
                return False
            collection = self._collections['labels']
            
            # Prepare data for storage
            records = []
            for timestamp, label in labels.items():
                record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'label': int(label),
                    'created_at': datetime.now()
                }
                records.append(record)
            
            # Insert data
            result = collection.insert_many(records)
            logger.info(f"Stored {len(result.inserted_ids)} label records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store labels: {e}")
            return False
    
    def get_labels(self, symbol: str, timeframe: str,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.Series:
        """
        Retrieve market regime labels from MongoDB.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            pd.Series: Market regime labels
        """
        try:
            if not self._ensure_connected():
                return pd.Series(dtype=int)
            collection = self._collections['labels']
            
            # Build query
            query = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            
            # Convert to Series
            data = list(cursor)
            if not data:
                return pd.Series(dtype=int)
            
            labels = {}
            for record in data:
                labels[record['timestamp']] = record['label']
            
            series = pd.Series(labels)
            
            logger.info(f"Retrieved {len(series)} label records for {symbol} {timeframe}")
            return series
            
        except Exception as e:
            logger.error(f"Failed to retrieve labels: {e}")
            return pd.Series(dtype=int)
    
    def store_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Store trade record in MongoDB.
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._ensure_connected():
                return False
            collection = self._collections['trades']
            
            trade_data['created_at'] = datetime.now()
            result = collection.insert_one(trade_data)
            
            logger.info(f"Stored trade record: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
            return False
    
    def get_trades(self, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve trade records from MongoDB.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            pd.DataFrame: Trade records
        """
        try:
            if not self._ensure_connected():
                return pd.DataFrame()
            collection = self._collections['trades']
            
            # Build query
            query = {}
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            
            # Convert to DataFrame
            data = list(cursor)
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} trade records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return pd.DataFrame()
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about stored data.
        
        Returns:
            Dict[str, Any]: Data information
        """
        try:
            if not self._ensure_connected():
                return {}
            info = {}
            
            for collection_name, collection in self._collections.items():
                count = collection.count_documents({})
                info[collection_name] = {
                    'count': count,
                    'size_mb': collection.estimated_document_count() * 0.001  # Rough estimate
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get data info: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """
        Clean up old data to manage storage.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._ensure_connected():
                return False
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for collection_name, collection in self._collections.items():
                if collection_name in ['ohlcv_data', 'features', 'labels', 'predictions']:
                    result = collection.delete_many({
                        'timestamp': {'$lt': cutoff_date}
                    })
                    logger.info(f"Cleaned up {result.deleted_count} old records from {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
