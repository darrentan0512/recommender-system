import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import redis
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import jsonpickle


class RealTimeRecommender:
    def __init__(self, redis_host='redis', redis_port=6379, similarity_threshold=0.3):
        """
        Initialize the real-time recommender system.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            similarity_threshold: Minimum similarity score to consider an item as a recommendation
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.similarity_threshold = similarity_threshold
        self.user_vectors = {}
        self.item_vectors = {}
        
    def track_event(self, user_id, item_id, event_type, timestamp=None):
        """
        Track a user interaction event in real-time.
        
        Args:
            user_id: Unique identifier for the user
            item_id: Unique identifier for the item
            event_type: Type of interaction (view, like, purchase, etc.)
            timestamp: Event timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Define weights for different event types
        event_weights = {
            'view': 1.0,
            'click': 2.0,
            'add_to_cart': 3.0,
            'purchase': 5.0,
            'like': 2.5,
            'rate': 3.0
        }
        
        weight = event_weights.get(event_type, 1.0)
        
        # Store the event in Redis
        event_data = {
            'user_id': user_id,
            'item_id': item_id,
            'event_type': event_type,
            'timestamp': timestamp,
            'weight': weight
        }
        
        # Add to time-series data
        self.redis_client.zadd(f"user:{user_id}:events", {json.dumps(event_data): timestamp})
        self.redis_client.zadd(f"item:{item_id}:events", {json.dumps(event_data): timestamp})
        
        # Update user-item interaction matrix
        self._update_user_item_matrix(user_id, item_id, weight)
        
        # Update real-time features
        self._update_real_time_features(user_id, item_id, event_type, weight)
        
    def _update_user_item_matrix(self, user_id, item_id, weight):
        """Update the user-item interaction matrix with the new event."""
        # Get current score if exists
        current_score = float(self.redis_client.hget(f"user:{user_id}:scores", item_id) or 0)
        
        # Apply time decay to old score (optional)
        decay_factor = 0.95
        new_score = current_score * decay_factor + weight
        
        # Store updated score
        self.redis_client.hset(f"user:{user_id}:scores", item_id, str(new_score))
        self.redis_client.hset(f"item:{item_id}:users", user_id, str(new_score))
        
    def _update_real_time_features(self, user_id, item_id, event_type, weight):
        """Update real-time features for the user and item."""
        # Update item popularity
        self.redis_client.zincrby("item:popularity", weight, item_id)
        
        # Update item in user's recent activity
        self.redis_client.zadd(f"user:{user_id}:recent", {item_id: datetime.now().timestamp()})
        
        # Keep only recent items (last 100)
        self.redis_client.zremrangebyrank(f"user:{user_id}:recent", 0, -101)
        
        # Update item category affinity if applicable
        item_categories = self.redis_client.smembers(f"item:{item_id}:categories")
        for category in item_categories:
            self.redis_client.zincrby(f"user:{user_id}:categories", weight, category)
    
    def get_user_recommendations(self, user_id, n=10, strategy="hybrid"):
        """
        Get real-time recommendations for a user.
        
        Args:
            user_id: User ID to get recommendations for
            n: Number of recommendations to return
            strategy: Recommendation strategy (collaborative, content, hybrid)
            
        Returns:
            List of recommended item IDs
        """
        if strategy == "collaborative":
            return self._get_collaborative_recommendations(user_id, n)
        elif strategy == "content":
            return self._get_content_recommendations(user_id, n)
        else:  # hybrid
            collab_recs = self._get_collaborative_recommendations(user_id, n)
            content_recs = self._get_content_recommendations(user_id, n)
            
            # Combine and deduplicate recommendations
            hybrid_recs = []
            seen = set()
            
            # Interleave recommendations from both sources
            for i in range(max(len(collab_recs), len(content_recs))):
                if i < len(collab_recs) and collab_recs[i] not in seen:
                    hybrid_recs.append(collab_recs[i])
                    seen.add(collab_recs[i])
                if i < len(content_recs) and content_recs[i] not in seen:
                    hybrid_recs.append(content_recs[i])
                    seen.add(content_recs[i])
                    
            return hybrid_recs[:n]
    
    def _get_collaborative_recommendations(self, user_id, n=10):
        """Get collaborative filtering recommendations."""
        # Get items the user has interacted with
        user_items = self.redis_client.hgetall(f"user:{user_id}:scores")
        if not user_items:
            return self._get_popularity_recommendations(n)
            
        # Find similar users
        similar_users = self._find_similar_users(user_id)
        
        # Get items from similar users
        candidate_items = defaultdict(float)
        for sim_user, sim_score in similar_users:
            if sim_user == user_id:
                continue
                
            sim_user_items = self.redis_client.hgetall(f"user:{sim_user}:scores")
            print("sim user items",sim_user_items)
            for item_id, score in sim_user_items.items():
                if item_id not in user_items:  # Don't recommend items the user already interacted with
                    candidate_items[item_id] += float(score) * sim_score
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_candidates[:n]]
    
    def _get_content_recommendations(self, user_id, n=10):
        """Get content-based recommendations."""
        # Get user's category preferences
        user_categories = self.redis_client.zrange(f"user:{user_id}:categories", 0, -1, withscores=True)
        if not user_categories:
            return self._get_popularity_recommendations(n)
            
        # Get recent items for the user
        recent_items = self.redis_client.zrange(f"user:{user_id}:recent", 0, -1)
        
        # Find items with similar categories
        candidate_items = defaultdict(float)
        for category, weight in user_categories:
            category_items = self.redis_client.smembers(f"category:{category}:items")
            for item_id in category_items:
                if item_id not in recent_items:
                    candidate_items[item_id] += float(weight)
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_candidates[:n]]
    
    def _get_popularity_recommendations(self, n=10):
        """Get popularity-based recommendations."""
        # Return the most popular items
        popular_items = self.redis_client.zrange("item:popularity", 0, n-1, desc=True)
        return popular_items
    
    def _find_similar_users(self, user_id):
        """Find users similar to the given user."""
        # Get user's item interactions
        user_items = self.redis_client.hgetall(f"user:{user_id}:scores")
        
        # Find users who have interacted with the same items
        candidate_users = set()
        for item_id in user_items:
            item_users = self.redis_client.hkeys(f"item:{item_id}:users")
            candidate_users.update(item_users)
        
        # Calculate similarity scores
        user_similarities = []
        for candidate_user in candidate_users:
            if candidate_user == user_id:
                continue
                
            candidate_items = self.redis_client.hgetall(f"user:{candidate_user}:scores")
            
            # Calculate Jaccard similarity
            common_items = set(user_items.keys()) & set(candidate_items.keys())
            if not common_items:
                continue
                
            similarity = len(common_items) / len(set(user_items.keys()) | set(candidate_items.keys()))
            
            if similarity >= self.similarity_threshold:
                user_similarities.append((candidate_user, similarity))
        
        # Sort by similarity score
        return sorted(user_similarities, key=lambda x: x[1], reverse=True)
    
    def update_item_metadata(self, item_id, metadata):
        """
        Update metadata for an item.
        
        Args:
            item_id: Item ID
            metadata: Dictionary of metadata (categories, tags, etc.)
        """

        # Serialize metadata
        serialized_metadata = {}
    
        for key, value in metadata.items():
            # Serialize any complex data type
            serialized_metadata[key] = jsonpickle.encode(value)

        # Store item metadata
        self.redis_client.hset(f"item:{item_id}:metadata", mapping=serialized_metadata)
        
        # Update category indices
        if 'categories' in metadata:
            for category in metadata['categories']:
                self.redis_client.sadd(f"category:{category}:items", item_id)
                self.redis_client.sadd(f"item:{item_id}:categories", category)
    
    def batch_update_model(self):
        """
        Run a batch update to refresh the model.
        This would typically be run periodically to update the model with new data.
        """
        # This is where you'd implement offline model training
        # For example, matrix factorization, item2vec, etc.
        print("Batch update started...")
        
        # Here you could export Redis data to train a more complex model
        # Then update the model parameters back in Redis
        
        print("Batch update completed.")

# Example usage
if __name__ == "__main__":
    # Initialize the recommender
    recommender = RealTimeRecommender()
    
    # Add some item metadata
    recommender.update_item_metadata("item1", {
        "categories": ["electronics", "computers"],
        "tags": ["laptop", "gaming", "high-performance"]
    })
    
    recommender.update_item_metadata("item2", {
        "categories": ["electronics", "accessories"],
        "tags": ["keyboard", "mechanical", "gaming"]
    })
    
    # Track some events
    recommender.track_event("user1", "item1", "view")
    recommender.track_event("user1", "item2", "view")
    recommender.track_event("user1", "item1", "add_to_cart")
    
    # Get recommendations
    recs = recommender.get_user_recommendations("user1", n=5)
    print(f"Recommendations for user1: {recs}")
