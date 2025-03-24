import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)


# Get a key
value = r.get('categories')
print(value)
value = r.get('tags')
print(value)