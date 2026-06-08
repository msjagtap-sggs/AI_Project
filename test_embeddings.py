from app import get_vector_store

print('Calling get_vector_store with two small chunks...')
res = get_vector_store(['Hello world', 'This is a test'])
print('Result:', res)
