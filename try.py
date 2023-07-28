import pinecone 
pinecone.init(api_key='YOUR_API_KEY', environment='us-east1-gcp') 
index = pinecone.Index('example-index') 

delete_response = index.delete(ids=['vec1', 'vec2'], namespace='example-namespace')
