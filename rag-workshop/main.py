import ollama

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

# Name of the collection in the vector database (ie table name in traditional database)
COLLECTION_NAME = "example_text"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:4b"


def populate_database():

    client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.DOT),
    )
    
    datapoints = []
    
    for i, d in enumerate(documents):
        response = ollama.embed(model=EMBEDDING_MODEL, input=d)
        embedding = response["embeddings"][0]
        datapoints += [PointStruct(id=i, vector=embedding, payload={"text":d})]

    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=datapoints,
    )
    
    print("Inserted datapoints")

def run_query(query):
    
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )
    
    query_embedding = response["embeddings"]
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding[0],
        with_payload=True,
        limit=1
    ).points

    # print(search_result)
    
    response_context = ""
    
    for result in search_result:
        response_context += result.payload['text'] +" "
        # print(result)

    # print(response_context)

    output = ollama.generate(
        model=LLM_MODEL,
        prompt=f"Using this data: {response_context}. Respond to this prompt: {query}"
    )

    print(output['response'])


if __name__ == "__main__":
    populate_database()
    run_query("What animals are llamas related to?")
