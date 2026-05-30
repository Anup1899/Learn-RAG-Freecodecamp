# Type of Chunking in RAG
# 1. Fixed-size chunking -- The document is split into chunks of a fixed size, such as a certain number of characters or tokens. This approach is simple and easy to implement, but it may not always result in natural or meaningful chunks, especially if the document has a complex structure or contains long sentences.

# 2. Recursive chunking (Default) -- The document is recursively split into smaller chunks until the desired chunk size is achieved. This approach allows for more natural chunking based on the structure of the document, such as paragraphs or sections, rather than relying on fixed sizes.

# 3. Semantic chunking (Best Quality - when accuracy matters more than speed) -- The document is chunked based on the semantic meaning of the content, such as topics or themes. This approach can be more effective for certain types of documents, such as news articles or research papers, where the content may be organized around specific topics or themes.

# 4. Late chunking -- Embeddings are generated for the entire document, and then the document is chunked based on the similarity of the embeddings. This approach allows for more flexible and context-aware chunking, as it can adapt to the content of the document rather than relying on fixed sizes or recursive splitting.


# Chunking decision making
# Content type : General docs
# Strategy : Recursive chunking (Default)
# Chunk size : 500-1000 tokens (Default)

# Content type : Techinal
# Strategy : Semantic chunking (Best Quality - when accuracy matters more than speed)
# Chunk size : Auto (Based on the semantic meaning of the content)

# Content type : Code
# Strategy : Code-Splitter
# Chunk size : Auto (Based on the structure of the code, such as functions or classes)

# Content type : Markdown
# Strategy : Markdown-Splitter
# Chunk size : Auto (Based on the structure of the markdown, such as headings or sections)


# Embedding

# ChatModels vs Embedding Models
# 1. Chat Models are designed to generate human-like responses to prompts
# 2. Embedding models are designed to generate vector representations of text. List of numbers which represent the meaning of the text in a high-dimensional space.


# Embedding Dimensions
# 1. text-embedding-3-small has 1536 dimensions
# 2. text-embedding-3-large has 3072 dimensions
# 3. Gemini Embedding has 768 dimensions
# 4. BGE-Small has 384 dimensions

# More dimensions a vector has, the more information it can capture about the text, but it also requires more computational resources to process and store. The choice of embedding dimensions depends on the specific use case and the trade-off between accuracy and efficiency.
# For example, if you are working with a large corpus of text and need to generate embeddings for a large number of documents, you may want to choose a smaller embedding dimension to reduce the computational resources required. On the other hand, if you are working with a smaller corpus of text and need to capture more nuanced information about the meaning of the text, you may want to choose a larger embedding dimension.


# RAG Pipeline

# INDEXING Phase 1: Document Loading -> Chunking -> Embedding -> Store in vector database
# QUERYING Phase 2: Embedding Query -> Seach in vector database -> Retrieval -> Generate response with LLM (Augment) -> Answer

# Both phase we should be using the same embedding model to ensure that the embeddings generated for the documents and the query are in the same vector space, which allows for accurate similarity comparisons and retrieval of relevant documents based on the query.


# Three Rules for Production-Ready RAG Systems
# 1. Same embedding model for both indexing and querying to ensure consistency in the vector space and improve retrieval accuracy.
# 2. Embedding quality > Quantity  - Prioritize high-quality relavant vectors over a large number of low-quality vectors to improve retrieval performance and reduce noise in the results.
# 3. Test retrieval separately - Validate retrieval performance independent of generation


# Vector Database workflow
# 1. Query is asked in the AP
# 2. Query is embedded using the same embedding model used for indexing
# 3. The embedded query is sent to the vector database to search for similar vectors
# 4. The vector database returns the most similar vectors along with their associated metadata (e.g., document ID, chunk ID, etc.)
# 5. The retrieved vectors are then used to generate a response using the LLM, which can be augmented with the retrieved information to provide a more accurate and relevant answer to the user's query.


from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Document,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "freecodecamp_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# connect to Qdrant Cloud
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, cloud_inference=True)

# --- INDEXING PHASE ---
# recreate collection to ensure correct vector config
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

# create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# create payload index on category to enable filtered search
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)


menu_items = [
    (
        "Pad Thai with Tofu",
        "Stir-fried rice noodles with tofu bean sprouts scallions and crushed peanuts in traditional tamarind sauce",
        "$13.95",
        "Noodles",
    ),
    (
        "Grilled Salmon Fillet",
        "Wild-caught Atlantic salmon grilled with lemon butter and fresh herbs served with seasonal vegetables",
        "$24.50",
        "Seafood Entrees",
    ),
    (
        "Mushroom Risotto",
        "Creamy arborio rice with mixed mushrooms parmesan truffle oil and fresh thyme",
        "$16.75",
        "Vegetarian",
    ),
    (
        "Bibimbap Bowl",
        "Korean rice bowl with seasoned vegetables fried egg gochujang sauce and choice of protein",
        "$14.50",
        "Korean Bowls",
    ),
    (
        "Falafel Wrap",
        "Crispy chickpea fritters with hummus tahini cucumber tomato and pickled vegetables in warm pita",
        "$11.25",
        "Mediterranean",
    ),
    (
        "Shrimp Tacos",
        "Three soft tacos with grilled shrimp cabbage slaw chipotle aioli and fresh lime",
        "$13.00",
        "Tacos",
    ),
    (
        "Vegetable Curry",
        "Mixed vegetables in aromatic coconut curry sauce with jasmine rice and naan bread",
        "$12.95",
        "Indian Curries",
    ),
    (
        "Tuna Poke Bowl",
        "Fresh ahi tuna with avocado edamame cucumber seaweed salad over sushi rice with spicy mayo",
        "$16.50",
        "Poke Bowls",
    ),
    (
        "Margherita Pizza",
        "Fresh mozzarella san marzano tomatoes basil and extra virgin olive oil on wood-fired crust",
        "$14.00",
        "Pizza",
    ),
    (
        "Chicken Tikka Masala",
        "Tandoori chicken in creamy tomato sauce with aromatic spices served with basmati rice",
        "$15.95",
        "Indian Entrees",
    ),
    (
        "Greek Salad",
        "Romaine lettuce tomatoes cucumbers kalamata olives feta cheese red onion with lemon oregano dressing",
        "$10.50",
        "Salads",
    ),
    (
        "Lobster Roll",
        "Fresh Maine lobster meat with light mayo on toasted buttery roll served with chips",
        "$22.00",
        "Seafood Sandwiches",
    ),
    (
        "Quinoa Buddha Bowl",
        "Organic quinoa with roasted chickpeas kale sweet potato tahini dressing and hemp seeds",
        "$13.50",
        "Healthy Bowls",
    ),
    (
        "Beef Pho",
        "Traditional Vietnamese beef noodle soup with rice noodles fresh herbs bean sprouts and lime",
        "$12.75",
        "Noodle Soups",
    ),
    (
        "Eggplant Parmesan",
        "Breaded eggplant layered with marinara mozzarella and parmesan served with pasta",
        "$15.25",
        "Italian Entrees",
    ),
    (
        "Crab Cakes",
        "Maryland-style lump crab cakes with remoulade sauce and mixed greens",
        "$18.50",
        "Seafood Appetizers",
    ),
    (
        "Tofu Stir Fry",
        "Crispy tofu with broccoli bell peppers snap peas in garlic ginger sauce over steamed rice",
        "$12.50",
        "Vegetarian Entrees",
    ),
    (
        "Salmon Sushi Platter",
        "12 pieces of fresh salmon nigiri and sashimi with wasabi pickled ginger and soy sauce",
        "$19.95",
        "Sushi",
    ),
    (
        "Caprese Sandwich",
        "Fresh mozzarella tomatoes basil pesto balsamic glaze on ciabatta bread",
        "$11.75",
        "Sandwiches",
    ),
    (
        "Tom Yum Soup",
        "Spicy and sour Thai soup with shrimp lemongrass galangal mushrooms and kaffir lime leaves",
        "$11.50",
        "Soups",
    ),
    (
        "Lentil Dal",
        "Red lentils simmered with turmeric cumin coriander served with rice and naan",
        "$11.95",
        "Vegan Entrees",
    ),
    (
        "Fish and Chips",
        "Beer-battered cod with crispy fries malt vinegar and tartar sauce",
        "$16.00",
        "British Classics",
    ),
    (
        "Veggie Burger",
        "House-made black bean and quinoa patty with avocado sprouts tomato on brioche bun",
        "$13.25",
        "Burgers",
    ),
    (
        "Miso Ramen",
        "Rich miso broth with ramen noodles soft-boiled egg bamboo shoots nori and scallions",
        "$14.50",
        "Ramen",
    ),
    (
        "Stuffed Bell Peppers",
        "Roasted bell peppers filled with rice vegetables herbs and melted cheese",
        "$13.75",
        "Vegetarian Entrees",
    ),
    (
        "Scallop Risotto",
        "Pan-seared sea scallops over creamy parmesan risotto with white wine and lemon",
        "$26.50",
        "Seafood Specials",
    ),
    (
        "Spring Rolls",
        "Fresh rice paper rolls with vegetables tofu rice noodles herbs and peanut dipping sauce",
        "$8.95",
        "Appetizers",
    ),
    (
        "Oyster Po Boy",
        "Fried oysters with lettuce tomato pickles and remoulade on french bread",
        "$15.50",
        "Sandwiches",
    ),
    (
        "Portobello Mushroom Steak",
        "Grilled portobello cap marinated in balsamic with roasted vegetables and quinoa",
        "$14.95",
        "Vegan Entrees",
    ),
    (
        "Coconut Shrimp",
        "Jumbo shrimp breaded in shredded coconut served with sweet chili sauce",
        "$14.25",
        "Seafood Appetizers",
    ),
]

# points generator
points = []
for i, menu_item in enumerate(menu_items):
    point = PointStruct(
        id=i,
        vector=Document(
            text=f"{menu_item[0]} {menu_item[1]}",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        payload={
            "item_name": menu_item[0],
            "description": menu_item[1],
            "price": menu_item[2],
            "category": menu_item[3],
        },
    )
    points.append(point)

# upsert points to collection
client.upsert(
    collection_name="freecodecamp_collection",
    points=points,
)

# --- QUERYING PHASE ---

# --- Search 1: Basic similarity search ---
query_text = "Pad Thai with Tofu"

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=Document(text=query_text, model=EMBEDDING_MODEL),
    with_payload=True,
    limit=5,
    # score_threshold=0.3,
)

print(f"\n=== Basic Similarity Search: '{query_text}' ===")
for result in results.points:
    print(f"Item: {result.payload.get('item_name', 'N/A')}")
    print(f"Score: {result.score:.4f}")
    print(f"Description: {result.payload['description'][:150]}...")
    print(f"Price: {result.payload.get('price', 'N/A')}")
    print("---")

# --- Search 2: Filtered similarity search (by category) ---
query_text = "Pad Thai with Tofu"
category_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="Noodles"),
        )
    ]
)

filtered_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=Document(text=query_text, model=EMBEDDING_MODEL),
    query_filter=category_filter,
    with_payload=True,
    limit=5,
    # score_threshold=0.7,
)

print(f"\n=== Filtered Similarity Search: '{query_text}' (Seafood Appetizers only) ===")
for result in filtered_results.points:
    print(f"Item: {result.payload.get('item_name', 'N/A')}")
    print(f"Score: {result.score:.4f}")
    print(f"Category: {result.payload.get('category', 'N/A')}")
    print(f"Description: {result.payload['description'][:150]}...")
    print(f"Price: {result.payload.get('price', 'N/A')}")
    print("---")
