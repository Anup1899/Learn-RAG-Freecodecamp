from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    Language,
)


# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()


SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(SAMPLE_TEXT)
    
    print(f"Original length: {len(SAMPLE_TEXT)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"\nFirst chunk:\n{chunks[0]}...")



def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat to increase length

    # Without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    chunks_no_overlap = no_overlap.split_text(text)


    # With overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    chunks_with_overlap = with_overlap.split_text(text)

    print("Without overlap:")
    print(f"  Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_no_overlap[1][:20]}...")    

    print("\nWith overlap:")
    print(f"  Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_with_overlap[1][:20]}...")    


if __name__ == "__main__":
    # print("Testing RecursiveCharacterTextSplitter...")
    # recursive_splitter()
    print("\nTesting overlap importance...")
    overlap_importance()