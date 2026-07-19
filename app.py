from llama_index.core.schema import TextNode

from llama_index.core import MultiModalVectorStoreIndex
from llama_index.readers.file import PDFReaders, ImageReader
from llama_index.core.node_parser import HierarchicalNodeParser

#from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import geminiMultiModal
'''
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
'''

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#from llama_index.postprocessor.cohere_rerank import CohereRerank

client = QdrantClient(
    host = "localhost",
    port = 6333
)

vector_store = QdrantVectorStore(
    client = client,
  collection_name = "NewCollection"
)


GEMINI_API_KEY = "YOUR-API-KEY"
"""
documents = SimpleDirectoryReader(
    input_dir = "data/newfolder"
).load_data()"""


pdf_reader = PDFReader()
pdf_docs = pdf_reader.from_documents("data")

image_reader = ImageReader()
image_docs = image_reader.from_documents("data")

documents = pdf_docs + image_docs

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes= [2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents(documents)

node = TextNode(
    text = "Employees receive 20 annual leaves",
    metadata = {
        "department" : "HR",
        "year" : 2026,
        "source" : "HR_policy.pdf"
    }
)

node1 = TextNode(
    text = "Sleepy students will get extra homework",
    metadata = {
        "department" : "HR",
        "year" : 2026,
        "source" : "HR_policy.pdf"
    }
)

llm = GeminiMultiModal(model_name="gemini-embedding-2")

EmbeddingModel = HuggingFaceEmbedding(
    model_name = "BAAI/bge-base-en-v1.5"
)



#index = VectorStoreIndex([node, node1, nodes])

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    vector_Store = vector_store,
    embed_model = EmbeddingModel
)

query_engine = index.as_query_engine(llm = llm)

response = query_engine.query(
    "What does the MRI Suggest"
)

print(response)
