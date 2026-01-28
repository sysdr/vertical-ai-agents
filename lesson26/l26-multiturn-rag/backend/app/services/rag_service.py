import logging
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables - try multiple possible locations
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # backend/.env
    Path.cwd() / "backend" / ".env",  # if running from project root
    Path.cwd() / ".env",  # if running from backend directory
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    # Fallback: try loading from current directory
    load_dotenv()

from app.services.conversation_memory import ConversationMemory
from app.services.ambiguity_detector import ambiguity_detector
from app.services.context_expander import context_expander

logger = logging.getLogger(__name__)

class RAGService:
    """Multi-turn RAG with conversation memory"""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.gemini_model = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize RAG components"""
        logger.info("Initializing RAG service...")
        
        # Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDB
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection("documents")
            logger.info(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection("documents")
            logger.info("Created new collection")
            self._load_sample_documents()
        
        self.initialized = True
        logger.info("RAG service initialized")
    
    def _load_sample_documents(self):
        """Load sample documents for testing"""
        sample_docs = [
            {
                "id": "doc1",
                "text": "Our Basic plan costs $29/month and includes 100 API calls, email support, and 1 user account. It's perfect for small projects and individual developers.",
                "metadata": {"type": "pricing", "plan": "basic"}
            },
            {
                "id": "doc2",
                "text": "The Enterprise plan is $299/month with unlimited API calls, 24/7 phone support, dedicated account manager, and unlimited users. Custom SLAs available.",
                "metadata": {"type": "pricing", "plan": "enterprise"}
            },
            {
                "id": "doc3",
                "text": "Support hours are Monday-Friday 9am-5pm EST for Basic and Pro plans. Enterprise customers get 24/7 support with 1-hour response time guarantee.",
                "metadata": {"type": "support"}
            },
            {
                "id": "doc4",
                "text": "API authentication uses Bearer tokens. Include your API key in the Authorization header: 'Authorization: Bearer YOUR_API_KEY'. Keys can be generated in the dashboard.",
                "metadata": {"type": "api", "topic": "authentication"}
            },
            {
                "id": "doc5",
                "text": "Rate limits are 100 requests/minute for Basic, 1000/min for Pro, and unlimited for Enterprise. Exceeded limits return 429 status code.",
                "metadata": {"type": "api", "topic": "limits"}
            }
        ]
        
        ids = [doc["id"] for doc in sample_docs]
        texts = [doc["text"] for doc in sample_docs]
        metadatas = [doc["metadata"] for doc in sample_docs]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Loaded {len(sample_docs)} sample documents")
    
    def health_check(self) -> bool:
        """Check RAG service health"""
        return self.initialized and self.collection is not None
    
    async def query(
        self,
        query: str,
        memory: ConversationMemory,
        top_k: int = 3
    ) -> Tuple[str, bool, float, List[str], List[Dict[str, Any]]]:
        """
        Process query with multi-turn context
        
        Returns: (response, is_ambiguous, ambiguity_score, clarifications, documents)
        """
        # Get conversation context
        context_string = memory.get_context_string(last_n=5)
        
        # Detect ambiguity
        is_ambiguous, ambiguity_score = ambiguity_detector.detect(
            query, context_string, retrieval_confidence=0.5
        )
        
        # If ambiguous, generate clarifications
        if is_ambiguous:
            clarifications = ambiguity_detector.generate_clarifications(
                query, context_string
            )
            return "", True, ambiguity_score, clarifications, []
        
        # Expand query with context
        expanded_query = context_expander.expand_query(query, context_string)
        
        # Retrieve documents
        documents = await self._retrieve(expanded_query, top_k)
        
        # Generate response
        response = await self._generate(
            original_query=query,
            expanded_query=expanded_query,
            documents=documents,
            conversation_context=context_string
        )
        
        return response, False, ambiguity_score, [], documents
    
    async def _retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    "id": results['ids'][0][i],
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        logger.info(f"Retrieved {len(documents)} documents for query: {query}")
        return documents
    
    async def _generate(
        self,
        original_query: str,
        expanded_query: str,
        documents: List[Dict[str, Any]],
        conversation_context: str
    ) -> str:
        """Generate response using retrieved context"""
        # Format retrieved context
        retrieved_context = "\n\n".join([
            f"[Document {i+1}]\n{doc['text']}"
            for i, doc in enumerate(documents)
        ])
        
        # Build prompt (extends L25 prompting strategies)
        prompt = f"""You are a helpful assistant answering questions using provided context.

CONVERSATION HISTORY:
{conversation_context if conversation_context else "(No prior conversation)"}

RETRIEVED CONTEXT:
{retrieved_context if retrieved_context else "(No relevant documents found)"}

USER QUESTION: {original_query}

INSTRUCTIONS:
1. Answer using ONLY information from the retrieved context
2. If context doesn't contain the answer, say so clearly
3. Reference specific details from the context
4. Maintain conversation continuity - acknowledge previous exchanges
5. Be concise but complete

ANSWER:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            logger.info(f"Generated response ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Response generation failed: {e}")
            
            # Check for API key issues
            if "API key" in error_msg.lower() or "API_KEY" in error_msg or "expired" in error_msg.lower():
                logger.error("Gemini API key is invalid or expired. Please update GEMINI_API_KEY in backend/.env")
                # Fallback: return a simple summary based on retrieved documents
                if documents:
                    doc_summaries = "\n".join([
                        f"- {doc['text'][:200]}..." if len(doc['text']) > 200 else f"- {doc['text']}"
                        for doc in documents[:3]
                    ])
                    return f"Based on the available information:\n\n{doc_summaries}\n\nNote: API key needs to be updated for full AI-generated responses."
                else:
                    return "I found some information, but the API key needs to be updated for full AI-generated responses. Please check backend/.env and update GEMINI_API_KEY."
            
            return "I apologize, but I encountered an error generating a response. Please try rephrasing your question."

rag_service = RAGService()
