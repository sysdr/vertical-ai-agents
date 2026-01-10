"""Document processing and chunking utilities"""
from typing import List, Dict
import re
import io
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk text with overlap for context preservation.
        Returns list of chunks with position and overlap info.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Get main chunk
            chunk_text = text[start:end]
            
            # Get overlap for next chunk
            overlap_end = min(end + self.overlap, text_length)
            next_overlap = text[end:overlap_end] if end < text_length else ""
            
            chunks.append({
                'text': chunk_text,
                'position': start,
                'size': len(chunk_text),
                'next_overlap': next_overlap
            })
            
            # Move start position (chunk_size - overlap)
            start += (self.chunk_size - self.overlap)
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        return text.strip()
    
    def extract_metadata(self, filename: str, text: str) -> Dict:
        """Extract basic metadata from document"""
        return {
            'filename': filename,
            'size': len(text),
            'word_count': len(text.split()),
            'estimated_chunks': (len(text) // self.chunk_size) + 1
        }
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF file"""
        if not PDF_AVAILABLE:
            raise ValueError("PyPDF2 is not installed. Cannot process PDF files.")
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_parts.append(page.extract_text())
            
            return '\n\n'.join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
