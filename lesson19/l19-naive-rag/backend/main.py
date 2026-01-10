"""FastAPI backend for Naïve RAG system"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import aiofiles
import os
from urllib.parse import unquote

from backend.knowledge_base import KnowledgeBase
from backend.document_processor import DocumentProcessor
from backend.query_engine import QueryEngine

app = FastAPI(title="Naïve RAG System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
kb = KnowledgeBase()
processor = DocumentProcessor(chunk_size=500, overlap=50)
query_engine = QueryEngine(kb, api_key=os.getenv('GEMINI_API_KEY'))

# Request models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    chunks_used: List[Dict]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "naive-rag"}

def is_binary_content(content: bytes) -> bool:
    """Detect if content appears to be binary (non-text)"""
    if len(content) == 0:
        return False
    
    # CRITICAL: Check for PDF magic bytes FIRST (most reliable)
    # PDFs MUST start with %PDF, but check more thoroughly
    if len(content) >= 4:
        if content.startswith(b'%PDF'):
            return True
        # Also check first 1024 bytes for PDF header
        # Some PDFs have leading whitespace or metadata
        if len(content) > 1024:
            if b'%PDF' in content[:1024]:
                return True
        else:
            if b'%PDF' in content:
                return True
    
    # Check for null bytes (common in binary files, NEVER in text)
    if b'\x00' in content[:8192]:  # Check first 8KB
        return True
    
    # Check for high percentage of non-printable/non-ASCII bytes
    # This catches binary files even if they have some text
    sample = content[:8192] if len(content) > 8192 else content
    if len(sample) == 0:
        return False
    
    # Count printable ASCII and common whitespace
    text_chars = sum(1 for byte in sample if 32 <= byte < 127 or byte in (9, 10, 13))
    text_ratio = text_chars / len(sample)
    
    # If less than 70% printable ASCII, it's binary
    if text_ratio < 0.7:
        return True
    
    # ADDITIONAL CHECK: Look for binary indicators even in "text-like" content
    # Check for high byte values (>127) that aren't valid UTF-8 sequences
    # This catches binary data that might pass the ratio test
    high_bytes = sum(1 for byte in sample if byte > 127)
    if high_bytes > 0 and len(sample) > 100:
        # If there are high bytes, try to validate they form valid UTF-8
        # If UTF-8 decode fails, it's likely binary
        try:
            import sys
            import traceback
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"DECODE CALL #1 at main.py:91", file=sys.stderr)
            print(f"  Data type: {type(sample).__name__}, length: {len(sample)}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr, limit=5)
            result = sample.decode('utf-8', errors='strict')
            print(f"  Decode SUCCESS", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
        except (UnicodeDecodeError, UnicodeError) as e:
            print(f"  Decode FAILED: {e}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            # Decode failed = likely binary content
            return True
    
    return False

def is_pdf_file(content: bytes, filename: str = None, content_type: str = None) -> bool:
    """Detect if file is a PDF using multiple methods"""
    # Check PDF magic bytes (PDF files start with %PDF)
    if len(content) >= 4:
        if content.startswith(b'%PDF'):
            return True
        # Also check first 1024 bytes
        if len(content) > 1024 and b'%PDF' in content[:1024]:
            return True
    
    # Check file extension
    if filename:
        filename_lower = filename.lower().strip()
        if filename_lower.endswith('.pdf'):
            return True
    
    # Check content-type header
    if content_type:
        content_type_lower = content_type.lower().strip()
        if 'application/pdf' in content_type_lower:
            return True
    
    return False

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    text = None  # Initialize text variable
    
    try:
        # Read file content (DO NOT decode - keep as bytes)
        import sys
        print(f"DEBUG ENTRY: file type: {type(file)}, filename: {file.filename}", file=sys.stderr)
        content = await file.read()
        print(f"DEBUG AFTER READ: content type: {type(content)}, length: {len(content) if content else 0}", file=sys.stderr)
        if content and len(content) > 0:
            print(f"DEBUG: First 10 bytes: {content[:10]}, starts with %PDF: {content.startswith(b'%PDF')}", file=sys.stderr)
        
        # CRITICAL: Check if content is empty
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty. Please upload a valid file."
            )
        
        # Ensure we have valid filename - handle URL encoding and normalize
        raw_filename = file.filename if file.filename else "unknown_file"
        # Decode URL encoding if present (handles %20 for spaces, etc.)
        try:
            filename = unquote(raw_filename)
        except Exception:
            filename = raw_filename
        # Normalize filename (handle spaces, case, etc.)
        filename = filename.strip()
        
        content_type = file.content_type if file.content_type else ""
        
        # CRITICAL SAFETY CHECK: Before any processing, check if this looks like binary/PDF
        # This prevents any accidental UTF-8 decoding of binary files
        appears_binary = is_binary_content(content)
        filename_lower = filename.lower().strip()
        
        # STEP 1: Check PDF magic bytes FIRST (most reliable method)
        # PDF files MUST start with %PDF according to PDF specification
        # This is the MOST IMPORTANT check - if magic bytes say PDF, it's a PDF!
        # Also check for PDF anywhere in first 1024 bytes (some PDFs have leading whitespace)
        is_pdf_by_magic_bytes = False
        if len(content) >= 4:
            # Check start of file
            if content.startswith(b'%PDF'):
                is_pdf_by_magic_bytes = True
            # Also check first 1024 bytes for PDF header (handles files with leading bytes)
            elif len(content) > 1024:
                is_pdf_by_magic_bytes = b'%PDF' in content[:1024]
            else:
                is_pdf_by_magic_bytes = b'%PDF' in content
        
        # STEP 2: Check filename extension (handle spaces, case, URL encoding)
        # Check if filename ends with .pdf (robust check)
        filename_suggests_pdf = filename_lower.endswith('.pdf') or filename_lower.rstrip().endswith('.pdf')
        
        # STEP 3: Check content-type header
        content_type_lower = content_type.lower().strip() if content_type else ""
        content_type_suggests_pdf = 'application/pdf' in content_type_lower or content_type_lower == 'pdf'
        
        # CRITICAL DECISION: If ANY indicator says it's a PDF, treat it as PDF
        # Priority: Magic bytes > Filename > Content-Type
        # NEVER decode as UTF-8 if ANY of these are true!
        is_pdf = is_pdf_by_magic_bytes or filename_suggests_pdf or content_type_suggests_pdf
        
        if is_pdf:
            # This is definitely a PDF - NEVER try to decode as text
            try:
                import sys
                print(f"DEBUG: Entered PDF path at line 183", file=sys.stderr)
                print(f"DEBUG: Content type: {type(content)}, Length: {len(content)}", file=sys.stderr)
                # Check if PDF processing is available
                if not hasattr(processor, 'extract_text_from_pdf'):
                    raise HTTPException(
                        status_code=500,
                        detail="PDF processing is not available. PyPDF2 library is not installed."
                    )
                print(f"DEBUG: About to call extract_text_from_pdf at line 192", file=sys.stderr)
                text = processor.extract_text_from_pdf(content)
                print(f"DEBUG: extract_text_from_pdf returned, type: {type(text)}, length: {len(text) if text else 0}", file=sys.stderr)
                if not text or not text.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="PDF appears to be empty or contains no extractable text."
                    )
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")
            except ImportError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF processing library not available: {str(e)}. Please install PyPDF2."
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process PDF file: {str(e)}. Please ensure the PDF is not corrupted or password-protected."
                )
        
        # STEP 4: Only process as text if filename explicitly indicates text file AND it's NOT a PDF
        elif filename_lower.endswith(('.txt', '.md', '.mdx')):
            # CRITICAL: Check if it's actually binary BEFORE any decode attempt
            if is_binary_content(content):
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' has a text extension but appears to be binary (PDF or other binary format). If this is a PDF, please rename it with .pdf extension or ensure it's a valid text file."
                )
            
            # CRITICAL: Double-check PDF magic bytes even for text extensions
            # Some PDFs might be renamed with .txt extension
            if len(content) >= 4 and (content.startswith(b'%PDF') or (len(content) > 1024 and b'%PDF' in content[:1024])):
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' has a text extension but is actually a PDF file. Please rename it with .pdf extension."
                )
            
            # Explicit text file - safe to decode, but validate encoding first
            # CRITICAL: Try to decode a larger sample first to catch binary content
            # The binary check only samples first 8KB, but we need to validate more
            sample_size = min(16384, len(content))  # Check first 16KB
            if sample_size > 0:
                try:
                    import sys
                    import traceback
                    print(f"\n{'='*80}", file=sys.stderr)
                    print(f"DECODE CALL #2 at main.py:246", file=sys.stderr)
                    print(f"  Data type: {type(content[:sample_size]).__name__}, length: {len(content[:sample_size])}", file=sys.stderr)
                    traceback.print_stack(file=sys.stderr, limit=5)
                    result = content[:sample_size].decode('utf-8', errors='strict')
                    print(f"  Decode SUCCESS", file=sys.stderr)
                    print(f"{'='*80}\n", file=sys.stderr)
                except UnicodeDecodeError as sample_err:
                    import sys
                    print(f"  Decode FAILED: {sample_err}", file=sys.stderr)
                    print(f"{'='*80}\n", file=sys.stderr)
                    # Even the sample can't be decoded - definitely binary
                    import sys
                    print(f"DEBUG: Sample decode failed at line 236: {sample_err}", file=sys.stderr)
                    print(f"DEBUG: Content type: {type(content)}, Length: {len(content)}", file=sys.stderr)
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{filename}' appears to be binary data, not text. Please upload a valid text file or PDF."
                    )
            
            # Try UTF-8 decode on full content
            # CRITICAL: We already validated the sample can decode, so this should be safe
            # But add explicit type check as final safeguard
            if not isinstance(content, bytes):
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: content is not bytes (type: {type(content)}). This should never happen."
                )
            
            # FINAL SAFEGUARD: Validate the ENTIRE content can be decoded before attempting
            # This prevents the error from occurring at all
            try:
                import sys
                import traceback
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"DECODE CALL #3 at main.py:272", file=sys.stderr)
                print(f"  Data type: {type(content).__name__}, length: {len(content)}", file=sys.stderr)
                traceback.print_stack(file=sys.stderr, limit=5)
                result = content.decode('utf-8', errors='strict')
                print(f"  Decode SUCCESS", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
            except UnicodeDecodeError as e:
                import sys
                print(f"  Decode FAILED: {e}", file=sys.stderr)
                print(f"  Error at position: {e.start}", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                # Full content decode failed - this is binary, not text
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{filename}' cannot be decoded as UTF-8 text. It appears to be binary data. Please upload a valid text file or PDF."
                )
            
            # If we get here, the full content is valid UTF-8
            try:
                import sys
                import traceback
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"DECODE CALL #4 at main.py:282", file=sys.stderr)
                print(f"  Data type: {type(content).__name__}, length: {len(content)}", file=sys.stderr)
                traceback.print_stack(file=sys.stderr, limit=5)
                text = content.decode('utf-8', errors='strict')
                print(f"  Decode SUCCESS", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
            except UnicodeDecodeError as decode_error:
                # UTF-8 decode failed - this is definitely binary or wrong encoding
                import sys
                import traceback
                print(f"DEBUG: Decode error at line 247: {decode_error}", file=sys.stderr)
                print(f"DEBUG: Content type: {type(content)}, Length: {len(content)}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # At this point we know it's not valid UTF-8
                # Check if it might be a PDF that slipped through
                if len(content) >= 4 and (content.startswith(b'%PDF') or (len(content) > 1024 and b'%PDF' in content[:1024])):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{filename}' is actually a PDF file, not text. Please rename it with .pdf extension."
                    )
                # Check binary content again
                if is_binary_content(content):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{filename}' appears to be binary, not text. Error: {str(decode_error)}. Please upload a valid text file or PDF."
                    )
                # If not binary, try latin-1 as fallback (can decode any byte)
                # But only if it truly looks like text
                try:
                    import sys
                    import traceback
                    print(f"\n{'='*80}", file=sys.stderr)
                    print(f"DECODE CALL #5 at main.py:306 (latin-1 fallback)", file=sys.stderr)
                    print(f"  Data type: {type(content).__name__}, length: {len(content)}", file=sys.stderr)
                    traceback.print_stack(file=sys.stderr, limit=5)
                    text = content.decode('latin-1')
                    print(f"  Decode SUCCESS", file=sys.stderr)
                    print(f"{'='*80}\n", file=sys.stderr)
                    # Validate the decoded text doesn't contain too many control chars
                    # If it does, it's probably binary
                    control_chars = sum(1 for c in text[:1000] if ord(c) < 32 and c not in '\n\r\t')
                    if control_chars > len(text[:1000]) * 0.1:  # More than 10% control chars
                        raise HTTPException(
                            status_code=400,
                            detail=f"File '{filename}' appears to be binary data, not text. Please upload a valid text file or PDF."
                        )
                except HTTPException:
                    raise
                except Exception as decode_error2:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not decode text file '{filename}'. Error: {str(decode_error2)}. The file may be binary or corrupted."
                    )
        
        # STEP 5: Unknown file type - reject to avoid errors
        else:
            # Don't guess - ask user for correct file type
            # But first check if it might be a PDF that wasn't detected
            if is_pdf_by_magic_bytes or (len(content) >= 4 and b'%PDF' in content[:1024]):
                # Looks like a PDF but wasn't detected by filename/content-type - try processing as PDF
                try:
                    if hasattr(processor, 'extract_text_from_pdf'):
                        text = processor.extract_text_from_pdf(content)
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File appears to be a PDF but PDF processing is not available. Filename: {filename}"
                        )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {filename}. Please upload .txt, .md, or .pdf files only. Error: {str(e)}"
                    )
            elif is_binary_content(content):
                # It's binary but not a PDF - reject it
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported binary file type: {filename}. The file appears to be binary but is not a PDF. Please upload .txt, .md, or .pdf files only."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {filename}. Please upload .txt, .md, or .pdf files only."
                )
        
        # Verify we have text
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="File appears to be empty or could not be processed."
            )
        
        # Preprocess text
        clean_text = processor.preprocess_text(text)
        
        # Extract metadata (use the filename we validated earlier)
        metadata = processor.extract_metadata(filename, clean_text)
        
        # Chunk document
        chunks = processor.chunk_text(clean_text)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks could be created from the document."
            )
        
        # Add to knowledge base
        # Use the filename variable we already set (or fallback to original)
        doc_filename = file.filename if file.filename else filename
        doc_id = doc_filename.replace(' ', '_').replace('.', '_')
        chunk_ids = kb.add_chunks(chunks, doc_id)
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "metadata": metadata,
            "chunks_created": len(chunk_ids),
            "chunk_ids": chunk_ids[:10]  # Return first 10
        }
    
    except HTTPException:
        raise
    except UnicodeDecodeError as e:
        # Catch any UTF-8 decode errors that slipped through
        # This should never happen for PDFs, but handle it gracefully
        import sys
        import traceback
        print("=" * 80, file=sys.stderr)
        print("FOUND UnicodeDecodeError in exception handler!", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        filename_display = file.filename if file.filename else "unknown"
        raise HTTPException(
            status_code=400,
            detail=f"File encoding error: {str(e)}. File '{filename_display}' appears to be binary or has unsupported encoding. If this is a PDF, please ensure it's a valid PDF file. Otherwise, please upload .txt, .md, or .pdf files only."
        )
    except Exception as e:
        import traceback
        error_detail = str(e)
        # Print FULL stack trace to stderr for debugging
        print("=" * 80, file=__import__('sys').stderr)
        print("FULL STACK TRACE:", file=__import__('sys').stderr)
        traceback.print_exc(file=__import__('sys').stderr)
        print("=" * 80, file=__import__('sys').stderr)
        # Provide more helpful error message for encoding issues
        if 'codec' in error_detail.lower() and 'decode' in error_detail.lower():
            raise HTTPException(
                status_code=400,
                detail=f"File encoding error: {error_detail}. The file appears to be binary. Please upload .txt, .md, or .pdf files only."
            )
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_detail}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base"""
    try:
        result = query_engine.query(request.question, request.top_k)
        # Clean result to match QueryResponse model (remove any extra fields)
        return {
            'answer': result.get('answer', ''),
            'chunks_used': result.get('chunks_used', []),
            'retrieval_time_ms': result.get('retrieval_time_ms', 0),
            'generation_time_ms': result.get('generation_time_ms', 0),
            'total_time_ms': result.get('total_time_ms', 0)
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Log the full error for debugging
        print(f"Query error: {error_details}", file=__import__('sys').stderr)
        # Return a proper error response with 200 status but error in answer field
        # This prevents frontend from showing generic 500 error
        return {
            'answer': f"Error processing query: {str(e)[:500]}",
            'chunks_used': [],
            'retrieval_time_ms': 0,
            'generation_time_ms': 0,
            'total_time_ms': 0
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    kb_stats = kb.get_stats()
    query_history = query_engine.get_query_history(limit=5)
    
    return {
        "knowledge_base": kb_stats,
        "recent_queries": [
            {
                "question": q['question'],
                "total_time_ms": q['total_time_ms'],
                "chunks_used": len(q['chunks_used'])
            }
            for q in query_history
        ]
    }

@app.get("/chunks/{doc_id}")
async def get_document_chunks(doc_id: str):
    """Get all chunks for a specific document"""
    chunks = [
        chunk for chunk in kb.chunks.values()
        if chunk['doc_id'] == doc_id
    ]
    return {"doc_id": doc_id, "chunks": chunks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
