from fastapi import APIRouter, HTTPException
from typing import List
from models.document import Document, ValidationReport
from services.validator_service import validator_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/validator", tags=["validator"])

@router.post("/validate", response_model=ValidationReport)
async def validate_documents(documents: List[Document]):
    """
    Validate documents for factual consistency
    """
    try:
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        if len(documents) > 20:
            raise HTTPException(status_code=400, detail="Too many documents (max 20)")
        
        logger.info(f"Validating {len(documents)} documents")
        report = await validator_service.validate_documents(documents)
        
        return report
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "validator"}
