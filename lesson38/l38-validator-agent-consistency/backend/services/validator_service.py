import asyncio
import hashlib
import logging
from typing import List, Tuple
from datetime import datetime
from models.document import Document, ValidationScore, ValidatedDocument, ValidationReport
from services.gemini_client import gemini_client
from services.cache_service import cache_service
from config.settings import settings

logger = logging.getLogger(__name__)

class ValidatorService:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_VALIDATIONS)
        
    def generate_pairs(self, documents: List[Document]) -> List[Tuple[Document, Document]]:
        """Generate all unique document pairs for validation"""
        pairs = []
        n = len(documents)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((documents[i], documents[j]))
        return pairs
    
    def compute_pair_hash(self, doc1_id: str, doc2_id: str) -> str:
        """Generate cache key for document pair"""
        # Sort IDs to ensure same hash regardless of order
        ids = sorted([doc1_id, doc2_id])
        return hashlib.md5(f"{ids[0]}-{ids[1]}".encode()).hexdigest()
    
    async def validate_pair(self, doc1: Document, doc2: Document) -> ValidationScore:
        """Validate consistency between two documents with caching"""
        pair_hash = self.compute_pair_hash(doc1.id, doc2.id)
        pair_id = f"{doc1.id}-{doc2.id}"
        
        # Check cache first
        cached_score = await cache_service.get_validation(pair_hash)
        if cached_score:
            logger.info(f"Cache hit for pair {pair_id}")
            cached_score.cached = True
            return cached_score
        
        # Validate using Gemini
        async with self.semaphore:
            try:
                logger.info(f"Validating pair {pair_id} via Gemini")
                result = await asyncio.wait_for(
                    gemini_client.validate_consistency(doc1.content, doc2.content),
                    timeout=settings.VALIDATION_TIMEOUT
                )
                
                score = ValidationScore(
                    consistency_score=result["consistency_score"],
                    contradictions=result["contradictions"],
                    confidence=result["confidence"],
                    pair_id=pair_id,
                    cached=False
                )
                
                # Cache the result
                await cache_service.set_validation(pair_hash, score)
                
                return score
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout validating pair {pair_id}")
                return ValidationScore(
                    consistency_score=0.7,
                    contradictions=[],
                    confidence=0.0,
                    pair_id=pair_id
                )
            except Exception as e:
                logger.error(f"Error validating pair {pair_id}: {str(e)}")
                return ValidationScore(
                    consistency_score=0.7,
                    contradictions=[],
                    confidence=0.0,
                    pair_id=pair_id
                )
    
    async def validate_documents(self, documents: List[Document]) -> ValidationReport:
        """Validate all documents for consistency"""
        start_time = datetime.utcnow()
        
        if len(documents) < 2:
            # Nothing to validate
            validated_docs = [
                ValidatedDocument(
                    **doc.model_dump(),
                    validation_status="skipped",
                    consistency_score=1.0,
                    contradiction_flags=[]
                ) for doc in documents
            ]
            return ValidationReport(
                overall_consistency=1.0,
                total_pairs=0,
                validated_pairs=0,
                high_conflict_pairs=[],
                cache_hit_rate=0.0,
                processing_time_ms=0,
                documents=validated_docs
            )
        
        # Generate pairs
        pairs = self.generate_pairs(documents)
        logger.info(f"Validating {len(pairs)} pairs from {len(documents)} documents")
        
        # Validate all pairs in parallel
        validation_tasks = [self.validate_pair(doc1, doc2) for doc1, doc2 in pairs]
        scores = await asyncio.gather(*validation_tasks)
        
        # Compute metrics
        cache_hits = sum(1 for s in scores if s.cached)
        cache_hit_rate = cache_hits / len(scores) if scores else 0.0
        
        # Aggregate scores per document
        doc_scores = {doc.id: [] for doc in documents}
        high_conflict_pairs = []
        
        for score, (doc1, doc2) in zip(scores, pairs):
            doc_scores[doc1.id].append(score.consistency_score)
            doc_scores[doc2.id].append(score.consistency_score)
            
            if score.consistency_score < settings.CONSISTENCY_THRESHOLD:
                high_conflict_pairs.append({
                    "doc1_id": doc1.id,
                    "doc2_id": doc2.id,
                    "consistency_score": score.consistency_score,
                    "contradictions": score.contradictions
                })
        
        # Create validated documents
        validated_docs = []
        for doc in documents:
            avg_score = sum(doc_scores[doc.id]) / len(doc_scores[doc.id]) if doc_scores[doc.id] else 1.0
            
            # Determine status
            if avg_score < 0.5:
                status = "failed"
            elif avg_score < 0.7:
                status = "warning"
            else:
                status = "passed"
            
            # Collect contradictions
            contradictions = []
            for score, (doc1, doc2) in zip(scores, pairs):
                if (doc1.id == doc.id or doc2.id == doc.id) and score.contradictions:
                    contradictions.extend(score.contradictions)
            
            validated_docs.append(ValidatedDocument(
                **doc.model_dump(),
                validation_status=status,
                consistency_score=avg_score,
                contradiction_flags=contradictions[:5]  # Top 5
            ))
        
        # Overall consistency
        overall_consistency = sum(s.consistency_score for s in scores) / len(scores) if scores else 1.0
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return ValidationReport(
            overall_consistency=overall_consistency,
            total_pairs=len(pairs),
            validated_pairs=len(scores),
            high_conflict_pairs=high_conflict_pairs,
            cache_hit_rate=cache_hit_rate,
            processing_time_ms=processing_time,
            documents=validated_docs
        )

validator_service = ValidatorService()
