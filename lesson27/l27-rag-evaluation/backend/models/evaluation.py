from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class EvaluationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String, index=True)
    status = Column(Enum(EvaluationStatus), default=EvaluationStatus.PENDING)
    model_config = Column(JSON)
    
    # RAGAS Metrics
    faithfulness_score = Column(Float, nullable=True)
    answer_relevance_score = Column(Float, nullable=True)
    context_precision_score = Column(Float, nullable=True)
    context_recall_score = Column(Float, nullable=True)
    
    # Metadata
    total_queries = Column(Integer)
    completed_queries = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
class QueryEvaluation(Base):
    __tablename__ = "query_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, index=True)
    
    question = Column(String)
    answer = Column(String)
    contexts = Column(JSON)  # List of retrieved contexts
    ground_truth = Column(String, nullable=True)
    
    # Individual Scores
    faithfulness = Column(Float)
    answer_relevance = Column(Float)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    
    # Timing
    evaluation_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
