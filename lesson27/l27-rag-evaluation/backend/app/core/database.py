from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./rag_eval.db")
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def init_db():
    try:
        from models.evaluation import Base as EvalBase
        async with engine.begin() as conn:
            await conn.run_sync(EvalBase.metadata.create_all)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"DB init skipped (using in-memory): {e}")

async def get_db():
    async with async_session_maker() as session:
        yield session
