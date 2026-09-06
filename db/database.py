import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from config import settings

logger = logging.getLogger(__name__)

# Handle postgresql:// vs postgresql+psycopg2:// if needed
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Check if SQLite or Postgres
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    echo=settings.DB_ECHO,
    connect_args=connect_args,
    pool_pre_ping=True if not db_url.startswith("sqlite") else False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


_db_initialized = False

def init_db(engine_instance=None):
    """Initializes the database schema and enables pgvector extension if postgres."""
    global _db_initialized
    target_engine = engine_instance or engine
    if not str(target_engine.url).startswith("sqlite"):
        try:
            with target_engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
                logger.info("pgvector extension verified/created.")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension (might already exist or not postgres): {e}")

    # Import models here to register declarative schemas
    import db.models  # noqa
    Base.metadata.create_all(bind=target_engine)
    _db_initialized = True


def get_db() -> Generator[Session, None, None]:
    if not _db_initialized:
        init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
