"""initial schema with pgvector

Revision ID: 001_initial_pgvector
Revises: 
Create Date: 2026-08-22 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

revision: str = '001_initial_pgvector'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable extension if on Postgres
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=100), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('doc_metadata', sa.JSON(), nullable=False),
    )
    op.create_index('ix_documents_filename', 'documents', ['filename'])
    op.create_index('ix_documents_file_hash', 'documents', ['file_hash'])

    # 3. Create chunks table
    embedding_col = sa.Column('embedding', Vector(384), nullable=True) if (HAS_PGVECTOR and bind.dialect.name == 'postgresql') else sa.Column('embedding', sa.JSON(), nullable=True)
    
    op.create_table(
        'chunks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('document_id', sa.String(length=36), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('start_char', sa.Integer(), nullable=True),
        sa.Column('end_char', sa.Integer(), nullable=True),
        sa.Column('chunking_strategy', sa.String(length=50), nullable=False),
        embedding_col,
        sa.Column('chunk_metadata', sa.JSON(), nullable=False),
    )
    op.create_index('ix_chunks_document_id', 'chunks', ['document_id'])

    # 4. HNSW Index for Postgres pgvector
    if HAS_PGVECTOR and bind.dialect.name == 'postgresql':
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
        )


def downgrade() -> None:
    op.drop_table('chunks')
    op.drop_table('documents')
