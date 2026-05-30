-- AttenGene Database Schema
-- PostgreSQL with pgvector extension

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ClinVar variants table
CREATE TABLE IF NOT EXISTS variants (
    id SERIAL PRIMARY KEY,
    variation_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    gene TEXT NOT NULL,
    clinical_significance TEXT NOT NULL,
    review_status TEXT,
    phenotypes TEXT[],

    -- Document text for RAG (name + gene + phenotypes combined)
    document TEXT NOT NULL,

    -- Vector embedding (BGE-large-en = 1024 dimensions)
    embedding vector(1024),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_variants_gene ON variants(gene);
CREATE INDEX IF NOT EXISTS idx_variants_variation_id ON variants(variation_id);
CREATE INDEX IF NOT EXISTS idx_variants_clinical_significance ON variants(clinical_significance);

-- HNSW index for fast vector similarity search
-- Using cosine distance (best for normalized embeddings like BGE)
CREATE INDEX IF NOT EXISTS idx_variants_embedding ON variants
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS idx_variants_document_fts ON variants
USING gin(to_tsvector('english', document));

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating timestamp
DROP TRIGGER IF EXISTS variants_updated_at ON variants;
CREATE TRIGGER variants_updated_at
    BEFORE UPDATE ON variants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
