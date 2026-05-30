-- GeneReviews RAG schema for pgvector
-- Stores chunked sections from GeneReviews articles for semantic search

-- Drop existing table if needed (comment out in production)
-- DROP TABLE IF EXISTS genereviews_chunks;

CREATE TABLE IF NOT EXISTS genereviews_chunks (
    id SERIAL PRIMARY KEY,

    -- Article metadata
    nbk_id TEXT NOT NULL,              -- e.g., "NBK1211"
    shortname TEXT NOT NULL,           -- e.g., "hnpcc"
    article_title TEXT NOT NULL,       -- e.g., "Lynch Syndrome"
    condition_name TEXT,               -- e.g., "Lynch Syndrome" (standardized from metadata)
    gene_symbols TEXT[],               -- e.g., {"MLH1", "MSH2", "MSH6", "PMS2"}

    -- Section metadata
    section_id TEXT,                   -- e.g., "hnpcc.Diagnosis"
    section_title TEXT,                -- e.g., "Diagnosis"
    section_path TEXT,                 -- e.g., "Diagnosis > Suggestive Findings"
    section_type TEXT,                 -- normalized: diagnosis, management, etc.

    -- Content
    chunk_text TEXT NOT NULL,
    chunk_index INT DEFAULT 0,         -- for multi-chunk sections
    char_count INT,
    token_estimate INT,

    -- Vector embedding (BGE-large uses 1024 dimensions)
    embedding vector(1024),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for filtering
CREATE INDEX IF NOT EXISTS idx_genereviews_nbk_id ON genereviews_chunks(nbk_id);
CREATE INDEX IF NOT EXISTS idx_genereviews_shortname ON genereviews_chunks(shortname);
CREATE INDEX IF NOT EXISTS idx_genereviews_section_type ON genereviews_chunks(section_type);
CREATE INDEX IF NOT EXISTS idx_genereviews_gene_symbols ON genereviews_chunks USING GIN(gene_symbols);

-- HNSW index for vector similarity search (faster than IVFFlat for small-medium datasets)
CREATE INDEX IF NOT EXISTS idx_genereviews_embedding ON genereviews_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Text search index for hybrid search
CREATE INDEX IF NOT EXISTS idx_genereviews_text_search ON genereviews_chunks
    USING GIN(to_tsvector('english', chunk_text));

-- Useful views
CREATE OR REPLACE VIEW genereviews_stats AS
SELECT
    section_type,
    COUNT(*) as chunk_count,
    SUM(token_estimate) as total_tokens,
    AVG(token_estimate)::int as avg_tokens
FROM genereviews_chunks
GROUP BY section_type
ORDER BY chunk_count DESC;

CREATE OR REPLACE VIEW genereviews_articles AS
SELECT DISTINCT ON (nbk_id)
    nbk_id,
    shortname,
    article_title,
    gene_symbols,
    COUNT(*) OVER (PARTITION BY nbk_id) as chunk_count,
    SUM(token_estimate) OVER (PARTITION BY nbk_id) as total_tokens
FROM genereviews_chunks
ORDER BY nbk_id, id;

-- Comment on table
COMMENT ON TABLE genereviews_chunks IS 'GeneReviews article sections for RAG retrieval';
COMMENT ON COLUMN genereviews_chunks.embedding IS 'BGE-large-en-v1.5 embedding (1024 dimensions)';
