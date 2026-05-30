-- AttenGene Audit Tables
-- HIPAA-compliant logging (6-year retention required)

-- Audit log for all queries
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,

    -- Session and user info
    session_id TEXT NOT NULL,
    user_id TEXT,  -- NULL for anonymous/demo users
    client_ip INET,

    -- Query details
    query_text TEXT NOT NULL,
    query_type TEXT,  -- 'variant', 'gene', 'syndrome', 'general', 'rejected'

    -- Retrieval info (proves model didn't invent context)
    retrieved_variant_ids INTEGER[],
    retrieval_scores JSONB,  -- [{variant_id, score, rank}, ...]

    -- Response
    response_text TEXT,
    was_rejected BOOLEAN DEFAULT FALSE,
    rejection_reason TEXT,

    -- Model info
    model_name TEXT NOT NULL,
    model_version TEXT,

    -- Data version
    clinvar_version TEXT,

    -- Performance
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    total_time_ms INTEGER,

    -- Errors
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for querying audit logs
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_rejected ON audit_logs(was_rejected);

-- Sessions table for tracking demo limits
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    client_ip INET,
    query_count INTEGER DEFAULT 0,
    first_query_at TIMESTAMP,
    last_query_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_ip ON sessions(client_ip);

-- Function to increment query count
CREATE OR REPLACE FUNCTION increment_query_count(p_session_id TEXT, p_client_ip INET)
RETURNS INTEGER AS $$
DECLARE
    new_count INTEGER;
BEGIN
    INSERT INTO sessions (session_id, client_ip, query_count, first_query_at, last_query_at)
    VALUES (p_session_id, p_client_ip, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ON CONFLICT (session_id) DO UPDATE
    SET query_count = sessions.query_count + 1,
        last_query_at = CURRENT_TIMESTAMP
    RETURNING query_count INTO new_count;

    RETURN new_count;
END;
$$ LANGUAGE plpgsql;

-- Authentication event log
CREATE TABLE IF NOT EXISTS auth_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,       -- 'api_key_success', 'api_key_failure', 'api_key_missing'
    client_ip INET,
    user_agent TEXT,
    endpoint TEXT,                  -- e.g. '/v1/chat/completions'
    detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_auth_events_type ON auth_events(event_type);
CREATE INDEX IF NOT EXISTS idx_auth_events_created ON auth_events(created_at);
CREATE INDEX IF NOT EXISTS idx_auth_events_ip ON auth_events(client_ip);
