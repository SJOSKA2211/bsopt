-- ============================================================================
-- Black-Scholes Option Pricing Platform - Transactional Outbox
-- ============================================================================

CREATE TABLE IF NOT EXISTS outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'pending' -- pending, processed, failed
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox (created_at) WHERE status = 'pending';

-- Optional: Retention/Cleanup
-- DELETE FROM outbox WHERE processed_at < NOW() - INTERVAL '1 day';
