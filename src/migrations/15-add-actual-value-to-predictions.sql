-- Migration: Add actual_value to model_predictions
-- Created: 2026-02-06

ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS actual_value NUMERIC;
