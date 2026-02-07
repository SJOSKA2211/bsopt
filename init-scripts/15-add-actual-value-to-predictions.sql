-- Migration: Add actual_value to model_predictions
-- Created: 2026-02-06

ALTER TABLE model_predictions ADD COLUMN actual_value NUMERIC;
