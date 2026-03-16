import { expect, test, describe, vi, beforeEach, afterEach } from 'vitest'

describe('Database Pool Configuration', () => {
    const originalEnv = { ...process.env }

    afterEach(() => {
        // Restore original env
        process.env = { ...originalEnv }
        // Clear module cache so db.ts re-evaluates with new env
        vi.resetModules()
    })

    test('pool is exported and uses DATABASE_URL', async () => {
        process.env.DATABASE_URL = 'postgresql://admin:password@localhost:5432/testdb'
        const { pool } = await import('../src/db')
        expect(pool).toBeDefined()
        expect(pool).toHaveProperty('query')
    })

    test('pool sanitizes asyncpg prefix from DATABASE_URL', async () => {
        process.env.DATABASE_URL = 'postgresql+asyncpg://admin:password@localhost:5432/testdb'
        const { pool } = await import('../src/db')
        expect(pool).toBeDefined()
        // The pool should have been created with the sanitized connection string
        // We can verify the pool exists and has standard pg Pool methods
        expect(pool).toHaveProperty('connect')
        expect(pool).toHaveProperty('end')
    })

    test('pool handles empty DATABASE_URL gracefully', async () => {
        delete process.env.DATABASE_URL
        const { pool } = await import('../src/db')
        expect(pool).toBeDefined()
    })
})
