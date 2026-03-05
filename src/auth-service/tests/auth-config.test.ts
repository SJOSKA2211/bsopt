import { expect, test, describe, vi, beforeEach, afterEach } from 'vitest'

describe('Auth Configuration Validation', () => {
    const originalEnv = { ...process.env }

    afterEach(() => {
        // Restore original env
        process.env = { ...originalEnv }
        // Clear module cache
        vi.resetModules()
    })

    test('Throws error if DATABASE_URL is missing', async () => {
        // Mock dotenv.config to do nothing, so it doesn't reload .env
        const dotenv = await import('dotenv');
        vi.spyOn(dotenv.default, 'config').mockImplementation(() => ({ parsed: {} }));

        delete process.env.DATABASE_URL
        // auth.ts should throw an error upon import
        await expect(import('../src/auth')).rejects.toThrow('DATABASE_URL is required')
    })


    test('Throws error if BETTER_AUTH_SECRET is missing in production', async () => {
        process.env.DATABASE_URL = 'postgresql://admin:password@localhost:5432/testdb'
        delete process.env.BETTER_AUTH_SECRET
        process.env.NODE_ENV = 'production'

        await expect(import('../src/auth')).rejects.toThrow('BETTER_AUTH_SECRET is required in production')
    })

    test('Uses fallback secret in development if BETTER_AUTH_SECRET is missing', async () => {
        process.env.DATABASE_URL = 'postgresql://admin:password@localhost:5432/testdb'
        delete process.env.BETTER_AUTH_SECRET
        process.env.NODE_ENV = 'development'

        // This should not throw
        const { auth } = await import('../src/auth')
        expect(auth).toBeDefined()
    })
})
