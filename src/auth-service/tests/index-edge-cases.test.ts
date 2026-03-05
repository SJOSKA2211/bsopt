import { expect, test, describe, vi, beforeEach, afterEach } from 'vitest'

const { mockServe } = vi.hoisted(() => ({
    mockServe: vi.fn()
}))

vi.mock('@hono/node-server', () => ({
    serve: mockServe
}))

describe('Index Edge Cases', () => {
    const originalEnv = { ...process.env }

    afterEach(() => {
        process.env = { ...originalEnv }
        vi.resetModules()
        vi.useRealTimers()
        vi.restoreAllMocks()
        mockServe.mockClear()
    })

    test('Rate limit resets after window expires', async () => {
        const { app } = await import('../src/index')
        const ip = 'reset-ip-' + Date.now()
        const headers = { 'x-forwarded-for': ip }

        // Start at a fixed time
        vi.useFakeTimers()
        const startTime = 1000000
        vi.setSystemTime(startTime)

        // First request to initialize the record
        await app.request('/api/auth/reset-test', { headers })

        // Fast forward 61 seconds
        vi.advanceTimersByTime(61000)

        // Second request should trigger the reset branch
        const res = await app.request('/api/auth/reset-test', { headers })
        expect(res.status).not.toBe(429)
    })

    test('Server starts in non-test environment', async () => {
        // Mock console.log to avoid noise
        const spyLog = vi.spyOn(console, 'log').mockImplementation(() => { })

        process.env.NODE_ENV = 'development'
        process.env.PORT = '3005'

        // Import index to trigger the startup block
        await import('../src/index')

        expect(mockServe).toHaveBeenCalled()
        expect(mockServe.mock.calls[0][0]).toMatchObject({ port: 3005 })
        spyLog.mockRestore()
    })

    test('Server uses default port when PORT env is missing', async () => {
        process.env.NODE_ENV = 'development'
        delete process.env.PORT

        await import('../src/index')

        expect(mockServe).toHaveBeenCalled()
        expect(mockServe.mock.calls[0][0]).toMatchObject({ port: 3001 })
    })
})

describe('DB Edge Cases', () => {
    test('db.ts uses empty string as fallback for DATABASE_URL branch', async () => {
        vi.resetModules()
        process.env.DATABASE_URL = '' // Empty string to trigger the || "" branch
        const { pool } = await import('../src/db')
        expect(pool).toBeDefined()
    })
})
