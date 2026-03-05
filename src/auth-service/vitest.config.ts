import { defineConfig } from 'vitest/config'

export default defineConfig({
    test: {
        include: ['tests/**/*.test.ts'],
        environment: 'node',
        coverage: {
            provider: 'v8',
            include: ['src/**/*.ts'],
            exclude: ['src/**/*.d.ts'],
            thresholds: {
                statements: 96,
                branches: 90,
                functions: 96,
                lines: 96,
            },
        },
        // Ensure each test file gets a clean module state
        restoreMocks: true,
    },
})
