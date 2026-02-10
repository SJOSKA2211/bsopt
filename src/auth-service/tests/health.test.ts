import { expect, test } from 'vitest'
import { app } from '../src/index'

test('GET / returns health message', async () => {
  const res = await app.request('/')
  expect(res.status).toBe(200)
  expect(await res.text()).toBe('Better Auth Service Running ')
})
