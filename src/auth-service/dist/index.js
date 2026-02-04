"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
require("dotenv/config");
const node_server_1 = require("@hono/node-server");
const hono_1 = require("hono");
const auth_1 = require("./auth");
const app = new hono_1.Hono();
const authApp = new hono_1.Hono(); // NEW LINE
authApp.all('*', async (c) => {
    return auth_1.auth.handler(c.req.raw);
}); // NEW BLOCK END
app.route('/api/auth', authApp); // NEW LINE
app.get('/', (c) => c.text('Better Auth Service Running 🥒'));
app.get('/openapi.json', async (c) => {
    const openAPISchema = await auth_1.auth.api.generateOpenAPISchema();
    return c.json(openAPISchema);
});
const port = 4000;
console.log(`Server is running on port ${port}`);
(0, node_server_1.serve)({
    fetch: app.fetch,
    port
});
