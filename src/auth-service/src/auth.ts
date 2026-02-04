import { betterAuth } from "better-auth";
import { Pool } from "pg";
import dotenv from "dotenv";
import { openAPI } from "better-auth/plugins"; // New Import

export const auth = betterAuth({
    database: new Pool({ connectionString: process.env.DATABASE_URL }),
    emailAndPassword: {
        enabled: true
    },
    plugins: [
        openAPI(),
    ],
    basePath: '/api/auth' // NEW LINE
});
