import { betterAuth } from "better-auth";
import { Pool } from "pg";
import dotenv from "dotenv";
import { openAPI } from "better-auth/plugins"; // New Import

export const auth = betterAuth({
    database: new Pool({ connectionString: process.env.DATABASE_URL }),
    emailAndPassword: {
        enabled: true
    },
    user: {
        modelName: "users",
        fields: {
            emailVerified: "is_verified",
            name: "full_name",
            createdAt: "created_at",
            updatedAt: "last_login" // Close enough for Better Auth's internal use
        }
    },
    session: {
        modelName: "sessions",
        fields: {
            userId: "user_id",
            createdAt: "created_at",
            updatedAt: "updated_at",
            ipAddress: "ip_address",
            userAgent: "user_agent"
        }
    },
    account: {
        modelName: "oauth_accounts",
    },
    verification: {
        modelName: "email_verification_tokens",
    },
    plugins: [
        openAPI(),
    ],
    jwt: {
        issuer: "bsopt-auth",
        expiresIn: "7d"
    },
    basePath: '/api/auth'
});
