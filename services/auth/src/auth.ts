import { betterAuth } from "better-auth";
import { pool } from "./db";
import dotenv from "dotenv";
dotenv.config();
import { openAPI, twoFactor, admin } from "better-auth/plugins"; // Added twoFactor and admin

if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required");
}

if (!process.env.BETTER_AUTH_SECRET && process.env.NODE_ENV === "production") {
    throw new Error("BETTER_AUTH_SECRET is required in production");
}

export const auth = betterAuth({
    database: pool,
    secret: process.env.BETTER_AUTH_SECRET || "development-secret-123", // Fallback for dev only
    emailAndPassword: {
        enabled: true
    },
    user: {
        modelName: "users",
        fields: {
            emailVerified: "is_verified",
            name: "full_name",
            createdAt: "created_at",
            updatedAt: "last_login",
            // Map two-factor plugin fields
            twoFactorEnabled: "is_mfa_enabled",
            twoFactorSecret: "mfa_secret",
            twoFactorBackupCodes: "mfa_backup_codes",
            // Map admin plugin fields
            role: "tier"
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
        twoFactor(),
        admin()
    ],
    jwt: {
        issuer: "bsopt-auth",
        expiresIn: "7d"
    },
    basePath: '/api/auth'
});
