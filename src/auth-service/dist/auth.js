"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.auth = void 0;
const better_auth_1 = require("better-auth");
const pg_1 = require("pg");
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const plugins_1 = require("better-auth/plugins"); // Added twoFactor and admin
if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required");
}
if (!process.env.BETTER_AUTH_SECRET && process.env.NODE_ENV === "production") {
    throw new Error("BETTER_AUTH_SECRET is required in production");
}
exports.auth = (0, better_auth_1.betterAuth)({
    database: new pg_1.Pool({
        connectionString: (process.env.DATABASE_URL || "").replace("postgresql+asyncpg://", "postgresql://")
    }),
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
        (0, plugins_1.openAPI)(),
        (0, plugins_1.twoFactor)(),
        (0, plugins_1.admin)()
    ],
    jwt: {
        issuer: "bsopt-auth",
        expiresIn: "7d"
    },
    basePath: '/api/auth'
});
