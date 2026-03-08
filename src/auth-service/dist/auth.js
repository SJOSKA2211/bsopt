"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
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
        enabled: true,
        password: {
            hash: async (password) => {
                const { hash } = await Promise.resolve().then(() => __importStar(require("@node-rs/argon2")));
                return hash(password, {
                    memoryCost: 65536,
                    timeCost: 3,
                    parallelism: 4
                });
            }
        }
    },
    user: {
        modelName: "users",
        fields: {
            emailVerified: "is_verified",
            name: "full_name",
            createdAt: "created_at",
            updatedAt: "last_login"
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
