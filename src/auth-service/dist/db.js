"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.pool = void 0;
const pg_1 = require("pg");
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const rawUrl = process.env.DATABASE_URL || "";
const connectionString = rawUrl.replace("postgresql+asyncpg://", "postgresql://");
exports.pool = new pg_1.Pool({
    connectionString,
    max: 5, // Limit connections for this service
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000,
});
