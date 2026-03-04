import { Pool } from "pg";
import dotenv from "dotenv";
dotenv.config();

const rawUrl = process.env.DATABASE_URL || "";
const connectionString = rawUrl.replace("postgresql+asyncpg://", "postgresql://");

export const pool = new Pool({
  connectionString,
});
