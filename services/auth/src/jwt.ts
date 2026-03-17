import jwt from 'jsonwebtoken';
import fs from 'fs';
import path from 'path';

// Load keys from .env (which are Base64 encoded by bootstrap.sh)
const RS256_PRIVATE = Buffer.from(process.env.JWT_RS256_PRIVATE || '', 'base64').toString('utf-8');
const RS256_PUBLIC = Buffer.from(process.env.JWT_RS256_PUBLIC || '', 'base64').toString('utf-8');
const ES256_PRIVATE = Buffer.from(process.env.JWT_ES256_PRIVATE || '', 'base64').toString('utf-8');
const ES256_PUBLIC = Buffer.from(process.env.JWT_ES256_PUBLIC || '', 'base64').toString('utf-8');

export interface JWTPayload {
    sub: string;
    email: string;
    role: string;
    iat?: number;
    exp?: number;
    iss?: string;
    aud?: string;
}

export class JWTService {
    /**
     * Signs a payload using the RS256 (RSA) private key.
     */
    static signRS256(payload: JWTPayload): string {
        if (!RS256_PRIVATE) throw new Error("RS256_PRIVATE key missing");
        return jwt.sign(payload, RS256_PRIVATE, { 
            algorithm: 'RS256', 
            expiresIn: '7d',
            issuer: 'bsopt-auth'
        });
    }

    /**
     * Verifies a token using the RS256 (RSA) public key.
     */
    static verifyRS256(token: string): JWTPayload {
        if (!RS256_PUBLIC) throw new Error("RS256_PUBLIC key missing");
        return jwt.verify(token, RS256_PUBLIC, { 
            algorithms: ['RS256'],
            issuer: 'bsopt-auth'
        }) as JWTPayload;
    }

    /**
     * Signs a payload using the ES256 (ECC) private key (Higher Performance).
     */
    static signES256(payload: JWTPayload): string {
        if (!ES256_PRIVATE) throw new Error("ES256_PRIVATE key missing");
        return jwt.sign(payload, ES256_PRIVATE, { 
            algorithm: 'ES256', 
            expiresIn: '7d',
            issuer: 'bsopt-auth'
        });
    }

    /**
     * Verifies a token using the ES256 (ECC) public key.
     */
    static verifyES256(token: string): JWTPayload {
        if (!ES256_PUBLIC) throw new Error("ES256_PUBLIC key missing");
        return jwt.verify(token, ES256_PUBLIC, { 
            algorithms: ['ES256'],
            issuer: 'bsopt-auth'
        }) as JWTPayload;
    }
}
