import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import path from 'path';
import { auth } from './auth';
import { JWTService } from './jwt';

import { redis } from './redis';

const PROTO_PATH = path.resolve(__dirname, '../../protos/auth.proto');

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
});

const authProto = grpc.loadPackageDefinition(packageDefinition) as any;

export function startGrpcServer() {
    const server = new grpc.Server();

    server.addService(authProto.auth.AuthService.service, {
        validateToken: async (call: any, callback: any) => {
            const { token } = call.request;
            try {
                // 0. Check Redis blocklist
                const isRevoked = await redis.get(`revoked_token:${token}`);
                if (isRevoked) {
                    callback(null, { valid: false });
                    return;
                }

                // 1. Institutional High-Performance Asymmetric Validation (ES256)
                let decoded;
                try {
                    decoded = JWTService.verifyES256(token);
                } catch (e) {
                    // Fallback to RS256 for legacy compatibility
                    decoded = JWTService.verifyRS256(token);
                }

                if (decoded) {
                    callback(null, { 
                        valid: true, 
                        token, 
                        role: decoded.role || 'user' 
                    });
                    return;
                }

                // 2. Fallback to session check if JWT fails but might be a session ID
                const session = await auth.api.getSession({
                    headers: new Headers({
                        Authorization: `Bearer ${token}`
                    })
                });

                if (session) {
                    callback(null, { 
                        valid: true, 
                        token, 
                        role: session.user.role || 'user' 
                    });
                } else {
                    callback(null, { valid: false });
                }
            } catch (err) {
                callback(null, { valid: false });
            }
        },
        generateToken: async (call: any, callback: any) => {
            const { user_id, role } = call.request;
            try {
                const token = JWTService.signES256({
                    sub: user_id,
                    email: '', // Not always available in internal rpc
                    role: role || 'user'
                });
                callback(null, { valid: true, token, role });
            } catch (err) {
                callback(err);
            }
        },
        revokeToken: async (call: any, callback: any) => {
            const { token } = call.request;
            try {
                // Add to Redis blocklist with 7 days expiration (matching JWT expiry)
                await redis.set(`revoked_token:${token}`, '1', 'EX', 7 * 24 * 60 * 60);
                callback(null, { success: true });
            } catch (err) {
                callback(err);
            }
        },
    });

    const port = process.env.GRPC_PORT || '50051';
    server.bindAsync(`0.0.0.0:${port}`, grpc.ServerCredentials.createInsecure(), (err, port) => {
        if (err) {
            console.error(`gRPC server failed to bind: ${err.message}`);
            return;
        }
        console.log(`gRPC Auth Service running on port ${port}`);
    });
}
