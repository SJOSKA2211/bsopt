import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import path from 'path';
import { auth } from './auth';

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
                // Better Auth session validation (using public key if configured as asymmetric JWT)
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
                callback(err);
            }
        },
        generateToken: async (call: any, callback: any) => {
            // Internal token generation logic if needed
            callback({
                code: grpc.status.UNIMPLEMENTED,
                message: 'GenerateToken not implemented on gRPC yet. Use HTTP login.',
            });
        },
        revokeToken: async (call: any, callback: any) => {
            const { token } = call.request;
            // Add to Redis blocklist or revoke via Better Auth
            callback(null, { success: true });
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
