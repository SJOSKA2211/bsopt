import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Particles component that renders a field of floating points with subtle rotation.
 */
function Particles({ count = 2000 }) {
    const points = useRef<THREE.Points>(null!);

    // Create a stable randomized position array for particles
    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            // Distribute in a spherical or cubic field
            pos[i * 3] = (Math.random() - 0.5) * 15;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 15;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 15;
        }
        return pos;
    }, [count]);

    // Animate the rotation and subtle movement
    useFrame((state) => {
        const t = state.clock.getElapsedTime() * 0.05;
        points.current.rotation.x = t * 1.5;
        points.current.rotation.y = t * 2.2;
        // points.current.rotation.z = Math.sin(t) * 0.2;
    });

    return (
        <Points ref={points} positions={positions} stride={3}>
            <PointMaterial
                transparent
                color="#00FFFF" // Quantum Cyan
                size={0.012}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                opacity={0.4}
            />
        </Points>
    );
}

/**
 * QuantumField component provides the 3D background context for the application.
 */
export const QuantumField: React.FC = () => {
    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                zIndex: -1,
                pointerEvents: 'none',
                background: 'radial-gradient(circle at center, #020617 0%, #000 100%)'
            }}
        >
            <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
                <fog attach="fog" args={['#020617', 5, 15]} />
                <Particles />
            </Canvas>
        </div>
    );
};
