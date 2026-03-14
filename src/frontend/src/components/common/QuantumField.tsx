import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, type RootState } from '@react-three/fiber';
import { Points, PointMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Particles component that renders a field of floating points with subtle rotation.
 */
function Stars({ count = 1500 }) {
    const points = useRef<THREE.Points>(null!);

    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            pos[i * 3] = (Math.random() - 0.5) * 20;
            pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
        }
        return pos;
    }, [count]);

    useFrame((state: RootState) => {
        const t = state.clock.getElapsedTime() * 0.02;
        points.current.rotation.x = t;
        points.current.rotation.y = t * 1.2;
    });

    return (
        <Points ref={points} positions={positions} stride={3}>
            <PointMaterial
                transparent
                color="#00FFFF" // Quantum Cyan
                size={0.015}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                opacity={0.3}
            />
        </Points>
    );
}

/**
 * Nebula particles create a soft, colored cloud effect.
 */
function NebulaCloud({ count = 40, color = "#7B68EE" }) {
    const points = useRef<THREE.Points>(null!);

    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            // Clumped distribution
            const theta = Math.random() * Math.PI * 2;
            const r = 2 + Math.random() * 3;
            pos[i * 3] = Math.cos(theta) * r + (Math.random() - 0.5) * 4;
            pos[i * 3 + 1] = Math.sin(theta) * r + (Math.random() - 0.5) * 4;
            pos[i * 3 + 2] = (Math.random() - 0.5) * 6;
        }
        return pos;
    }, [count]);

    useFrame((state: RootState) => {
        const t = state.clock.getElapsedTime() * 0.1;
        points.current.rotation.z = Math.sin(t * 0.5) * 0.2;
        points.current.position.y = Math.cos(t * 0.3) * 0.5;
    });

    return (
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
            <Points ref={points} positions={positions} stride={3}>
                <PointMaterial
                    transparent
                    color={color}
                    size={0.8}
                    sizeAttenuation={true}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                    opacity={0.05}
                />
            </Points>
        </Float>
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
            <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
                <fog attach="fog" args={['#020617', 5, 20]} />
                <Stars />
                <NebulaCloud color="#7B68EE" count={30} />
                <NebulaCloud color="#00FFFF" count={20} />
                <NebulaCloud color="#D4AF37" count={15} />
            </Canvas>
        </div>
    );
};

