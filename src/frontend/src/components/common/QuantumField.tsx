import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, type RootState } from '@react-three/fiber';
import { Points, PointMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Particles component that renders a field of floating points with subtle rotation.
 */
function Stars({ count = 2000 }) {
    const points = useRef<THREE.Points>(null!);

    const seededRandom = (seed: number) => {
        const x = Math.sin(seed) * 10000;
        return x - Math.floor(x);
    };

    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            pos[i * 3] = (seededRandom(i * 1.1) - 0.5) * 20;
            pos[i * 3 + 1] = (seededRandom(i * 1.2) - 0.5) * 20;
            pos[i * 3 + 2] = (seededRandom(i * 1.3) - 0.5) * 20;
        }
        return pos;
    }, [count]);

    useFrame((state: RootState) => {
        const t = state.clock.getElapsedTime() * 0.015;
        points.current.rotation.x = t;
        points.current.rotation.y = t * 1.2;
    });

    return (
        <Points ref={points} positions={positions} stride={3}>
            <PointMaterial
                transparent
                color="#00ffa3" // Institutional Mint
                size={0.012}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                opacity={0.2}
            />
        </Points>
    );
}

/**
 * Nebula particles create a soft, colored cloud effect.
 */
function NebulaCloud({ count = 50, color = "#a855f7" }) {
    const points = useRef<THREE.Points>(null!);

    const seededRandom = (seed: number) => {
        const x = Math.sin(seed) * 10000;
        return x - Math.floor(x);
    };

    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const theta = seededRandom(i * 2.1) * Math.PI * 2;
            const r = 3 + seededRandom(i * 2.2) * 4;
            pos[i * 3] = Math.cos(theta) * r + (seededRandom(i * 2.3) - 0.5) * 5;
            pos[i * 3 + 1] = Math.sin(theta) * r + (seededRandom(i * 2.4) - 0.5) * 5;
            pos[i * 3 + 2] = (seededRandom(i * 2.5) - 0.5) * 8;
        }
        return pos;
    }, [count]);

    useFrame((state: RootState) => {
        const t = state.clock.getElapsedTime() * 0.08;
        points.current.rotation.z = Math.sin(t * 0.4) * 0.15;
        points.current.position.y = Math.cos(t * 0.25) * 0.4;
    });

    return (
        <Float speed={1.5} rotationIntensity={0.4} floatIntensity={0.3}>
            <Points ref={points} positions={positions} stride={3}>
                <PointMaterial
                    transparent
                    color={color}
                    size={1.2}
                    sizeAttenuation={true}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                    opacity={0.035}
                />
            </Points>
        </Float>
    );
}

/**
 * Interactive wrapper to add parallax based on mouse position.
 */
function ParticleGroup({ children }: { children: React.ReactNode }) {
    const group = useRef<THREE.Group>(null!);
    
    useFrame((state) => {
        const { x, y } = state.mouse;
        group.current.position.x = THREE.MathUtils.lerp(group.current.position.x, x * 0.8, 0.04);
        group.current.position.y = THREE.MathUtils.lerp(group.current.position.y, y * 0.8, 0.04);
    });

    return <group ref={group}>{children}</group>;
}

/**
 * QuantumField component provides the 3D background context for the application.
 */
export const QuantumField: React.FC = () => {
    return (
        <div className="fixed inset-0 z-[-1] pointer-events-none bg-[radial-gradient(circle_at_center,_#020617_0%,_#000_100%)]">
            <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
                <fog attach="fog" args={['#020617', 2, 25]} />
                <ParticleGroup>
                  <Stars count={2500} />
                  <NebulaCloud color="#a855f7" count={40} /> {/* Purple */}
                  <NebulaCloud color="#2dd4bf" count={30} /> {/* Teal */}
                  <NebulaCloud color="#fbbf24" count={20} /> {/* Amber */}
                </ParticleGroup>
            </Canvas>
        </div>
    );
};

