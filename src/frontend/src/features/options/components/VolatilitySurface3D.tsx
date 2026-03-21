import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  useTheme,
  alpha,
  Stack,
} from '@mui/material';
import { Canvas, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Html, Float, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

interface VolatilitySurface3DProps {
  symbol: string;
}

interface HoverData {
  strike: number;
  time: number;
  price: number;
  point: THREE.Vector3;
}

const Surface: React.FC<{ 
  theme: any; 
  data: number[]; 
  strikes: number[]; 
  times: number[]; 
  onHover: (data: HoverData | null) => void;
}> = ({ theme, data, strikes, times, onHover }) => {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const sizeX = strikes.length;
  const sizeY = times.length;
  
  const { geometry } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(10, 10, sizeX - 1, sizeY - 1);
    const vertices = geo.attributes.position.array as Float32Array;
    const colors = new Float32Array(vertices.length);
    
    const colorLow = new THREE.Color(theme.palette.primary.main);
    const colorHigh = new THREE.Color(theme.palette.secondary.main);
    
    let maxPrice = Math.max(...data);
    let minPrice = Math.min(...data);
    if (maxPrice === minPrice) maxPrice += 0.001;

    for (let i = 0; i < data.length; i++) {
      vertices[i * 3 + 2] = data[i] * 3; // Scale height for drama
      
      const t = (data[i] - minPrice) / (maxPrice - minPrice);
      const color = new THREE.Color().lerpColors(colorLow, colorHigh, t);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return { geometry: geo };
  }, [data, sizeX, sizeY, theme]);

  const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const strikeIdx = Math.round(((e.point.x + 5) / 10) * (sizeX - 1));
    const timeIdx = Math.round(((e.point.z + 5) / 10) * (sizeY - 1));
    const strike = strikes[strikeIdx];
    const time = times[timeIdx];
    const price = data[timeIdx * sizeX + strikeIdx];
    onHover({ strike, time, price, point: e.point });
  }, [data, strikes, times, sizeX, sizeY, onHover]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh 
        ref={meshRef} 
        geometry={geometry} 
        onPointerMove={handlePointerMove}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          wireframe={false}
          transparent
          opacity={0.8}
          roughness={0.1}
          metalness={0.9}
        />
      </mesh>
      <mesh geometry={geometry}>
        <meshBasicMaterial vertexColors wireframe transparent opacity={0.15} />
      </mesh>
    </group>
  );
};

export const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({ symbol }) => {
  const theme = useTheme();
  const { isLoaded, batchCalculate } = useWasmPricing();
  const [surfaceData, setSurfaceData] = useState<number[]>([]);
  const [hovered, setHovered] = useState<HoverData | null>(null);

  const strikes = useMemo(() => Array.from({ length: 15 }, (_, i) => 140 + i * 5), []);
  const times = useMemo(() => Array.from({ length: 15 }, (_, i) => 0.1 + i * 0.1), []);

  useEffect(() => {
    if (!isLoaded) return;
    const params: any[] = [];
    const spot = 155.5;
    for (const t of times) {
      for (const k of strikes) {
        params.push({ spot, strike: k, time: t, vol: 0.25, rate: 0.045, div: 0.01, is_call: true });
      }
    }
    const fetchData = async () => {
      const results: any = await batchCalculate(params);
      setSurfaceData(results.map((r: any) => r.price));
    };
    fetchData();
  }, [isLoaded, batchCalculate, strikes, times]);

  return (
    <Box
      sx={{ 
        width: '100%', 
        height: '100%', 
        position: 'relative',
        borderRadius: 6,
        overflow: 'hidden',
        background: `radial-gradient(circle at 50% 50%, ${alpha('#0f172a', 0.8)} 0%, #020617 100%)`,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
        boxShadow: `0 0 40px ${alpha('#000', 0.6)} inset`
      }}
    >
      <Box sx={{ position: 'absolute', top: 24, left: 24, zIndex: 10 }}>
        <Typography variant="h5" sx={{ fontWeight: 900, color: 'primary.main', letterSpacing: '-0.02em' }}>
          VolX Manifold: {symbol}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          Theoretical Neural Surface • WASM Accelerated
        </Typography>
      </Box>

      {!isLoaded && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 1, textAlign: 'center' }}>
          <CircularProgress size={24} sx={{ mb: 1 }} aria-label="Loading Volatility Surface" />
          <Typography variant="caption" display="block">Initializing Neural Rendering...</Typography>
        </Box>
      )}
      
      <Box sx={{ height: '100%', width: '100%' }}>
        <Canvas shadows dpr={[1, 2]}>
          <PerspectiveCamera makeDefault position={[15, 15, 15]} fov={35} />
          <OrbitControls 
            enableDamping 
            dampingFactor={0.05} 
            maxPolarAngle={Math.PI / 2.2}
            minDistance={8}
            maxDistance={30}
          />
          
          <ambientLight intensity={0.5} />
          <spotLight position={[10, 20, 10]} angle={0.15} penumbra={1} intensity={2} color={theme.palette.primary.main} />
          <pointLight position={[-10, -10, -10]} intensity={1} color={theme.palette.secondary.main} />

          <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
            {surfaceData.length > 0 && (
              <Surface 
                theme={theme} 
                data={surfaceData} 
                strikes={strikes} 
                times={times} 
                onHover={setHovered}
              />
            )}
          </Float>

          <gridHelper args={[20, 20, alpha(theme.palette.primary.main, 0.3), alpha(theme.palette.divider, 0.05)]} position={[0, -0.01, 0]} />
          
          {/* Labels */}
          <Billboard position={[6, 0, 0]}>
            <Text fontSize={0.35} color={theme.palette.text.secondary} fontWeight={900}>STRIKE (K)</Text>
          </Billboard>
          <Billboard position={[0, 0, 6]}>
            <Text fontSize={0.35} color={theme.palette.text.secondary} fontWeight={900} rotation={[0, Math.PI / 2, 0]}>TIME (T)</Text>
          </Billboard>

          {/* Tooltip */}
          {hovered && (
            <Html position={[hovered.point.x, hovered.point.y + 1, hovered.point.z]} center distanceFactor={12}>
              <Box sx={{ 
                bgcolor: alpha('#0f172a', 0.95),
                backdropFilter: 'blur(16px)',
                p: 2,
                borderRadius: 3,
                border: `1px solid ${alpha(theme.palette.primary.main, 0.4)}`,
                boxShadow: `0 20px 50px rgba(0,0,0,0.8), 0 0 30px ${alpha(theme.palette.primary.main, 0.2)}`,
                minWidth: 160,
                pointerEvents: 'none'
              }}>
                <Stack spacing={0.5}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 900, textTransform: 'uppercase', fontSize: '0.6rem' }}>
                    Surface Parameters
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 900, color: 'text.primary', fontFamily: 'JetBrains Mono' }}>
                    K: ${hovered.strike} | T: {hovered.time.toFixed(2)}Y
                  </Typography>
                  <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 900, fontFamily: 'JetBrains Mono' }}>
                    ${hovered.price.toFixed(4)}
                  </Typography>
                </Stack>
              </Box>
            </Html>
          )}
        </Canvas>
      </Box>
    </Box>
  );
};

export default VolatilitySurface3D;
