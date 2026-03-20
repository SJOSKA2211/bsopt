import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  useTheme,
  alpha,
} from '@mui/material';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Html, Float } from '@react-three/drei';
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
  
  const { geometry, colorArray } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(10, 10, sizeX - 1, sizeY - 1);
    const vertices = geo.attributes.position.array as Float32Array;
    const colors = new Float32Array(vertices.length);
    
    const colorLow = new THREE.Color(theme.palette.primary.main);
    const colorHigh = new THREE.Color(theme.palette.secondary.main);
    
    let maxPrice = Math.max(...data);
    let minPrice = Math.min(...data);
    if (maxPrice === minPrice) maxPrice += 0.001;

    for (let i = 0; i < data.length; i++) {
      // Three.js PlaneGeometry layout: vertices are row-major
      // but the data is structured as [time1[strike1...strikeN], time2[...], ...]
      // We need to map data index to vertex height correctly.
      vertices[i * 3 + 2] = data[i] * 2; 
      
      const t = (data[i] - minPrice) / (maxPrice - minPrice);
      const color = new THREE.Color().lerpColors(colorLow, colorHigh, t);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return { geometry: geo, colorArray: colors };
  }, [data, sizeX, sizeY, theme]);

  const handlePointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (!e.face) return;

    const { x, y, z } = e.point;
    // Map world coordinates back to strike/time
    // Plane is 10x10, centered at 0,0. Rotation is -PI/2 on X, so Z is now Y in world space?
    // Actually, we rotated the mesh: rotation={[-Math.PI / 2, 0, 0]}
    // So world X is strike (-5 to 5), world Z is time (-5 to 5), world Y is price.
    
    const strikeIdx = Math.round(((x + 5) / 10) * (sizeX - 1));
    const timeIdx = Math.round(((z + 5) / 10) * (sizeY - 1));
    
    const strike = strikes[strikeIdx];
    const time = times[timeIdx];
    const price = data[timeIdx * sizeX + strikeIdx];

    onHover({ strike, time, price, point: e.point });
  }, [data, strikes, times, sizeX, sizeY, onHover]);

  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry} 
      rotation={[-Math.PI / 2, 0, 0]}
      onPointerMove={handlePointerMove}
      onPointerOut={() => onHover(null)}
    >
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.9}
        roughness={0.3}
        metalness={0.8}
        emissive={theme.palette.primary.main}
        emissiveIntensity={0.1}
      />
    </mesh>
  );
};

export const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({ symbol }) => {
  const theme = useTheme();
  const { isLoaded, batchCalculate } = useWasmPricing();
  const [surfaceData, setSurfaceData] = useState<number[]>([]);
  const [hovered, setHovered] = useState<HoverData | null>(null);

  const strikes = useMemo(() => [140, 145, 150, 155, 160, 165, 170, 175, 180, 185], []);
  const times = useMemo(() => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], []);

  useEffect(() => {
    if (!isLoaded) return;

    const params: any[] = [];
    const spot = 155.5;
    const vol = 0.25;
    const rate = 0.045;
    const div = 0.01;

    for (const t of times) {
      for (const k of strikes) {
        params.push({ spot, strike: k, time: t, vol, rate, div, is_call: true });
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
      data-testid="volatility-surface-container"
      aria-label="3D Volatility Surface Visualization"
      role="figure"
      sx={{ 
        width: '100%', 
        height: '100%', 
        position: 'relative',
        borderRadius: 4,
        overflow: 'hidden',
        background: `radial-gradient(circle at 50% 50%, ${alpha(theme.palette.background.default, 0.8)} 0%, ${theme.palette.background.default} 100%)`,
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}
    >
      <Box sx={{ position: 'absolute', top: 16, left: 16, zIndex: 10 }}>
        <Typography variant="h6" sx={{ fontWeight: 900, color: 'primary.main', textShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.5)}` }}>
          VOLATILITY MANIFOLD
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800 }}>
          {symbol} • THEORETICAL PRICE SURFACE
        </Typography>
      </Box>
      
      {!isLoaded && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 1, textAlign: 'center' }}>
          <CircularProgress size={24} sx={{ mb: 1 }} aria-label="Loading volatility surface" />
          <Typography variant="caption" display="block">Initializing Neural Rendering...</Typography>
        </Box>
      )}

      <Box sx={{ height: '100%', width: '100%', opacity: isLoaded ? 1 : 0.3 }}>
        <Canvas shadows dpr={[1, 2]}>
          <PerspectiveCamera makeDefault position={[12, 12, 12]} fov={40} />
          <OrbitControls 
            enableDamping 
            dampingFactor={0.05} 
            rotateSpeed={0.5} 
            maxPolarAngle={Math.PI / 2.1}
            minDistance={5}
            maxDistance={25}
          />
          
          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={1.5} castShadow />
          <spotLight position={[-10, 20, 10]} angle={0.15} penumbra={1} intensity={2} castShadow />

          <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
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

          <gridHelper args={[20, 20, alpha(theme.palette.primary.main, 0.2), alpha(theme.palette.divider, 0.05)]} position={[0, -0.1, 0]} />
          
          {/* Tooltip */}
          {hovered && (
            <Html position={[hovered.point.x, hovered.point.y + 0.5, hovered.point.z]} center distanceFactor={15}>
              <Box sx={{ 
                bgcolor: alpha(theme.palette.background.paper, 0.9),
                backdropFilter: 'blur(10px)',
                p: 1.5,
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                boxShadow: `0 10px 30px rgba(0,0,0,0.5), 0 0 20px ${alpha(theme.palette.primary.main, 0.2)}`,
                minWidth: 120,
                pointerEvents: 'none'
              }}>
                <Typography variant="caption" display="block" sx={{ color: 'text.secondary', fontWeight: 800 }}>
                  STRIKE: <Box component="span" sx={{ color: 'text.primary' }}>${hovered.strike}</Box>
                </Typography>
                <Typography variant="caption" display="block" sx={{ color: 'text.secondary', fontWeight: 800 }}>
                  EXPIRY: <Box component="span" sx={{ color: 'text.primary' }}>{hovered.time.toFixed(2)}Y</Box>
                </Typography>
                <Typography variant="subtitle2" sx={{ color: 'primary.main', fontWeight: 900, mt: 0.5 }}>
                  ${hovered.price.toFixed(4)}
                </Typography>
              </Box>
            </Html>
          )}

          {/* Core Labels */}
          <Text position={[6, 0, 0]} fontSize={0.4} color={theme.palette.text.secondary} rotation={[-Math.PI / 2, 0, 0]}>STRIKE →</Text>
          <Text position={[0, 0, 6]} fontSize={0.4} color={theme.palette.text.secondary} rotation={[-Math.PI / 2, 0, -Math.PI / 2]}>TIME →</Text>
        </Canvas>
      </Box>
    </Box>
  );
};
