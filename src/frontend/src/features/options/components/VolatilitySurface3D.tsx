import React, { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  useTheme,
  alpha,
} from '@mui/material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

interface VolatilitySurface3DProps {
  symbol: string;
}

const Surface: React.FC<{ theme: any; data: number[] }> = ({ theme, data }) => {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const size = Math.sqrt(data.length);
  
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(10, 10, size - 1, size - 1);
    const vertices = geo.attributes.position.array as Float32Array;
    
    for (let i = 0; i < data.length; i++) {
      // Index 2 is Z-axis in Three.js plane
      vertices[i * 3 + 2] = data[i] * 2; 
    }
    
    geo.computeVertexNormals();
    return geo;
  }, [data, size]);

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshPhongMaterial
        color={theme.palette.primary.main}
        specular={0x111111}
        shininess={100}
        side={THREE.DoubleSide}
        transparent
        opacity={0.8}
        wireframe
      />
    </mesh>
  );
};

export const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({ symbol }) => {
  const theme = useTheme();
  const { isLoaded, batchCalculate } = useWasmPricing();
  const [surfaceData, setSurfaceData] = useState<number[]>([]);

  useEffect(() => {
    if (!isLoaded) return;

    // Generate grid points for surface
    const strikes = [140, 145, 150, 155, 160, 165, 170, 175, 180, 185];
    const times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    const params: any[] = [];
    const spot = 155.5;
    const vol = 0.25;
    const rate = 0.05;
    const div = 0.01;

    for (const t of times) {
      for (const k of strikes) {
        params.push({
          spot,
          strike: k,
          time: t,
          vol,
          rate,
          div,
          is_call: true
        });
      }
    }

    const fetchData = async () => {
      // @ts-ignore
      const results = await batchCalculate(params);
      setSurfaceData(results.map((r: any) => r.price));
    };
    fetchData();
  }, [isLoaded, batchCalculate]);

  return (
    <Box
      data-testid="volatility-surface-container"
      aria-label="3D Volatility Surface Visualization"
      role="figure"
    >
      <Typography variant="subtitle2" align="center" sx={{ pt: 1, color: 'text.secondary' }}>
        3D Theoretical Price Surface (WASM) - {symbol}
      </Typography>
      
      {!isLoaded && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 1, textAlign: 'center' }}>
          <CircularProgress size={24} sx={{ mb: 1 }} />
          <Typography variant="caption" display="block">Loading WASM Engine...</Typography>
        </Box>
      )}

      <Box sx={{ height: 'calc(100% - 30px)', width: '100%', opacity: isLoaded ? 1 : 0.3 }}>
        <Canvas>
          <PerspectiveCamera makeDefault position={[10, 10, 10]} />
          <OrbitControls enableDamping dampingFactor={0.05} rotateSpeed={0.5} />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          {surfaceData.length > 0 && <Surface theme={theme} data={surfaceData} />}
          <gridHelper args={[10, 10, 0x444444, 0x222222]} rotation={[Math.PI / 2, 0, 0]} />
          <Text position={[6, -5, 0]} fontSize={0.5} color={theme.palette.text.secondary}>Time</Text>
          <Text position={[-6, 0, 0]} fontSize={0.5} color={theme.palette.text.secondary} rotation={[0, 0, Math.PI / 2]}>Strike</Text>
          <Text position={[0, 0, 3]} fontSize={0.5} color={theme.palette.text.secondary}>Price</Text>
        </Canvas>
      </Box>
    </Box>
  );
};
