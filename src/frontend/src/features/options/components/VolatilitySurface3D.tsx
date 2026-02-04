import React, { useMemo, useEffect, useState } from 'react';
import { Box, Typography, useTheme, CircularProgress } from '@mui/material';
import type { Theme } from '@mui/material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

interface VolatilitySurface3DProps {
  symbol: string;
}

const Surface: React.FC<{ theme: Theme, data: number[] }> = ({ theme, data }) => {
  const geometry = useMemo(() => {
    const segments = 30;
    const geo = new THREE.PlaneGeometry(10, 10, segments, segments);
    const vertices = geo.attributes.position.array;

    // Map WASM results to vertices
    for (let i = 0; i < vertices.length; i += 3) {
      const index = i / 3;
      if (data[index] !== undefined) {
        vertices[i + 2] = data[index] * 0.5; // Scale height for visibility
      }
    }
    
    geo.computeVertexNormals();
    return geo;
  }, [data]);

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 3, 0, 0]}>
      <meshStandardMaterial
        color={theme.palette.primary.main}
        wireframe={true}
        transparent={true}
        opacity={0.6}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

export const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({ symbol }) => {
  const theme = useTheme();
  const { batchCalculate, isLoaded } = useWasmPricing();
  const [surfaceData, setSurfaceData] = useState<number[]>([]);

  useEffect(() => {
    if (!isLoaded) return;

    // Generate grid points for the surface
    const segments = 30;
    const size = segments + 1;
    const params = [];
    const spot = 150; // Mock spot
    const rate = 0.05;
    const div = 0.0;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const time = 0.1 + (i / segments) * 2.0; // 0.1 to 2.1 years
        const strike = spot * (0.7 + (j / segments) * 0.6); // 70% to 130% of spot
        const vol = 0.2 + Math.abs(strike - spot) / spot * 0.5; // Smile mock

        params.push({
          spot,
          strike,
          time,
          vol,
          rate,
          div,
          is_call: true
        });
      }
    }

    const results = batchCalculate(params);
    setSurfaceData(results.map(r => r.price));
  }, [isLoaded, batchCalculate]);

  return (
    <Box
      data-testid="volatility-surface-container"
      sx={{ width: '100%', height: '100%', minHeight: 400, bgcolor: 'background.paper', borderRadius: 1, position: 'relative' }}
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
