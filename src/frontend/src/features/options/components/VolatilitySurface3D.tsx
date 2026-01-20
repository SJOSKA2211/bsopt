import React, { useMemo } from 'react';
import { Box, Typography, useTheme, Theme } from '@mui/material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';

interface VolatilitySurface3DProps {
  symbol: string;
}

const Surface: React.FC<{ theme: Theme }> = ({ theme }) => {
  const geometry = useMemo(() => {
    const segments = 30;
    const geo = new THREE.PlaneGeometry(10, 10, segments, segments);
    const vertices = geo.attributes.position.array;

    for (let i = 0; i < vertices.length; i += 3) {
      const y = vertices[i + 1]; // Strike
      const x = vertices[i];     // Expiry
      const distFromATM = Math.abs(y);
      const timeEffect = 1 / (x + 6);
      const iv = 0.2 + (distFromATM * distFromATM * 0.05) + (timeEffect * 0.1);
      vertices[i + 2] = iv * 2;
    }
    
    geo.computeVertexNormals();
    return geo;
  }, []);

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

  return (
    <Box
      data-testid="volatility-surface-container"
      sx={{ width: '100%', height: '100%', minHeight: 400, bgcolor: 'background.paper', borderRadius: 1 }}
    >
      <Typography variant="subtitle2" align="center" sx={{ pt: 1, color: 'text.secondary' }}>
        3D Volatility Surface - {symbol}
      </Typography>
      <Box sx={{ height: 'calc(100% - 30px)', width: '100%' }}>
        <Canvas>
          <PerspectiveCamera makeDefault position={[10, 10, 10]} />
          <OrbitControls enableDamping dampingFactor={0.05} rotateSpeed={0.5} />
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <Surface theme={theme} />
          <gridHelper args={[10, 10, 0x444444, 0x222222]} rotation={[Math.PI / 2, 0, 0]} />
          <Text position={[6, -5, 0]} fontSize={0.5} color={theme.palette.text.secondary}>Expiry</Text>
          <Text position={[-6, 0, 0]} fontSize={0.5} color={theme.palette.text.secondary} rotation={[0, 0, Math.PI / 2]}>Strike</Text>
          <Text position={[0, 0, 3]} fontSize={0.5} color={theme.palette.text.secondary}>IV</Text>
        </Canvas>
      </Box>
    </Box>
  );
};
