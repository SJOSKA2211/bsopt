import React, { useState, useEffect, useRef, ReactNode } from 'react';
import { Box, Skeleton } from '@mui/material';

interface LazyComponentProps {
  children: ReactNode;
  placeholder?: ReactNode;
  height?: string | number;
  rootMargin?: string;
}

const LazyComponent: React.FC<LazyComponentProps> = ({
  children,
  placeholder,
  height = 200,
  rootMargin = '100px',
}) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsIntersecting(true);
          observer.disconnect();
        }
      },
      { rootMargin }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [rootMargin]);

  return (
    <div ref={containerRef} style={{ minHeight: isIntersecting ? 'auto' : height }}>
      {isIntersecting ? (
        children
      ) : (
        placeholder || (
          <Box sx={{ width: '100%', height }}>
            <Skeleton variant="rectangular" width="100%" height={height} />
          </Box>
        )
      )}
    </div>
  );
};

export default LazyComponent;
