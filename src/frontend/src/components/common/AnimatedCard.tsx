import { motion, HTMLMotionProps } from 'framer-motion';
import React from 'react';

interface AnimatedCardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}

/**
 * A reusable Bento-style card with premium entrance animations and hover effects.
 */
export const AnimatedCard: React.FC<AnimatedCardProps> = ({ 
  children, 
  className = '', 
  delay = 0,
  ...props 
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        duration: 0.5, 
        delay, 
        ease: [0.23, 1, 0.32, 1] 
      }}
      whileHover={{ 
        y: -5,
        transition: { duration: 0.2 }
      }}
      className={`bento-card ${className}`}
      {...props}
    >
      {/* Subtle shimmer effect on entry */}
      <motion.div 
        className="shimmer-overlay"
        initial={{ x: '-100%' }}
        animate={{ x: '100%' }}
        transition={{ duration: 1.5, delay: delay + 0.5, ease: "easeInOut" }}
      />
      {children}
    </motion.div>
  );
};
