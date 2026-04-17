import React from 'react';
import { motion } from 'framer-motion';

export const AnimatedCard = ({ children, delay = 0, className = "" }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
    className={`bento-card ${className}`}
  >
    {children}
  </motion.div>
);
