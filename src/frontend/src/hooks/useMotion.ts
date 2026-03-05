import { useMemo } from 'react';
import type { Variants } from 'framer-motion';

/**
 * Custom hook providing standardized Framer Motion variants 
 * tailored for the Quantum Financial Deity (QFD) aesthetic.
 */
export const useMotion = () => {
    const transitions = useMemo(() => ({
        spring: {
            type: 'spring',
            stiffness: 300,
            damping: 30,
        },
        smooth: {
            type: 'tween',
            ease: 'easeOut',
            duration: 0.3,
        },
        hover: {
            type: 'spring',
            stiffness: 400,
            damping: 10,
        }
    }), []);

    const variants: Record<string, Variants> = useMemo(() => ({
        fadeIn: {
            initial: { opacity: 0 },
            animate: { opacity: 1 },
            exit: { opacity: 0 },
            transition: transitions.smooth
        },
        slideUp: {
            initial: { opacity: 0, y: 20 },
            animate: { opacity: 1, y: 0 },
            exit: { opacity: 0, y: -20 },
            transition: transitions.spring
        },
        glassCard: {
            initial: { opacity: 0, scale: 0.95 },
            animate: { opacity: 1, scale: 1 },
            hover: {
                scale: 1.02,
                y: -5,
                transition: transitions.hover
            },
            transition: transitions.spring
        },
        staggerContainer: {
            animate: {
                transition: {
                    staggerChildren: 0.05
                }
            }
        }
    }), [transitions]);

    return { variants, transitions };
};
