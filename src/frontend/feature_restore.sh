#!/bin/bash
ROOT="/home/kamau/bsopt/src/frontend"
mkdir -p "$ROOT/src/components/common"
mkdir -p "$ROOT/src/features/dashboard/components"

# AnimatedCard.tsx
cat > "$ROOT/src/components/common/AnimatedCard.tsx" <<'EOF'
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
EOF

# DeepInferenceEngine.tsx
cat > "$ROOT/src/features/dashboard/components/DeepInferenceEngine.tsx" <<'EOF'
import React from 'react';
import { motion } from 'framer-motion';

export const DeepInferenceEngine = () => (
  <div className="h-full flex flex-col justify-center items-center opacity-10">
     <div className="w-24 h-24 border-2 border-mint rounded-full animate-ping mb-8" />
     <span className="text-[10px] font-black tracking-[1em] uppercase">SYSTEM_BOOTLOADER_ACTIVE</span>
  </div>
);
EOF

# RiskExposureGrid.tsx
cat > "$ROOT/src/features/dashboard/components/RiskExposureGrid.tsx" <<'EOF'
import React from 'react';
import { motion } from 'framer-motion';

export const RiskExposureGrid = () => (
  <div className="p-8 space-y-6">
     {[...Array(4)].map((_, i) => (
        <div key={i} className="space-y-2">
           <div className="flex justify-between text-[10px] font-black text-white/40 uppercase">
              <span>SECTOR_ALPHA_{i}</span>
              <span className="text-mint">{(85 - i * 15)}%</span>
           </div>
           <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
              <motion.div initial={{ width: 0 }} animate={{ width: `${85 - i * 15}%` }} className="h-full bg-mint shadow-[0_0_10px_#00ffa3]" />
           </div>
        </div>
     ))}
  </div>
);
EOF

echo "Feature Components Restored."
