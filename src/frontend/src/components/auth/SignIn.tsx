import { useState } from 'react';
import { AnimatedCard } from '../common/AnimatedCard';

const SystemStatus: React.FC = () => (
  <div className="flex items-center gap-3 mb-6">
    <div className="relative w-2 h-2">
      <div className="status-pill healthy w-2 h-2 p-0 rounded-full bg-mint" />
      <div className="absolute -inset-1 rounded-full border border-mint/30 animate-pulse" />
    </div>
    <span className="text-[10px] text-mint font-bold tracking-[0.1em] uppercase data-mono">
      Manifold_L1_Active
    </span>
  </div>
);

const InputField: React.FC<{
  label: string;
  type: string;
  value: string;
  onChange: (val: string) => void;
  icon?: React.ReactNode;
  placeholder?: string;
}> = ({ label, type, value, onChange, icon, placeholder }) => (
  <div className="flex flex-col gap-2">
    <label className="text-[11px] font-bold text-white/40 uppercase tracking-wider ml-1">{label}</label>
    <div className="relative group">
       {icon && (
         <div className="absolute left-4 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-mint transition-colors duration-300">
           {icon}
         </div>
       )}
       <input 
         type={type}
         value={value}
         onChange={(e) => onChange(e.target.value)}
         placeholder={placeholder}
         className={`w-full bg-white/5 border border-white/10 rounded-xl py-3 ${icon ? 'pl-11' : 'px-4'} pr-4 text-white text-sm outline-none transition-all duration-300 focus:border-mint/50 focus:bg-white/10 focus:ring-1 focus:ring-mint/20 placeholder:text-white/10`}
       />
    </div>
  </div>
);

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    // Mock authentication logic matching premium transition speed
    setTimeout(() => {
      setLoading(false);
      window.location.href = '/';
    }, 1200);
  };

  return (
    <div className="min-h-screen bg-bento-bg flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Decorative Element */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-mint/5 rounded-full blur-[120px] pointer-events-none" />
      
      <AnimatedCard className="w-full max-w-[420px] !p-12 relative z-10 backdrop-blur-2xl border-white/5 shadow-2xl">
        <SystemStatus />

        <div className="flex flex-col gap-1 mb-10">
          <div className="flex items-center gap-3">
             <div className="w-10 h-10 rounded-xl bg-mint flex items-center justify-center">
                <svg className="w-6 h-6 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
             </div>
             <div>
                <h1 className="text-2xl font-black tracking-tight text-white m-0 leading-none">
                  BS_OPT
                </h1>
                <span className="text-[10px] text-mint font-bold uppercase tracking-[0.2em] opacity-80">v4.2.0_ENGINE</span>
             </div>
          </div>
          <p className="text-xs text-white/40 font-medium mt-3 tracking-wide">
            Institutional Grade Quantitative Intelligence Terminal
          </p>
        </div>

        <form onSubmit={signIn} className="space-y-6">
          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-bold animate-shake">
              {error}
            </div>
          )}

          <div className="flex flex-col gap-5">
            <InputField 
              label="QUANT_IDENTITY"
              type="email"
              value={email}
              onChange={setEmail}
              placeholder="id@bsopt.pro"
              icon={
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.206" />
                </svg>
              }
            />

            <InputField 
              label="SECURE_KEY"
              type="password"
              value={password}
              onChange={setPassword}
              placeholder="••••••••"
              icon={
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              }
            />

            <button
              type="submit"
              disabled={loading || !email || !password}
              className="w-full py-4 rounded-xl bg-mint text-black font-black text-sm tracking-[0.05em] uppercase hover:bg-teal-400 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-20 disabled:grayscale disabled:scale-100 flex items-center justify-center gap-3 relative overflow-hidden group shadow-[0_0_20px_-5px_rgba(0,255,163,0.5)]"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-shimmer" />
              {loading ? (
                <div className="w-5 h-5 border-2 border-black/30 border-t-black rounded-full animate-spin" />
              ) : (
                <>
                  INITIALIZE_ACCESS
                  <svg className="w-4 h-4 translate-x-0 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </form>

        <div className="grid grid-cols-2 gap-4 mt-10">
           <button className="flex items-center justify-center gap-2 py-3 bg-white/5 border border-white/10 rounded-xl text-white/60 text-xs font-bold hover:bg-white/10 transition-colors">
              SSO_ACCESS
           </button>
           <button className="flex items-center justify-center gap-2 py-3 bg-white/5 border border-white/10 rounded-xl text-white/60 text-xs font-bold hover:bg-white/10 transition-colors">
              OAUTH_SYNC
           </button>
        </div>

        <div className="flex items-center justify-center gap-2 mt-8">
          <span className="text-[10px] text-white/20 font-bold uppercase tracking-widest">New operative?</span>
          <a href="/signup" className="text-[10px] text-mint font-black uppercase tracking-widest no-underline border-b border-mint/20 hover:border-mint transition-colors">
            Request_Vault_Entry
          </a>
        </div>
      </AnimatedCard>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 opacity-20 hover:opacity-50 transition-opacity">
         <span className="text-[9px] text-white font-bold tracking-[0.3em] uppercase">Security_Protocol_256_Active</span>
      </div>
    </div>
  );
}
