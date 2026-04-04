import { SignUp } from '../../components/auth/SignUp';

const GREEK_SYMBOLS = ['Δ', 'Γ', 'Θ', 'Ρ', 'Σ', 'Λ', 'Φ', 'Ψ', '∑', '∂'];

const DecorativeBg: React.FC = () => (
  <div className="absolute inset-0 pointer-events-none overflow-hidden select-none">
    {/* Greek Symbols Background */}
    {GREEK_SYMBOLS.map((sym, i) => (
      <div
        key={i}
        className="absolute font-serif font-black"
        style={{
          color: `rgba(16, 185, 129, ${0.03 + (i % 3) * 0.012})`,
          fontSize: `${48 + (i % 4) * 24}px`,
          top: `${(i * 13 + 5) % 85}%`,
          left: `${(i * 17 + 3) % 90}%`,
          filter: 'blur(0.5px)',
          transform: `rotate(${(i * 15) % 360}deg)`,
        }}
      >
        {sym}
      </div>
    ))}
    
    {/* Background Glows */}
    <div className="absolute top-[8%] right-[10%] w-[400px] h-[400px] bg-mint/5 blur-[100px]" />
    <div className="absolute bottom-[12%] left-[5%] w-[350px] h-[350px] bg-teal/5 blur-[80px]" />
    <div className="absolute top-[50%] left-[42%] w-[280px] h-[280px] bg-purple-500/3 blur-[70px]" />
    
    {/* Tactical Grid */}
    <div className="absolute inset-0 bg-bento-grid bg-[length:48px_48px] opacity-[0.03]" />
  </div>
);

export default function SignUpPage() {
  return (
    <div className="min-h-screen bg-bento-bg flex items-center justify-center p-4 relative overflow-hidden">
      <DecorativeBg />
      <div className="relative z-10 w-full">
        <SignUp />
      </div>
    </div>
  );
}
