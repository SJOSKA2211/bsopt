import './App.css'
import { OptionsChain } from './components/OptionsChain'

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>BS-Opt v4.0</h1>
        <p>Next-Generation Quantitative Trading Platform</p>
      </header>
      <main>
        <OptionsChain />
      </main>
    </div>
  )
}

export default App