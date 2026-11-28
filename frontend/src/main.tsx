import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { EvaluationResults } from './EvaluationResults'
import { Replay } from './Replay'
import { replayData } from './replay-data'
import './styles.css'

function App() {
  const [selectedCase, setSelectedCase] = useState(0)
  const replay = replayData.cases[selectedCase]

  return (
    <main className="page">
      <div className="intro">
        <header className="page-header">
          <h1>PushT experiment replay</h1>
          <p>Watch two recorded runs push the same T-shaped block toward its target.</p>
        </header>

        <div className="case-picker" role="group" aria-label="Recorded example">
          {replayData.cases.map((item, index) => (
            <button
              key={item.scene}
              type="button"
              aria-pressed={selectedCase === index}
              onClick={() => setSelectedCase(index)}
            >
              <strong>{item.adaptive.success ? 'Gained success' : 'Lost baseline success'}</strong>
              <span>Scene {item.scene} · Draw {item.draw}</span>
            </button>
          ))}
        </div>
      </div>

      <Replay key={replay.scene} arena={replayData.arena} replay={replay} />
      <EvaluationResults />
    </main>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
