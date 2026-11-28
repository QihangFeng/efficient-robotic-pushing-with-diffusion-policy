import { useMemo } from 'react'

import type { Attempt, ReplayCase } from './replay-data'

function coveragePoints(attempt: Attempt) {
  return attempt.frames.map((frame, step) => `${step},${100 - frame.coverage * 100}`).join(' ')
}

export function CoverageTimeline({ replay, movement, lastMovement }: {
  replay: ReplayCase
  movement: number
  lastMovement: number
}) {
  const baseline = useMemo(() => coveragePoints(replay.baseline), [replay.baseline])
  const adaptive = useMemo(() => coveragePoints(replay.adaptive), [replay.adaptive])

  return (
    <section className="timeline" aria-labelledby="timeline-heading">
      <div className="timeline-header">
        <h2 id="timeline-heading">Coverage and planning events</h2>
        <ul className="legend series-legend" aria-label="Coverage series">
          <li className="baseline"><span className="series-dot" />Original</li>
          <li className="adaptive"><span className="series-dot" />Adaptive</li>
        </ul>
      </div>
      <div className="timeline-plot">
        <div className="coverage-axis" aria-hidden="true"><span>100%</span><span>50%</span><span>0%</span></div>
        <svg className="coverage-chart" viewBox={`0 0 ${lastMovement} 100`} preserveAspectRatio="none" role="img" aria-label="Recorded target coverage by movement. The dashed line marks the 95 percent threshold.">
          <line className="chart-guide" x1="0" x2={lastMovement} y1="50" y2="50" />
          <line className="chart-threshold" x1="0" x2={lastMovement} y1="5" y2="5" />
          <polyline className="coverage-line baseline" points={baseline} />
          <polyline className="coverage-line adaptive" points={adaptive} />
          <line className="playhead" x1={movement} x2={movement} y1="0" y2="100" />
        </svg>
        <span className="lane-label baseline">Original</span>
        <svg className="event-lane baseline" viewBox={`0 0 ${lastMovement} 24`} preserveAspectRatio="none" role="img" aria-label="Fresh predictions for the original run">
          {replay.baseline.predictionSteps.map((step) => <line key={step} className="prediction-mark" x1={step} x2={step} y1="6" y2="18" />)}
          <line className="playhead" x1={movement} x2={movement} y1="0" y2="24" />
        </svg>
        <span className="lane-label adaptive">Adaptive</span>
        <svg className="event-lane adaptive" viewBox={`0 0 ${lastMovement} 24`} preserveAspectRatio="none" role="img" aria-label="Fresh predictions, plan extensions, and interruptions for the adaptive run">
          {replay.adaptive.predictionSteps.map((step) => <line key={step} className="prediction-mark" x1={step} x2={step} y1="6" y2="18" />)}
          {replay.adaptive.decisions.filter((event) => event.kind !== 'trim').map(({ movement: step, kind }) => (
            <line key={step} className={`${kind}-mark`} x1={step} x2={step} y1="3" y2="21" />
          ))}
          <line className="playhead" x1={movement} x2={movement} y1="0" y2="24" />
        </svg>
        <div className="movement-axis" aria-hidden="true"><span>0</span><span>{Math.floor(lastMovement / 2)}</span><span>{lastMovement} movements</span></div>
      </div>
      <ul className="legend event-legend" aria-label="Event markers">
        <li><span className="marker-swatch prediction-swatch" />Fresh prediction</li>
        <li><span className="marker-swatch extend-swatch" />Plan extended</li>
        <li><span className="marker-swatch interrupt-swatch" />Plan interrupted</li>
        <li><span className="threshold-swatch" />95% coverage</li>
      </ul>
    </section>
  )
}
