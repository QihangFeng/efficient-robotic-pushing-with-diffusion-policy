import { useEffect, useId, useState } from 'react'

import type { Arena, Attempt, Decision, Frame, ReplayCase } from './replay-data'
import { CoverageTimeline } from './CoverageTimeline'

const MOVEMENTS_PER_SECOND = 30
const decisionLabels: Record<Decision['kind'], string> = {
  extend: 'Plan extended',
  trim: 'Queued movements discarded',
  interrupt: 'Plan interrupted',
}

function Board({ arena, frame }: { arena: Arena; frame: Frame }) {
  const gridId = useId()
  const outline = arena.blockOutline.map((point) => point.join(',')).join(' ')
  const [targetX, targetY, targetAngle] = arena.target
  const [blockX, blockY, blockAngle] = frame.block

  return (
    <svg className="board" viewBox={`0 0 ${arena.size} ${arena.size}`} aria-hidden="true">
      <defs>
        <pattern id={gridId} width="64" height="64" patternUnits="userSpaceOnUse">
          <path d="M64 0H0V64" fill="none" stroke="var(--board-grid)" strokeWidth="1.5" />
        </pattern>
      </defs>
      <rect width={arena.size} height={arena.size} fill={`url(#${gridId})`} />
      <polygon className="target" points={outline} transform={`translate(${targetX} ${targetY}) rotate(${targetAngle})`} />
      <polygon className="block" points={outline} transform={`translate(${blockX} ${blockY}) rotate(${blockAngle})`} />
      <circle className="pusher" cx={frame.pusher[0]} cy={frame.pusher[1]} r={arena.pusherRadius} />
    </svg>
  )
}

type RunPanelProps = {
  arena: Arena
  attempt: Attempt
  movement: number
  method: 'baseline' | 'adaptive'
}

function RunPanel({ arena, attempt, movement, method }: RunPanelProps) {
  const lastMovement = attempt.frames.length - 1
  const shownMovement = Math.min(movement, lastMovement)
  const frame = attempt.frames[shownMovement]
  const predictions = attempt.predictionSteps.filter((step) => step <= shownMovement).length
  const decision = attempt.decisions.find((event) => event.movement === movement)
  const finished = movement >= lastMovement
  const status = finished
    ? (attempt.success ? 'Success' : 'Movement limit')
    : null
  const currentEvent = decision ? decisionLabels[decision.kind] :
    attempt.predictionSteps.includes(movement) ? 'Fresh prediction' : null

  return (
    <section className={`run-panel ${method}`}>
      <header className="run-header">
        <h2><span className="series-dot" />{method === 'baseline' ? 'Original fixed-eight' : 'Fine-tuned adaptive'}</h2>
        {status && <span className={`result ${attempt.success ? 'success' : ''}`}>{status}</span>}
      </header>
      <Board arena={arena} frame={frame} />
      <dl className="run-metrics">
        <div><dt>Target coverage</dt><dd>{(frame.coverage * 100).toFixed(1)}<small>%</small></dd></div>
        <div><dt>Prediction calls</dt><dd>{predictions}</dd></div>
        <div className="run-movement"><dt>Movement</dt><dd>{shownMovement}</dd></div>
      </dl>
      <p className="run-event">{currentEvent ?? (finished ? 'Final frame' : movement === 0 ? 'Starting scene' : 'Executing plan')}</p>
    </section>
  )
}

export function Replay({ arena, replay }: { arena: Arena; replay: ReplayCase }) {
  const [movement, setMovement] = useState(0)
  const [playing, setPlaying] = useState(false)
  const lastMovement = Math.max(replay.baseline.frames.length, replay.adaptive.frames.length) - 1
  const finished = movement === lastMovement
  const running = playing && !finished

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => {
      setMovement((current) => Math.min(current + 1, lastMovement))
    }, 1000 / MOVEMENTS_PER_SECOND)
    return () => window.clearInterval(timer)
  }, [running, lastMovement])

  function togglePlayback() {
    if (finished) setMovement(0)
    setPlaying(!running)
  }

  return (
    <>
      <div className="transport">
        <button type="button" className="play-button" onClick={togglePlayback}>
          <svg width="18" height="18" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            {running ? <path d="M5.5 4h3v12h-3zm6 0h3v12h-3z" /> : <path d="M6 3.5 16 10 6 16.5z" />}
          </svg>
          {running ? 'Pause' : finished ? 'Replay' : 'Play'}
        </button>
        <label className="scrubber">
          <span>Movement</span>
          <span className="movement-count"><strong>{movement}</strong> / {lastMovement}</span>
          <input
            type="range"
            min="0"
            max={lastMovement}
            value={movement}
            aria-label="Movement"
            onChange={(event) => {
              setPlaying(false)
              setMovement(event.currentTarget.valueAsNumber)
            }}
          />
        </label>
      </div>

      <div className="runs">
        <RunPanel arena={arena} attempt={replay.baseline} movement={movement} method="baseline" />
        <RunPanel arena={arena} attempt={replay.adaptive} movement={movement} method="adaptive" />
      </div>
      <ul className="legend board-legend" aria-label="Workspace legend">
        <li><span className="swatch target-swatch" />Target</li>
        <li><span className="swatch block-swatch" />T-block</li>
        <li><span className="swatch pusher-swatch" />Pusher</li>
      </ul>

      <CoverageTimeline replay={replay} movement={movement} lastMovement={lastMovement} />
    </>
  )
}
