import data from './data/replays.json'

type Point = [x: number, y: number]
type Pose = [x: number, y: number, angle: number]

export type Arena = {
  size: number
  pusherRadius: number
  blockOutline: Point[]
  target: Pose
}

export type Frame = {
  pusher: Point
  block: Pose
  coverage: number
}

export type Decision = {
  movement: number
  kind: 'extend' | 'trim' | 'interrupt'
}

export type Attempt = {
  success: boolean
  frames: Frame[]
  predictionSteps: number[]
  decisions: Decision[]
}

export type ReplayCase = {
  scene: number
  draw: number
  baseline: Attempt
  adaptive: Attempt
}

// The verified export supplies fixed-length tuples and known decision kinds.
export const replayData = data as unknown as { arena: Arena; cases: ReplayCase[] }
