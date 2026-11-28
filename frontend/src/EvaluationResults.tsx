import { methods } from '../../results/robotics-recovery-fixed8/summary.json'

const configurations = [
  ['fixed8', 'Original', 'Fixed eight'],
  ['milestone12', 'Original', 'Adaptive 8/12'],
  ['recovery8', 'Fine-tuned', 'Fixed eight'],
  ['recovery12', 'Fine-tuned', 'Adaptive 8/12'],
] as const

const average = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const report = 'https://github.com/QihangFeng/efficient-robotic-pushing-with-diffusion-policy/blob/main/reports/2026-11-03-robotics-four-configuration-ablation.md'

export function EvaluationResults() {
  return (
    <section className="evaluation" aria-labelledby="evaluation-heading">
      <header className="evaluation-header">
        <div>
          <h2 id="evaluation-heading">Evaluation results</h2>
          <p>40 matched starting scenes, two prediction samples per scene, and 80 attempts per configuration.</p>
        </div>
        <a href={report}>Read the evaluation report</a>
      </header>

      <div className="evaluation-table-wrap" role="region" aria-label="Four-configuration comparison" tabIndex={0}>
        <table className="evaluation-table" aria-describedby="evaluation-metrics">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">Success</th>
              <th scope="col" aria-label="Mean final coverage">Coverage</th>
              <th scope="col" aria-label="Mean prediction calls">Calls</th>
              <th scope="col" aria-label="Mean recorded runtime">Time</th>
            </tr>
          </thead>
          <tbody>
            {configurations.map(([id, policy, execution]) => {
              const result = methods[id]
              const successRate = result.successes / result.attempts * 100

              return (
                <tr key={id} className={id === 'recovery12' ? 'combined' : undefined}>
                  <th scope="row">{policy}<span>{execution}</span></th>
                  <td><strong>{successRate.toFixed(1)}%</strong><span>{result.successes}/{result.attempts}</span></td>
                  <td>{average.format(result.mean_final_coverage * 100)}%</td>
                  <td>{average.format(result.mean_model_queries)}</td>
                  <td>{average.format(result.mean_wall_seconds)}&nbsp;s</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="evaluation-note" id="evaluation-metrics">
        Success requires more than 95% target coverage. Coverage, prediction calls, and runtime are per-attempt means. Runtime spans multiple collection sessions.
      </p>

      <div className="method-notes">
        <div>
          <h3>Adaptive execution</h3>
          <p>After movement eight, camera and path checks decide whether to reuse movements 9–12. With the same fine-tuned weights, adaptive execution used 18.6% fewer predictions per attempt.</p>
        </div>
        <div>
          <h3>Recovery fine-tuning</h3>
          <p>Successful retries and expert demonstrations were used to update the policy’s final convolution block. With fixed-eight execution, observed successes rose from 46 to 56 out of 80.</p>
        </div>
      </div>
    </section>
  )
}
