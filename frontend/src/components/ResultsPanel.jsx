import { useState } from 'react'

const IMPACT_BADGE = {
  high:   'badge-high',
  medium: 'badge-medium',
  low:    'badge-low',
}
const IMPACT_LABEL = {
  high: 'HIGH',
  medium: 'MED',
  low: 'LOW',
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <button className="btn btn-sm btn-ghost" onClick={handleCopy} type="button">
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  )
}

function ModelCard({ model, isLora }) {
  const isNotRec = model.recommended === false

  return (
    <div className={`card model-card${isNotRec ? ' model-card-not-rec' : ''}`}>
      <div className="model-card-header">
        <div className="model-card-title">
          <span className="model-name">{model.name}</span>
          {isLora && model.impact && (
            <span className={`badge ${IMPACT_BADGE[model.impact] ?? 'badge-low'}`}>
              {IMPACT_LABEL[model.impact] ?? model.impact.toUpperCase()}
            </span>
          )}
          {isNotRec && (
            <span className="badge badge-not-rec">✗ NOT REC.</span>
          )}
        </div>
        {model.relevance_score != null && (
          <span className="model-score">{model.relevance_score.toFixed(3)}</span>
        )}
      </div>

      {model.civitai_url && (
        <a
          href={model.civitai_url}
          target="_blank"
          rel="noreferrer"
          className="model-url"
        >
          {model.civitai_url}
        </a>
      )}

      {!isLora && (model.recommended_cfg || model.recommended_steps) && (
        <div className="model-settings">
          {[
            model.recommended_cfg    && `CFG ${model.recommended_cfg}`,
            model.recommended_steps  && `Steps ${model.recommended_steps}`,
            model.recommended_sampler,
          ].filter(Boolean).join(' · ')}
        </div>
      )}

      {model.compatibility_note && (
        <p className="model-note">{model.compatibility_note}</p>
      )}

      {isLora && (model.recommended_weight != null || model.trigger_words?.length > 0) && (
        <div className="model-lora-meta">
          {model.recommended_weight != null && (
            <span className="lora-meta-item">
              Weight: <strong>{model.recommended_weight}</strong>
            </span>
          )}
          {model.trigger_words?.length > 0 && (
            <span className="lora-meta-item">
              Triggers: <code className="trigger-words">{model.trigger_words.join(', ')}</code>
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export default function ResultsPanel({ results }) {
  const {
    phase,
    intent,
    checkpoints = [],
    loras = [],
    recommended_combination,
    combination_notes,
    prompt_additions = [],
  } = results

  return (
    <div className="results">
      <div className="results-meta">
        <span className="phase-badge">{phase}</span>
        {intent?.tags?.length > 0 && (
          <span className="intent-tags">
            Tags: {intent.tags.slice(0, 12).join(', ')}
          </span>
        )}
      </div>

      {prompt_additions.length > 0 && (
        <div className="card card-additions">
          <div className="additions-header">
            <span className="section-label">Add to your prompt</span>
            <CopyButton text={prompt_additions.join(', ')} />
          </div>
          <p className="additions-text">{prompt_additions.join(', ')}</p>
        </div>
      )}

      {recommended_combination && (
        <div className="card card-combo">
          <div className="section-label">Recommended Combination</div>
          <div className="combo-name">{recommended_combination}</div>
          {combination_notes && <p className="combo-notes">{combination_notes}</p>}
        </div>
      )}

      {checkpoints.length > 0 && (
        <section className="results-section">
          <h2 className="section-title">
            Checkpoints <span className="count">{checkpoints.length}</span>
          </h2>
          {checkpoints.map(m => (
            <ModelCard key={m.id} model={m} isLora={false} />
          ))}
        </section>
      )}

      {loras.length > 0 && (
        <section className="results-section">
          <h2 className="section-title">
            LoRAs <span className="count">{loras.length}</span>
          </h2>
          {loras.map(m => (
            <ModelCard key={m.id} model={m} isLora={true} />
          ))}
        </section>
      )}

      {checkpoints.length === 0 && loras.length === 0 && (
        <p className="text-muted">
          No models found. Try a broader prompt, or seed the catalog for this base model
          in the <strong>Catalog</strong> tab.
        </p>
      )}
    </div>
  )
}
