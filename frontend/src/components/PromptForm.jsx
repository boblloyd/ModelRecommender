import { useState } from 'react'

const KNOWN_BASE_MODELS = ['Flux.1 D', 'Flux.1 S', 'SDXL 1.0', 'SD 1.5', 'Pony']

export default function PromptForm({ onSubmit, loading }) {
  const [prompt, setPrompt]           = useState('')
  const [baseModel, setBaseModel]     = useState('Flux.1 D')
  const [maxLoras, setMaxLoras]       = useState(5)
  const [llmReasoning, setLlmReasoning] = useState(true)
  const [nsfwFilter, setNsfwFilter]   = useState(false)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!prompt.trim() || loading) return
    onSubmit({ prompt: prompt.trim(), baseModel, maxLoras, llmReasoning, nsfwFilter })
  }

  return (
    <form className="prompt-form card" onSubmit={handleSubmit}>
      <div className="form-group">
        <label className="form-label" htmlFor="prompt-input">Prompt</label>
        <textarea
          id="prompt-input"
          className="form-textarea"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Describe the image you want to generate…"
          rows={3}
          required
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="form-label" htmlFor="base-model">Base Model</label>
          <select
            id="base-model"
            className="form-select"
            value={baseModel}
            onChange={e => setBaseModel(e.target.value)}
          >
            {KNOWN_BASE_MODELS.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="max-loras">Max LoRAs</label>
          <input
            id="max-loras"
            type="number"
            className="form-input"
            value={maxLoras}
            onChange={e => setMaxLoras(Number(e.target.value))}
            min={1}
            max={15}
          />
        </div>

        <div className="form-toggles">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={llmReasoning}
              onChange={e => setLlmReasoning(e.target.checked)}
            />
            <span>LLM analysis</span>
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={nsfwFilter}
              onChange={e => setNsfwFilter(e.target.checked)}
            />
            <span>SFW only</span>
          </label>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading || !prompt.trim()}>
          {loading ? (
            <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />Searching…</>
          ) : 'Get Recommendations'}
        </button>
      </div>
    </form>
  )
}
