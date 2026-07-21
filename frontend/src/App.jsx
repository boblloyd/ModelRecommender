import { useState } from 'react'
import PromptForm from './components/PromptForm'
import ResultsPanel from './components/ResultsPanel'
import CatalogPanel from './components/CatalogPanel'
import { recommend } from './api/client'

export default function App() {
  const [tab, setTab]       = useState('recommend')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  const handleSubmit = async ({ prompt, baseModel, maxLoras, llmReasoning, nsfwFilter }) => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await recommend(prompt, {
        base_model:    baseModel,
        max_loras:     maxLoras,
        llm_reasoning: llmReasoning,
        nsfw_filter:   nsfwFilter,
      })
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">Model <span>Recommender</span></div>
        <nav className="tabs">
          <button
            className={`tab${tab === 'recommend' ? ' active' : ''}`}
            onClick={() => setTab('recommend')}
          >
            Recommend
          </button>
          <button
            className={`tab${tab === 'catalog' ? ' active' : ''}`}
            onClick={() => setTab('catalog')}
          >
            Catalog
          </button>
        </nav>
      </header>

      <main className="main">
        {tab === 'recommend' ? (
          <>
            <PromptForm onSubmit={handleSubmit} loading={loading} />
            {error && <div className="error-banner">{error}</div>}
            {results && <ResultsPanel results={results} />}
          </>
        ) : (
          <CatalogPanel />
        )}
      </main>
    </div>
  )
}
