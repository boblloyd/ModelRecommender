import { useState, useEffect, useCallback } from 'react'
import { getCacheStatus, triggerCrawl, importTensorArt } from '../api/client'

function formatDate(dateStr) {
  if (!dateStr) return 'Never'
  return new Date(dateStr).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function CatalogPanel() {
  const [status, setStatus]           = useState(null)
  const [statusLoading, setStatusLoading] = useState(true)
  const [statusError, setStatusError] = useState(null)
  const [crawling, setCrawling]       = useState({})   // key: `${model}-${mode}`
  const [crawlMsgs, setCrawlMsgs]     = useState({})   // key: model name
  const [newModel, setNewModel]       = useState('')
  const [newSource, setNewSource]     = useState('civitai')
  const [importing, setImporting]     = useState(false)
  const [importMsg, setImportMsg]     = useState(null)

  const fetchStatus = useCallback(async () => {
    setStatusLoading(true)
    try {
      const data = await getCacheStatus()
      setStatus(data)
      setStatusError(null)
    } catch (err) {
      setStatusError(err.message)
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => { fetchStatus() }, [fetchStatus])

  const startCrawl = async (baseModel, mode = 'full', source = 'civitai') => {
    const key = `${baseModel}-${mode}`
    setCrawling(prev => ({ ...prev, [key]: true }))
    setCrawlMsgs(prev => ({ ...prev, [baseModel]: null }))
    try {
      const result = await triggerCrawl(baseModel, mode, source)
      const msg = result.job_name
        ? `Job started: ${result.job_name}`
        : result.message ?? 'Crawl dispatched.'
      setCrawlMsgs(prev => ({ ...prev, [baseModel]: { text: msg, isError: false } }))
      setTimeout(fetchStatus, 4000)
    } catch (err) {
      setCrawlMsgs(prev => ({ ...prev, [baseModel]: { text: err.message, isError: true } }))
    } finally {
      setCrawling(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleAddModel = async (e) => {
    e.preventDefault()
    if (!newModel.trim()) return
    await startCrawl(newModel.trim(), 'full', newSource)
    setNewModel('')
  }

  const handleImport = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    setImportMsg(null)
    try {
      const result = await importTensorArt(file)
      setImportMsg({
        text: `Imported ${result.models_imported ?? result.count ?? '?'} models from ${file.name}.`,
        isError: false,
      })
      fetchStatus()
    } catch (err) {
      setImportMsg({ text: err.message, isError: true })
    } finally {
      setImporting(false)
      e.target.value = ''
    }
  }

  const baseModels = status?.base_models ?? []

  return (
    <div className="catalog">

      {/* ── Catalog status ── */}
      <section className="catalog-section">
        <div className="catalog-section-header">
          <h2 className="section-title">Catalog Status</h2>
          {status != null && (
            <span className="total-count">
              {status.total_models_cached.toLocaleString()} models total
            </span>
          )}
          <button className="btn btn-secondary btn-sm" onClick={fetchStatus} disabled={statusLoading}>
            {statusLoading ? '…' : 'Refresh'}
          </button>
        </div>

        {statusError && <div className="error-banner">{statusError}</div>}

        {!statusLoading && baseModels.length === 0 && !statusError && (
          <p className="text-muted">
            No base models indexed yet. Add one below to start crawling.
          </p>
        )}

        {baseModels.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table className="catalog-table">
              <thead>
                <tr>
                  <th>Base Model</th>
                  <th>Status</th>
                  <th>Models</th>
                  <th>Last Crawled</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {baseModels.map(bm => (
                  <tr key={bm.base_model_name}>
                    <td className="td-name">{bm.base_model_name}</td>
                    <td>
                      <span className={`badge ${bm.crawl_complete ? 'badge-complete' : 'badge-incomplete'}`}>
                        {bm.crawl_complete ? 'Complete' : 'Incomplete'}
                      </span>
                    </td>
                    <td>{bm.total_models?.toLocaleString() ?? '—'}</td>
                    <td className="td-date">{formatDate(bm.last_crawled)}</td>
                    <td className="td-actions">
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => startCrawl(bm.base_model_name, 'incremental')}
                        disabled={!!crawling[`${bm.base_model_name}-incremental`]}
                        title="Fetch only new models added since the last crawl"
                      >
                        {crawling[`${bm.base_model_name}-incremental`] ? '…' : '↺ Update'}
                      </button>
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() => startCrawl(bm.base_model_name, 'full')}
                        disabled={!!crawling[`${bm.base_model_name}-full`]}
                        title="Full re-crawl — fetches all pages from CivitAI"
                      >
                        {crawling[`${bm.base_model_name}-full`] ? '…' : '↻ Re-crawl'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {Object.entries(crawlMsgs).map(([model, msg]) =>
          msg ? (
            <div key={model} className={`crawl-status-msg${msg.isError ? ' error' : ''}`}>
              <strong>{model}:</strong> {msg.text}
            </div>
          ) : null
        )}
      </section>

      <hr className="divider" />

      {/* ── Add new base model ── */}
      <section className="catalog-section">
        <h2 className="section-title">Add Base Model</h2>
        <p className="text-muted">
          Crawl CivitAI or HuggingFace for a base model that isn't indexed yet.
          This runs as a background job and may take several minutes.
        </p>
        <form className="form-row" onSubmit={handleAddModel}>
          <div className="form-group" style={{ flex: 1, minWidth: 160 }}>
            <label className="form-label" htmlFor="new-model">Base Model Name</label>
            <input
              id="new-model"
              type="text"
              className="form-textarea"
              style={{ height: 'auto', padding: '.5rem .75rem' }}
              value={newModel}
              onChange={e => setNewModel(e.target.value)}
              placeholder="e.g. SDXL 1.0, Pony"
            />
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="new-source">Source</label>
            <select
              id="new-source"
              className="form-select"
              value={newSource}
              onChange={e => setNewSource(e.target.value)}
            >
              <option value="civitai">CivitAI</option>
              <option value="huggingface">HuggingFace</option>
            </select>
          </div>
          <div className="form-group" style={{ justifyContent: 'flex-end' }}>
            <button type="submit" className="btn btn-primary" disabled={!newModel.trim()}>
              Crawl
            </button>
          </div>
        </form>
      </section>

      <hr className="divider" />

      {/* ── TensorArt import ── */}
      <section className="catalog-section">
        <h2 className="section-title">TensorArt Import</h2>
        <p className="text-muted">
          Import a JSON export captured by the TamperMonkey script from TensorArt.
          Imported models appear in recommendation results alongside CivitAI and HuggingFace models.
        </p>
        <div className="form-row" style={{ alignItems: 'center' }}>
          <label className={`btn btn-secondary${importing ? ' disabled' : ''}`} style={{ cursor: importing ? 'not-allowed' : 'pointer' }}>
            {importing ? (
              <><span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />Importing…</>
            ) : 'Choose JSON File'}
            <input
              type="file"
              accept=".json"
              onChange={handleImport}
              disabled={importing}
              style={{ display: 'none' }}
            />
          </label>
        </div>
        {importMsg && (
          <div className={`crawl-status-msg${importMsg.isError ? ' error' : ''}`}>
            {importMsg.text}
          </div>
        )}
      </section>

    </div>
  )
}
