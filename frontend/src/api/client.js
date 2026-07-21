async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      msg = body.detail ?? body.message ?? msg
    } catch {
      // ignore parse error; use status text
    }
    throw new Error(msg)
  }
  return res.json()
}

export const recommend = (prompt, opts = {}) =>
  apiFetch('/recommend', {
    method: 'POST',
    body: JSON.stringify({ prompt, ...opts }),
  })

export const getCacheStatus = () => apiFetch('/cache/status')

export const triggerCrawl = (base_model, mode = 'full', source = 'civitai') =>
  apiFetch('/cache/crawl', {
    method: 'POST',
    body: JSON.stringify({ base_model, mode, source }),
  })

export const importTensorArt = async (file) => {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/catalog/import/tensorart', { method: 'POST', body: form })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try { msg = (await res.json()).detail ?? msg } catch { /* ignore */ }
    throw new Error(msg)
  }
  return res.json()
}
