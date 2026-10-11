import React, { useState } from 'react'
import './App.css'

interface SourceCitation {
  source_id: string
  chunk_id: string
  document_id: string
  section_title: string | null
  source_url: string
  score: number
}

interface AskResponse {
  query: string
  answer: string
  sources: SourceCitation[]
  total_sources: number
  model: string
}

export function App() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<AskResponse | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/v1/ask/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: topK })
      })

      const json = await res.json()

      if (!res.ok) {
        const detailMsg = typeof json.detail === 'string'
          ? json.detail
          : (Array.isArray(json.detail) ? json.detail.map((d: any) => d.msg).join(', ') : 'Failed to generate answer')
        throw new Error(detailMsg)
      }

      setData(json)
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  // Safe formatting of answer with markdown code blocks and paragraphs
  const renderFormattedAnswer = (text: string) => {
    // Split by code blocks
    const parts = text.split(/(```[\s\S]*?```)/g)

    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const lines = part.slice(3, -3).trim().split('\n')
        const firstLine = lines[0].trim()
        const isLang = /^[a-zA-Z0-9_-]+$/.test(firstLine)
        const lang = isLang ? firstLine : ''
        const codeContent = isLang ? lines.slice(1).join('\n') : lines.join('\n')

        return (
          <div key={index} className="code-block-wrapper">
            {lang && <span className="code-lang">{lang}</span>}
            <pre className="code-block">
              <code>{codeContent}</code>
            </pre>
          </div>
        )
      }

      // Paragraph / line blocks
      return (
        <div key={index} className="prose-block">
          {part.split('\n\n').map((paragraph, pIdx) => {
            if (!paragraph.trim()) return null
            return (
              <p key={pIdx} className="answer-paragraph">
                {paragraph.split('\n').map((line, lIdx) => (
                  <React.Fragment key={lIdx}>
                    {lIdx > 0 && <br />}
                    {renderInlineFormatting(line)}
                  </React.Fragment>
                ))}
              </p>
            )
          })}
        </div>
      )
    })
  }

  // Inline formatting for code `...` and citations [S1]
  const renderInlineFormatting = (line: string) => {
    const tokens = line.split(/(`[^`]+`|\[S\d+\])/g)
    return tokens.map((token, idx) => {
      if (token.startsWith('`') && token.endsWith('`')) {
        return <code key={idx} className="inline-code">{token.slice(1, -1)}</code>
      }
      if (/^\[S\d+\]$/.test(token)) {
        const sourceId = token.slice(1, -1)
        return (
          <span
            key={idx}
            className="citation-badge"
            title={`Referenced source ${sourceId}`}
          >
            {token}
          </span>
        )
      }
      return token
    })
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="badge-tag">FastAPI &amp; Python Technical Assistant</div>
        <h1 className="app-title">DevDocs RAG</h1>
        <p className="app-subtitle">
          Retrieval-Augmented technical documentation search powered by FAISS and Groq LLM
        </p>
      </header>

      <main className="app-main">
        <form className="query-form" onSubmit={handleSubmit}>
          <div className="input-group">
            <textarea
              className="query-input"
              rows={3}
              placeholder="Ask a technical question (e.g. How do I define a POST endpoint in FastAPI?)..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
          </div>

          <div className="form-controls">
            <div className="topk-control">
              <label htmlFor="top-k-select">Context Chunks:</label>
              <select
                id="top-k-select"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={loading}
              >
                <option value={3}>3 chunks</option>
                <option value={5}>5 chunks</option>
                <option value={8}>8 chunks (default)</option>
                <option value={12}>12 chunks</option>
              </select>
            </div>

            <button
              type="submit"
              className="submit-btn"
              disabled={!query.trim() || loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Generating Grounded Answer...
                </>
              ) : (
                'Ask DevDocs'
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-banner" role="alert">
            <div className="error-title">Generation Error</div>
            <p className="error-message">{error}</p>
          </div>
        )}

        {data && (
          <section className="results-container">
            <div className="answer-card">
              <div className="card-header">
                <h2>Grounded Answer</h2>
                <span className="model-badge">Model: {data.model}</span>
              </div>
              <div className="answer-content">
                {renderFormattedAnswer(data.answer)}
              </div>
            </div>

            {data.sources && data.sources.length > 0 && (
              <div className="sources-section">
                <h3 className="sources-title">
                  Retrieved Sources ({data.total_sources})
                </h3>
                <div className="sources-grid">
                  {data.sources.map((src) => (
                    <article key={src.source_id} className="source-card">
                      <div className="source-header">
                        <span className="source-id">{src.source_id}</span>
                        <span className="source-score" title="Vector similarity score">
                          Score: {src.score.toFixed(3)}
                        </span>
                      </div>
                      <h4 className="source-title">
                        {src.section_title || 'General Documentation'}
                      </h4>
                      <a
                        href={src.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        View Official Documentation &rarr;
                      </a>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>DevDocs RAG &bull; Grounded on official technical documentation &bull; Milestone 6</p>
      </footer>
    </div>
  )
}

export default App
