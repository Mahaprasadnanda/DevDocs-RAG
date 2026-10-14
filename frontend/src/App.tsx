import React, { useState } from 'react'
import './App.css'
import { MarkdownAnswer } from './MarkdownAnswer'
import { formatComponentLabel } from './markdownUtils'

interface SourceCitation {
  source_id: string
  chunk_id: string
  document_id: string
  section_title: string | null
  source_url: string
  score: number
  technology?: string
  doc_version?: string
  ecosystem_component?: string
}

interface AskResponse {
  query: string
  answer: string
  sources: SourceCitation[]
  total_sources: number
  model: string
  detected_technology?: string
}

interface LikelyCause {
  cause: string
  explanation: string
  citations: string[]
}

interface SuggestedFix {
  fix: string
  explanation: string
  citations: string[]
}

interface DebugResponse {
  error_summary: string
  likely_causes: LikelyCause[]
  diagnostic_checks: string[]
  suggested_fixes: SuggestedFix[]
  sources: SourceCitation[]
  total_sources: number
  uncertainty_notes: string[]
  model: string
  detected_technology?: string
}

type ActiveTab = 'ask' | 'debug'

const TECHNOLOGIES = [
  { id: 'auto', name: 'Auto-detect Ecosystem' },
  { id: 'fastapi', name: 'FastAPI' },
  { id: 'react', name: 'React' },
  { id: 'django', name: 'Django' },
  { id: 'flask', name: 'Flask' },
  { id: 'pydantic', name: 'Pydantic' },
  { id: 'spring-boot', name: 'Spring Boot' },
  { id: 'nodejs', name: 'Node.js' },
  { id: 'go', name: 'Go (Golang)' }
]

export function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('ask')

  // Ask Mode state
  const [query, setQuery] = useState('')
  const [askTech, setAskTech] = useState('auto')
  const [askTopK, setAskTopK] = useState(8)
  const [askLoading, setAskLoading] = useState(false)
  const [askError, setAskError] = useState<string | null>(null)
  const [askData, setAskData] = useState<AskResponse | null>(null)

  // Debug Mode state
  const [errorMessage, setErrorMessage] = useState('')
  const [debugTech, setDebugTech] = useState('auto')
  const [framework, setFramework] = useState('')
  const [language, setLanguage] = useState('')
  const [context, setContext] = useState('')
  const [debugTopK, setDebugTopK] = useState(8)
  const [debugLoading, setDebugLoading] = useState(false)
  const [debugError, setDebugError] = useState<string | null>(null)
  const [debugData, setDebugData] = useState<DebugResponse | null>(null)

  const handleAskTechChange = (tech: string) => {
    setAskTech(tech)
    setAskData(null)
    setAskError(null)
  }

  const handleDebugTechChange = (tech: string) => {
    setDebugTech(tech)
    setDebugData(null)
    setDebugError(null)
  }

  const formatErrorMessage = (res: Response, json: any): string => {
    if (res.status === 422) {
      if (typeof json?.detail === 'string') return json.detail
      if (Array.isArray(json?.detail)) {
        return json.detail.map((d: any) => d.msg || 'Invalid field').join(', ')
      }
      return 'Input validation failed. Please check your query or input.'
    }
    if (res.status === 502) {
      return 'Upstream AI provider error: The Groq LLM service reported an upstream error. Please retry shortly.'
    }
    if (res.status === 503) {
      return 'Service unavailable: Vector search index or LLM configuration is unavailable.'
    }
    if (res.status === 504) {
      return 'Request timeout: Upstream AI provider took too long to respond. Please try again.'
    }
    if (res.status >= 500) {
      return 'Internal server error: An unexpected error occurred while processing your request.'
    }
    return json?.detail || 'An unexpected error occurred.'
  }

  const handleAskSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || askLoading) return

    setAskLoading(true)
    setAskError(null)
    setAskData(null) // Prevents displaying stale response/citations while loading or on error

    try {
      const payload: any = { query: query.trim(), top_k: askTopK }
      if (askTech !== 'auto') {
        payload.technology = askTech
      }

      const res = await fetch('/api/v1/ask/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const json = await res.json()

      if (!res.ok) {
        throw new Error(formatErrorMessage(res, json))
      }

      setAskData(json)
    } catch (err: any) {
      if (err.name === 'TypeError' || err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError')) {
        setAskError('Network connection error: Unable to reach backend server. Please verify connection.')
      } else {
        setAskError(err.message || 'An unexpected error occurred.')
      }
    } finally {
      setAskLoading(false)
    }
  }

  const handleDebugSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!errorMessage.trim() || debugLoading) return

    setDebugLoading(true)
    setDebugError(null)
    setDebugData(null) // Prevents displaying stale diagnosis/citations while loading or on error

    try {
      const payload: any = {
        error_message: errorMessage.trim(),
        framework: framework.trim() || undefined,
        language: language.trim() || undefined,
        context: context.trim() || undefined,
        top_k: debugTopK
      }
      if (debugTech !== 'auto') {
        payload.technology = debugTech
      }

      const res = await fetch('/api/v1/debug/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const json = await res.json()

      if (!res.ok) {
        throw new Error(formatErrorMessage(res, json))
      }

      setDebugData(json)
    } catch (err: any) {
      if (err.name === 'TypeError' || err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError')) {
        setDebugError('Network connection error: Unable to reach backend server. Please verify connection.')
      } else {
        setDebugError(err.message || 'An unexpected error occurred.')
      }
    } finally {
      setDebugLoading(false)
    }
  }



  return (
    <div className="app-container">
      <header className="app-header">
        <div className="badge-tag">Documentation Intelligence</div>
        <h1 className="app-title">DevDocs RAG</h1>
        <p className="app-subtitle">
          Documentation-grounded assistant supporting 8 major frameworks with verified citations and diagnostic guidance.
        </p>

        {/* Tab Navigation */}
        <div className="mode-tabs">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'ask' ? 'active' : ''}`}
            onClick={() => setActiveTab('ask')}
          >
            Ask DevDocs
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'debug' ? 'active' : ''}`}
            onClick={() => setActiveTab('debug')}
          >
            Debug Assistant
          </button>
        </div>
      </header>

      <main className="app-main">
        {/* ASK DEVDOCS TAB */}
        {activeTab === 'ask' && (
          <>
            <form className="query-form" onSubmit={handleAskSubmit}>
              <div className="input-group">
                <textarea
                  className="query-input"
                  rows={3}
                  placeholder="Ask a technical question (e.g. How do I use useState in React? or How to configure Spring Boot properties?)..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={askLoading}
                />
              </div>

              <div className="form-controls">
                <div className="filter-group">
                  <div className="tech-control">
                    <label htmlFor="ask-tech-select">Technology:</label>
                    <select
                      id="ask-tech-select"
                      className="select-dropdown"
                      value={askTech}
                      onChange={(e) => handleAskTechChange(e.target.value)}
                      disabled={askLoading}
                    >
                      {TECHNOLOGIES.map((tech) => (
                        <option key={tech.id} value={tech.id}>{tech.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="topk-control">
                    <label htmlFor="ask-top-k-select">Context Chunks:</label>
                    <select
                      id="ask-top-k-select"
                      className="select-dropdown"
                      value={askTopK}
                      onChange={(e) => setAskTopK(Number(e.target.value))}
                      disabled={askLoading}
                    >
                      <option value={3}>3 chunks</option>
                      <option value={5}>5 chunks</option>
                      <option value={8}>8 chunks (default)</option>
                      <option value={12}>12 chunks</option>
                    </select>
                  </div>
                </div>

                <button
                  type="submit"
                  className="submit-btn"
                  disabled={!query.trim() || askLoading}
                >
                  {askLoading ? (
                    <>
                      <span className="spinner"></span> Generating Grounded Answer...
                    </>
                  ) : (
                    'Ask DevDocs'
                  )}
                </button>
              </div>
            </form>

            {askError && (
              <div className="error-banner" role="alert">
                <div className="error-title">Generation Error</div>
                <p className="error-message">{askError}</p>
              </div>
            )}

            {askData && (
              <section className="results-container">
                <div className="answer-card">
                  <div className="card-header">
                    <h2>Grounded Answer</h2>
                    <div className="badge-group">
                      {askData.detected_technology && (
                        <span className="tech-badge">Ecosystem: {askData.detected_technology}</span>
                      )}
                      <span className="model-badge">Model: {askData.model}</span>
                    </div>
                  </div>
                  <div className="answer-content">
                    <MarkdownAnswer content={askData.answer} sources={askData.sources} />
                  </div>
                </div>

                {askData.total_sources === 0 && (
                  <div className="evidence-notice-banner">
                    <strong>Limited Documentation Evidence:</strong> No indexed documentation sources met the relevance threshold for this query. The response is strictly bounded by available evidence.
                  </div>
                )}

                {askData.sources && askData.sources.length > 0 && (
                  <div className="sources-section">
                    <h3 className="sources-title">
                      Retrieved Sources ({askData.total_sources})
                    </h3>
                    <div className="sources-grid">
                      {askData.sources.map((src) => (
                        <article id={`source-${src.source_id}`} key={src.source_id} className="source-card">
                          <div className="source-header">
                            <div className="source-meta-left">
                              <span className="source-id">{src.source_id}</span>
                              {src.technology && (
                                <span className="source-tech-tag">
                                  {src.technology}{src.doc_version ? ` v${src.doc_version}` : ''}
                                </span>
                              )}
                              {src.ecosystem_component && (() => {
                                const comp = formatComponentLabel(src.ecosystem_component, src.technology)
                                return (
                                  <span
                                    className={`source-component-tag component-${comp.type}`}
                                    title={`Ecosystem Component: ${comp.label}`}
                                  >
                                    {comp.label}
                                  </span>
                                )
                              })()}
                            </div>
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
          </>
        )}

        {/* DEBUG ASSISTANT TAB */}
        {activeTab === 'debug' && (
          <>
            <form className="query-form" onSubmit={handleDebugSubmit}>
              <div className="input-group">
                <label className="input-label" htmlFor="error-msg-input">
                  Error Message or Stack Trace <span className="required-star">*</span>
                </label>
                <textarea
                  id="error-msg-input"
                  className="query-input code-font"
                  rows={5}
                  placeholder="Paste error message or traceback here (e.g. org.springframework.beans.factory.NoSuchBeanDefinitionException, or panic: runtime error...)..."
                  value={errorMessage}
                  onChange={(e) => setErrorMessage(e.target.value)}
                  disabled={debugLoading}
                />
              </div>

              <div className="input-grid">
                <div className="input-group">
                  <label className="input-label" htmlFor="debug-tech-select">
                    Target Technology
                  </label>
                  <select
                    id="debug-tech-select"
                    className="select-dropdown full-width"
                    value={debugTech}
                    onChange={(e) => handleDebugTechChange(e.target.value)}
                    disabled={debugLoading}
                  >
                    {TECHNOLOGIES.map((tech) => (
                      <option key={tech.id} value={tech.id}>{tech.name}</option>
                    ))}
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label" htmlFor="framework-input">
                    Framework / Subsystem (Optional)
                  </label>
                  <input
                    id="framework-input"
                    type="text"
                    className="text-input"
                    placeholder="e.g. Spring Web, Express, Goroutines"
                    value={framework}
                    onChange={(e) => setFramework(e.target.value)}
                    disabled={debugLoading}
                  />
                </div>

                <div className="input-group">
                  <label className="input-label" htmlFor="language-input">
                    Language (Optional)
                  </label>
                  <input
                    id="language-input"
                    type="text"
                    className="text-input"
                    placeholder="e.g. Java, Go, TypeScript, Python"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    disabled={debugLoading}
                  />
                </div>
              </div>

              <div className="input-group">
                <label className="input-label" htmlFor="context-input">
                  Optional Context (What were you trying to do?)
                </label>
                <input
                  id="context-input"
                  type="text"
                  className="text-input"
                  placeholder="e.g. Injected service bean in Spring Boot REST controller"
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  disabled={debugLoading}
                />
              </div>

              <div className="form-controls">
                <div className="topk-control">
                  <label htmlFor="debug-top-k-select">Context Chunks:</label>
                  <select
                    id="debug-top-k-select"
                    className="select-dropdown"
                    value={debugTopK}
                    onChange={(e) => setDebugTopK(Number(e.target.value))}
                    disabled={debugLoading}
                  >
                    <option value={3}>3 chunks</option>
                    <option value={5}>5 chunks</option>
                    <option value={8}>8 chunks (default)</option>
                    <option value={12}>12 chunks</option>
                  </select>
                </div>

                <button
                  type="submit"
                  className="submit-btn debug-btn"
                  disabled={!errorMessage.trim() || debugLoading}
                >
                  {debugLoading ? (
                    <>
                      <span className="spinner"></span> Diagnosing Error...
                    </>
                  ) : (
                    'Diagnose Error'
                  )}
                </button>
              </div>
            </form>

            {debugError && (
              <div className="error-banner" role="alert">
                <div className="error-title">Diagnosis Error</div>
                <p className="error-message">{debugError}</p>
              </div>
            )}

            {debugData && (
              <section className="results-container">
                {/* Error Summary Card */}
                <div className="answer-card">
                  <div className="card-header">
                    <h2>Diagnosis Summary</h2>
                    <div className="badge-group">
                      {debugData.detected_technology && (
                        <span className="tech-badge">Ecosystem: {debugData.detected_technology}</span>
                      )}
                      <span className="model-badge">Model: {debugData.model}</span>
                    </div>
                  </div>
                  <div className="diagnosis-summary">
                    <MarkdownAnswer content={debugData.error_summary} sources={debugData.sources} />
                  </div>
                </div>

                {/* Likely Causes */}
                {debugData.likely_causes && debugData.likely_causes.length > 0 && (
                  <div className="diagnosis-section">
                    <h3 className="section-title">Likely Causes</h3>
                    <div className="causes-list">
                      {debugData.likely_causes.map((cause, idx) => (
                        <div key={idx} className="cause-card">
                          <div className="cause-header">
                            <h4 className="cause-name">
                              <MarkdownAnswer content={cause.cause} sources={debugData.sources} inline={true} />
                            </h4>
                            {cause.citations && cause.citations.length > 0 && (
                              <div className="citation-pills">
                                {cause.citations.map((c, cIdx) => (
                                  <span key={cIdx} className="citation-badge">[{c}]</span>
                                ))}
                              </div>
                            )}
                          </div>
                          <div className="cause-explanation">
                            <MarkdownAnswer content={cause.explanation} sources={debugData.sources} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Diagnostic Checks */}
                {debugData.diagnostic_checks && debugData.diagnostic_checks.length > 0 && (
                  <div className="diagnosis-section">
                    <h3 className="section-title">Diagnostic Checks</h3>
                    <ol className="checks-list">
                      {debugData.diagnostic_checks.map((check, idx) => (
                        <li key={idx} className="check-item">
                          <span><MarkdownAnswer content={check} sources={debugData.sources} /></span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Suggested Fixes */}
                {debugData.suggested_fixes && debugData.suggested_fixes.length > 0 && (
                  <div className="diagnosis-section">
                    <h3 className="section-title">Suggested Fixes</h3>
                    <div className="fixes-list">
                      {debugData.suggested_fixes.map((fix, idx) => (
                        <div key={idx} className="fix-card">
                          <div className="fix-header">
                            <h4 className="fix-name">
                              <MarkdownAnswer content={fix.fix} sources={debugData.sources} inline={true} />
                            </h4>
                            {fix.citations && fix.citations.length > 0 && (
                              <div className="citation-pills">
                                {fix.citations.map((c, cIdx) => (
                                  <span key={cIdx} className="citation-badge">[{c}]</span>
                                ))}
                              </div>
                            )}
                          </div>
                          <div className="fix-explanation">
                            <MarkdownAnswer content={fix.explanation} sources={debugData.sources} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Uncertainty Notes */}
                {debugData.uncertainty_notes && debugData.uncertainty_notes.length > 0 && (
                  <div className="uncertainty-box">
                    <div className="uncertainty-title">Documentation &amp; Diagnostic Scope Notes</div>
                    <ul className="uncertainty-list">
                      {debugData.uncertainty_notes.map((note, idx) => (
                        <li key={idx}><MarkdownAnswer content={note} sources={debugData.sources} inline={true} /></li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Retrieved Sources */}
                {debugData.sources && debugData.sources.length > 0 && (
                  <div className="sources-section">
                    <h3 className="sources-title">
                      Verified Documentation Citations ({debugData.total_sources})
                    </h3>
                    <div className="sources-grid">
                      {debugData.sources.map((src) => (
                        <article id={`source-${src.source_id}`} key={src.source_id} className="source-card">
                          <div className="source-header">
                            <div className="source-meta-left">
                              <span className="source-id">{src.source_id}</span>
                              {src.technology && (
                                <span className="source-tech-tag">
                                  {src.technology}{src.doc_version ? ` v${src.doc_version}` : ''}
                                </span>
                              )}
                              {src.ecosystem_component && (() => {
                                const comp = formatComponentLabel(src.ecosystem_component, src.technology)
                                return (
                                  <span
                                    className={`source-component-tag component-${comp.type}`}
                                    title={`Ecosystem Component: ${comp.label}`}
                                  >
                                    {comp.label}
                                  </span>
                                )
                              })()}
                            </div>
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
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>DevDocs RAG &bull; Grounded on official technical documentation &bull; Eight Supported Technologies</p>
      </footer>
    </div>
  )
}

export default App
