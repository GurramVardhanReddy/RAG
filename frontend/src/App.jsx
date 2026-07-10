import { useState, useRef, useCallback } from 'react'

const API = '/api/v1'



// ═══════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════
function scoreClass(val) {
  if (val >= 0.7) return 'high'
  if (val >= 0.4) return 'medium'
  return 'low'
}
function metricQuality(val) {
  if (val >= 0.7) return 'metric-good'
  if (val >= 0.4) return 'metric-medium'
  return 'metric-poor'
}
function pct(val) { return `${(val * 100).toFixed(1)}%` }
function fmt(val, dp = 4) { return typeof val === 'number' ? val.toFixed(dp) : val }


// ═══════════════════════════════════════════════════════════════
// DomainXAIPanel — Drop this component into your App.jsx
// Shows domain detection result + XAI keyword attribution
// after a successful document upload.
//
// USAGE in App.jsx, inside the {result && (...)} block,
// add <DomainXAIPanel xai={result.domain_detection} />
// ═══════════════════════════════════════════════════════════════

const DOMAIN_ICONS = {
  'Finance/Regulatory':  '💰',
  'Legal':               '⚖️',
  'Medical/Biomedical':  '🧬',
  'Scientific/Academic': '🔬',
  'Code/Software':       '💻',
  'E-commerce/Product':  '🛒',
  'General':             '📄',
}

const DOMAIN_COLORS = {
  'Finance/Regulatory':  { bg: '#f0fdf4', border: '#86efac', accent: '#16a34a', badge: '#dcfce7', badgeText: '#15803d' },
  'Legal':               { bg: '#faf5ff', border: '#d8b4fe', accent: '#7c3aed', badge: '#ede9fe', badgeText: '#6d28d9' },
  'Medical/Biomedical':  { bg: '#fff1f2', border: '#fda4af', accent: '#e11d48', badge: '#ffe4e6', badgeText: '#be123c' },
  'Scientific/Academic': { bg: '#eff6ff', border: '#93c5fd', accent: '#2563eb', badge: '#dbeafe', badgeText: '#1d4ed8' },
  'Code/Software':       { bg: '#f8fafc', border: '#94a3b8', accent: '#475569', badge: '#f1f5f9', badgeText: '#334155' },
  'E-commerce/Product':  { bg: '#fff7ed', border: '#fdba74', accent: '#ea580c', badge: '#ffedd5', badgeText: '#c2410c' },
  'General':             { bg: '#f9fafb', border: '#d1d5db', accent: '#6b7280', badge: '#f3f4f6', badgeText: '#374151' },
}

function ConfidenceRing({ value }) {
  const pct = Math.round(value * 100)
  const radius = 28
  const circumference = 2 * Math.PI * radius
  const strokeDash = (value * circumference).toFixed(1)
  const color = value >= 0.8 ? '#16a34a' : value >= 0.5 ? '#d97706' : '#dc2626'

  return (
    <div style={{ position: 'relative', width: 72, height: 72, flexShrink: 0 }}>
      <svg width="72" height="72" viewBox="0 0 72 72" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="36" cy="36" r={radius} fill="none" stroke="var(--surface-alt, #f1f5f9)" strokeWidth="6" />
        <circle
          cx="36" cy="36" r={radius} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={`${strokeDash} ${circumference}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 1s ease' }}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: '1rem', fontWeight: 700, color, lineHeight: 1 }}>{pct}%</span>
      </div>
    </div>
  )
}

function DomainProbabilityBar({ domain, probability, isSelected }) {
  const colors = DOMAIN_COLORS[domain] || DOMAIN_COLORS['General']
  const pct = (probability * 100).toFixed(1)
  return (
    <div style={{ marginBottom: '0.45rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
        <span style={{
          fontSize: '0.78rem',
          fontWeight: isSelected ? 700 : 400,
          color: isSelected ? colors.accent : 'var(--text-secondary, #64748b)',
          display: 'flex', alignItems: 'center', gap: '0.3rem'
        }}>
          {DOMAIN_ICONS[domain]} {domain}
          {isSelected && <span style={{ fontSize: '0.7rem', background: colors.badge, color: colors.badgeText, padding: '1px 6px', borderRadius: 8, fontWeight: 600 }}>Selected</span>}
        </span>
        <span style={{ fontSize: '0.78rem', fontWeight: isSelected ? 700 : 400, color: isSelected ? colors.accent : 'var(--text-tertiary, #94a3b8)' }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: 'var(--surface-alt, #f1f5f9)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(probability * 100, 100)}%`,
          background: isSelected ? colors.accent : '#cbd5e1',
          borderRadius: 4,
          transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  )
}

function KeywordChip({ kw, maxScore }) {
  const normalized = maxScore > 0 ? kw.importance_score / maxScore : 0
  const alpha = Math.max(0.15, normalized)
  return (
    <div
      title={kw.context_snippet || ''}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
        padding: '3px 10px', borderRadius: 20,
        background: `rgba(37, 99, 235, ${alpha * 0.15})`,
        border: `1px solid rgba(37, 99, 235, ${alpha * 0.4})`,
        fontSize: '0.78rem', fontWeight: 600,
        color: `rgba(30, 64, 175, ${0.5 + alpha * 0.5})`,
        cursor: 'default',
        transition: 'transform 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.05)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
    >
      <span>{kw.word}</span>
      <span style={{ opacity: 0.65, fontSize: '0.7rem' }}>×{kw.frequency}</span>
    </div>
  )
}

export function DomainXAIPanel({ xai }) {
  if (!xai) return null

  const colors = DOMAIN_COLORS[xai.detected_domain] || DOMAIN_COLORS['General']
  const icon = DOMAIN_ICONS[xai.detected_domain] || '📄'
  const topKeywords = (xai.top_keywords || []).slice(0, 12)
  const maxScore = topKeywords.length > 0 ? Math.max(...topKeywords.map(k => k.importance_score)) : 1

  // Sort domain probabilities descending
  const sortedDomains = Object.entries(xai.domain_probabilities || {})
    .sort(([, a], [, b]) => b - a)

  return (
    <div style={{
      marginTop: '1.5rem',
      border: `1.5px solid ${colors.border}`,
      borderRadius: 12,
      overflow: 'hidden',
      background: '#fff',
    }}>
      {/* Header */}
      <div style={{
        padding: '0.9rem 1.25rem',
        background: colors.bg,
        borderBottom: `1px solid ${colors.border}`,
        display: 'flex', alignItems: 'center', gap: '0.75rem',
      }}>
        <span style={{ fontSize: '1.4rem' }}>{icon}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: '0.95rem', color: colors.accent }}>
            Domain detected: {xai.detected_domain}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary, #64748b)', marginTop: 2 }}>
            XAI method: {xai.xai_method?.replace(/_/g, ' ')} · {xai.gpt_used ? 'GPT-4o verified' : 'Keyword attribution only'}
          </div>
        </div>
        <ConfidenceRing value={xai.confidence} />
      </div>

      <div style={{ padding: '1.1rem 1.25rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>

        {/* Left: Embedding model + XAI explanation */}
        <div>
          {/* Model selection box */}
          <div style={{
            padding: '0.75rem 1rem',
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 8,
            marginBottom: '1rem',
          }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-tertiary, #94a3b8)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.3rem' }}>
              Embedding model selected
            </div>
            <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', fontWeight: 700, color: colors.accent, wordBreak: 'break-all' }}>
              {xai.embedding_model}
            </div>
            <div style={{ fontSize: '0.73rem', color: 'var(--text-tertiary, #94a3b8)', marginTop: 4 }}>
              {xai.embedding_dimension}d vectors · cosine similarity
            </div>
          </div>

          {/* XAI explanation */}
          <div style={{
            padding: '0.75rem 1rem',
            background: 'var(--surface, #f8fafc)',
            border: '1px solid var(--border, #e2e8f0)',
            borderRadius: 8,
            fontSize: '0.8rem',
            color: 'var(--text-secondary, #64748b)',
            lineHeight: 1.6,
          }}>
            <div style={{ fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-tertiary, #94a3b8)', marginBottom: '0.4rem' }}>
              Why this domain?
            </div>
            {xai.xai_explanation}
          </div>

          {/* GPT-4o reasoning (if used) */}
          {xai.gpt_used && xai.gpt_reasoning && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.75rem 1rem',
              background: '#fefce8',
              border: '1px solid #fde047',
              borderRadius: 8,
              fontSize: '0.8rem',
              color: '#713f12',
              lineHeight: 1.6,
            }}>
              <div style={{ fontWeight: 700, marginBottom: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span>🤖</span> GPT-4o verification reasoning
              </div>
              {xai.gpt_reasoning}
            </div>
          )}
        </div>

        {/* Right: Domain probability bars */}
        <div>
          <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-tertiary, #94a3b8)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.75rem' }}>
            All domain scores
          </div>
          {sortedDomains.map(([domain, prob]) => (
            <DomainProbabilityBar
              key={domain}
              domain={domain}
              probability={prob}
              isSelected={domain === xai.detected_domain}
            />
          ))}
        </div>
      </div>

      {/* Keyword attribution section */}
      {topKeywords.length > 0 && (
        <div style={{
          padding: '0.9rem 1.25rem',
          borderTop: '1px solid var(--border, #e2e8f0)',
          background: 'var(--surface, #f8fafc)',
        }}>
          <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-tertiary, #94a3b8)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>
            🔑 Top triggering keywords — hover for context
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {topKeywords.map((kw, i) => (
              <KeywordChip key={i} kw={kw} maxScore={maxScore} />
            ))}
          </div>
          <div style={{ marginTop: '0.65rem', fontSize: '0.73rem', color: 'var(--text-tertiary, #94a3b8)' }}>
            Opacity and border intensity reflect keyword importance score (TF × IDF weighting). ×N = frequency in document.
          </div>
        </div>
      )}
    </div>
  )
}




// ═══════════════════════════════════════════════
// UPLOAD PAGE
// ═══════════════════════════════════════════════
function UploadPage({ documents, setDocuments }) {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragover, setDragover] = useState(false)
  const fileRef = useRef(null)

  const handleUpload = async (file) => {
    if (!file) return
    setUploading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setResult(data)
      setDocuments(prev => [...prev, { id: data.document_id, name: data.filename }])
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragover(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [])

  return (
    <div>
      <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '1.5rem' }}>Upload document</h2>

      <div
        className={`upload-zone ${dragover ? 'dragover' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragover(true) }}
        onDragLeave={() => setDragover(false)}
        onDrop={onDrop}
      >
        <div className="icon">📄</div>
        <h3>Drop your PDF or DOCX here</h3>
        <p>or click to browse. Supports .pdf, .docx, .txt</p>
        <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" hidden onChange={(e) => handleUpload(e.target.files[0])} />
      </div>

      {uploading && <div className="loading"><div className="spinner" /> Processing document...</div>}

      {error && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--danger-light)', border: '1px solid #fca5a5', borderRadius: 'var(--radius)', color: 'var(--danger)' }}>
          {error}
        </div>
      )}

      {result && (
        <div className="upload-result">
          <h4>✓ {result.filename} uploaded successfully</h4>
          <p style={{ fontSize: '0.85rem', color: '#15803d' }}>Document ID: <code>{result.document_id}</code></p>
          <div className="upload-stats">
            <div className="upload-stat"><div className="value">{result.total_pages}</div><div className="label">Pages</div></div>
            <div className="upload-stat"><div className="value">{result.total_chunks}</div><div className="label">Chunks</div></div>
            <div className="upload-stat"><div className="value">{result.total_paragraphs}</div><div className="label">Paragraphs</div></div>
            <div className="upload-stat"><div className="value">{result.visual_tables || 0}</div><div className="label">Tables</div></div>
            <div className="upload-stat"><div className="value">{result.visual_images || 0}</div><div className="label">Images</div></div>
            <div className="upload-stat"><div className="value">{result.processing_time_ms?.toFixed(0)}ms</div><div className="label">Processing time</div></div>
          </div>
          <DomainXAIPanel xai={result.domain_detection} />
        </div>
        
      )}

      {documents.length > 0 && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="card-header">📚 Uploaded documents</div>
          {documents.map(d => (
            <div key={d.id} style={{ padding: '0.5rem 0', borderBottom: '1px solid var(--surface-alt)', fontSize: '0.9rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>{d.name}</span>
              <code style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{d.id}</code>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════
// EXPLAINABILITY DASHBOARD
// ═══════════════════════════════════════════════
function ExplainabilityDashboard({ data, baselineData, limeData }) {
  const [tab, setTab] = useState('summary')
  const tl = data.three_layer_explainability
  const l1 = tl.layer1_quantitative
  const l2 = tl.layer2_visualization
  const l3 = tl.layer3_expert_evaluation

  const baselineEf = baselineData?.three_layer_explainability?.layer1_quantitative?.explanation_faithfulness

  const tabs = ['summary', 'attribution', 'provenance', 'faithfulness', 'heatmap', 'trace', 'pipeline']
  if (limeData) tabs.splice(5, 0, 'lime')  // Insert LIME tab before trace

  const tabLabels = {
    summary: '📊 Summary', attribution: '🎯 Attribution', provenance: '📍 Provenance',
    faithfulness: '✓ Faithfulness', heatmap: '🔥 Heatmap', lime: '🧪 LIME',
    trace: '🔗 Trace', pipeline: '⚡ Pipeline'
  }

  return (
    <div style={{ marginTop: '1.5rem' }}>
      <div className="explain-tabs">
        {tabs.map(t => (
          <button key={t} className={`explain-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {tabLabels[t]}
          </button>
        ))}
      </div>

      {tab === 'summary' && <SummaryTab tl={tl} l1={l1} limeData={limeData} />}
      {tab === 'attribution' && <AttributionTab sa={l1.source_attribution} />}
      {tab === 'provenance' && <ProvenanceTab pc={l1.provenance_completeness} cc={l1.confidence_calibration} />}
      {tab === 'faithfulness' && <FaithfulnessTab ef={l1.explanation_faithfulness} baselineEf={baselineEf} />}
      {tab === 'heatmap' && <HeatmapTab hm={l2.chunk_relevance_heatmap} />}
      {tab === 'lime' && <LimeTab data={limeData} />}
      {tab === 'trace' && <TraceTab pt={l2.provenance_trace} />}
      {tab === 'pipeline' && <PipelineTab metrics={data.metrics} />}
    </div>
  )
}

function SummaryTab({ tl, l1, limeData }) {
  const s = tl.summary || {}
  return (
    <div>
      <div className="metrics-grid">
        <div className={`metric-card ${metricQuality(s.source_attribution_f1)}`}>
          <div className="metric-label">Attribution F1</div>
          <div className="metric-value">{pct(s.source_attribution_f1 || 0)}</div>
          <div className="score-bar"><div className={`score-bar-fill ${scoreClass(s.source_attribution_f1)}`} style={{ width: pct(s.source_attribution_f1 || 0) }} /></div>
          <div className="metric-sub">Source citation accuracy</div>
        </div>
        <div className={`metric-card ${metricQuality(s.provenance_completeness)}`}>
          <div className="metric-label">Provenance</div>
          <div className="metric-value">{pct(s.provenance_completeness || 0)}</div>
          <div className="score-bar"><div className={`score-bar-fill ${scoreClass(s.provenance_completeness)}`} style={{ width: pct(s.provenance_completeness || 0) }} /></div>
          <div className="metric-sub">Claims with traceable source</div>
        </div>
        <div className={`metric-card`}>
          <div className="metric-label">Confidence</div>
          <div className="metric-value">{pct(s.confidence_score || 0)}</div>
          <span className={`confidence-badge confidence-${s.confidence_tier}`}>{s.confidence_tier}</span>
        </div>
        <div className={`metric-card ${metricQuality(s.explanation_faithfulness)}`}>
          <div className="metric-label">Faithfulness</div>
          <div className="metric-value">{pct(s.explanation_faithfulness || 0)}</div>
          <div className="score-bar"><div className={`score-bar-fill ${scoreClass(s.explanation_faithfulness)}`} style={{ width: pct(s.explanation_faithfulness || 0) }} /></div>
          <div className="metric-sub">Answer grounded in sources</div>
        </div>
        <div className={`metric-card ${metricQuality(s.answer_grounding_ratio)}`}>
          <div className="metric-label">Grounding ratio</div>
          <div className="metric-value">{pct(s.answer_grounding_ratio || 0)}</div>
          <div className="score-bar"><div className={`score-bar-fill ${scoreClass(s.answer_grounding_ratio)}`} style={{ width: pct(s.answer_grounding_ratio || 0) }} /></div>
          <div className="metric-sub">Answer tokens from context</div>
        </div>
        {limeData ? (
          <div className={`metric-card ${metricQuality(limeData.stability_score)}`}>
            <div className="metric-label">Query stability (LIME)</div>
            <div className="metric-value">{pct(limeData.stability_score || 0)}</div>
            <div className="score-bar"><div className={`score-bar-fill ${scoreClass(limeData.stability_score)}`} style={{ width: pct(limeData.stability_score || 0) }} /></div>
            <div className="metric-sub">Higher = robust retrieval</div>
          </div>
        ) : (
          <div className={`metric-card`}>
            <div className="metric-label">Retrieval spread</div>
            <div className="metric-value">{fmt(s.retrieval_discriminativeness || 0, 3)}</div>
            <div className="metric-sub">Higher = better separation</div>
          </div>
        )}
      </div>
    </div>
  )
}

function AttributionTab({ sa }) {
  return (
    <div className="card">
      <div className="card-header">🎯 Source attribution precision / recall</div>
      <div className="metrics-grid" style={{ marginBottom: '1rem' }}>
        <div className={`metric-card ${metricQuality(sa.precision)}`}>
          <div className="metric-label">Precision</div>
          <div className="metric-value">{pct(sa.precision)}</div>
          <div className="metric-sub">Of cited chunks, how many are relevant</div>
        </div>
        <div className={`metric-card ${metricQuality(sa.recall)}`}>
          <div className="metric-label">Recall</div>
          <div className="metric-value">{pct(sa.recall)}</div>
          <div className="metric-sub">Of relevant chunks, how many were cited</div>
        </div>
        <div className={`metric-card ${metricQuality(sa.f1_score)}`}>
          <div className="metric-label">F1 score</div>
          <div className="metric-value">{pct(sa.f1_score)}</div>
          <div className="metric-sub">Harmonic mean</div>
        </div>
      </div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', gap: '2rem' }}>
        <span>Cited chunks: <strong>{sa.cited_chunk_count}</strong></span>
        <span>Relevant chunks: <strong>{sa.relevant_chunk_count}</strong></span>
        <span>Correctly cited: <strong>{sa.correctly_cited}</strong></span>
      </div>
    </div>
  )
}

function ProvenanceTab({ pc, cc }) {
  return (
    <div>
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header">📍 Provenance completeness</div>
        <div className="metrics-grid" style={{ marginBottom: '1rem' }}>
          <div className={`metric-card ${metricQuality(pc.completeness_score)}`}>
            <div className="metric-label">Completeness</div>
            <div className="metric-value">{pct(pc.completeness_score)}</div>
            <div className="metric-sub">{pc.claims_with_provenance} of {pc.total_claims} claims traceable</div>
          </div>
          <div className={`metric-card`}>
            <div className="metric-label">Granularity</div>
            <div className="metric-value">{fmt(pc.granularity_score, 2)}</div>
            <div className="metric-sub">1.0 = paragraph, 0.5 = page</div>
          </div>
        </div>

        {pc.claim_details && pc.claim_details.length > 0 && (
          <table className="claims-table">
            <thead>
              <tr><th>Claim</th><th>Provenance</th><th>Granularity</th><th>Source</th></tr>
            </thead>
            <tbody>
              {pc.claim_details.map((c, i) => (
                <tr key={i}>
                  <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.claim}</td>
                  <td>{c.has_provenance ? <span className="badge-high" style={{ padding: '2px 8px', borderRadius: 8, fontSize: '0.75rem' }}>✓ Yes</span> : <span className="badge-low" style={{ padding: '2px 8px', borderRadius: 8, fontSize: '0.75rem' }}>✗ No</span>}</td>
                  <td>{c.granularity === 1.0 ? '¶ Paragraph' : c.granularity === 0.5 ? '📄 Page' : c.granularity === 0.25 ? '📦 Chunk' : '—'}</td>
                  <td>{c.source_page >= 0 ? `Page ${c.source_page + 1}` : '—'}{c.source_para > 0 ? `, ¶${c.source_para}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="card-header">🎚 Confidence calibration</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
          <span className={`confidence-badge confidence-${cc.confidence_tier}`}>
            {pct(cc.confidence_score)} — {cc.confidence_tier}
          </span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Bucket: {cc.calibration_bucket}</span>
        </div>
        {cc.score_components && (
          <div style={{ fontSize: '0.85rem', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem' }}>
            <div>Avg retrieval: <strong>{fmt(cc.score_components.avg_retrieval_score, 3)}</strong></div>
            <div>Top retrieval: <strong>{fmt(cc.score_components.top_retrieval_score, 3)}</strong></div>
            <div>Score spread: <strong>{fmt(cc.score_components.score_spread, 3)}</strong></div>
            <div>Min retrieval: <strong>{fmt(cc.score_components.min_retrieval_score, 3)}</strong></div>
            <div>Combined signal: <strong>{fmt(cc.score_components.combined_signal, 3)}</strong></div>
            <div>Num chunks: <strong>{cc.score_components.num_chunks}</strong></div>
          </div>
        )}
      </div>
    </div>
  )
}

function FaithfulnessTab({ ef, baselineEf }) {
  const hasPerturbation = ef.perturbation_tested
  const hasComparison = hasPerturbation && baselineEf && !baselineEf.perturbation_tested

  return (
    <div>
      {/* Comparison banner when perturbation was run */}
      {hasComparison && (
        <div style={{ padding: '1rem 1.25rem', marginBottom: '1rem', background: 'var(--purple-light)', border: '1px solid #c4b5fd', borderRadius: 'var(--radius-lg)' }}>
          <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--purple)', marginBottom: '0.5rem' }}>
            🧪 Perturbation test comparison
          </div>
          <div style={{ fontSize: '0.85rem', color: '#5b21b6' }}>
            The system ran the query twice: first with standard token-overlap faithfulness, then with perturbation-based testing (removing each chunk and re-generating). Compare the results below.
          </div>
        </div>
      )}

      {/* Side-by-side comparison cards */}
      {hasComparison ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          {/* Baseline card */}
          <div className="card" style={{ borderTop: '3px solid var(--border)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Baseline (token overlap)
            </div>
            <div className={`metric-card ${metricQuality(baselineEf.overall_faithfulness)}`} style={{ marginBottom: '0.75rem' }}>
              <div className="metric-label">Overall faithfulness</div>
              <div className="metric-value">{pct(baselineEf.overall_faithfulness)}</div>
              <div className="score-bar"><div className={`score-bar-fill ${scoreClass(baselineEf.overall_faithfulness)}`} style={{ width: pct(baselineEf.overall_faithfulness) }} /></div>
            </div>
            <div className={`metric-card ${metricQuality(baselineEf.answer_grounding_ratio)}`}>
              <div className="metric-label">Grounding ratio</div>
              <div className="metric-value">{pct(baselineEf.answer_grounding_ratio)}</div>
              <div className="score-bar"><div className={`score-bar-fill ${scoreClass(baselineEf.answer_grounding_ratio)}`} style={{ width: pct(baselineEf.answer_grounding_ratio) }} /></div>
            </div>
          </div>

          {/* Perturbation card */}
          <div className="card" style={{ borderTop: '3px solid var(--purple)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--purple)', marginBottom: '1rem' }}>
              🧪 After perturbation test
            </div>
            <div className={`metric-card ${metricQuality(ef.overall_faithfulness)}`} style={{ marginBottom: '0.75rem' }}>
              <div className="metric-label">Overall faithfulness</div>
              <div className="metric-value">
                {pct(ef.overall_faithfulness)}
                {(() => {
                  const diff = ef.overall_faithfulness - baselineEf.overall_faithfulness
                  if (Math.abs(diff) < 0.005) return null
                  return <span style={{ fontSize: '0.8rem', marginLeft: 8, color: diff > 0 ? 'var(--success)' : 'var(--danger)' }}>
                    {diff > 0 ? '▲' : '▼'} {pct(Math.abs(diff))}
                  </span>
                })()}
              </div>
              <div className="score-bar"><div className={`score-bar-fill ${scoreClass(ef.overall_faithfulness)}`} style={{ width: pct(ef.overall_faithfulness) }} /></div>
              <div className="metric-sub">Based on chunk removal impact</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Chunks verified faithful</div>
              <div className="metric-value" style={{ color: 'var(--purple)' }}>
                {ef.chunk_faithfulness?.filter(c => c.is_faithful).length || 0} / {ef.chunk_faithfulness?.length || 0}
              </div>
              <div className="metric-sub">Removing these chunks changed the answer</div>
            </div>
          </div>
        </div>
      ) : (
        /* Standard view when no perturbation comparison */
        <div className="card" style={{ marginBottom: '1rem' }}>
          <div className="card-header">✓ Explanation faithfulness</div>
          <div className="metrics-grid" style={{ marginBottom: '1rem' }}>
            <div className={`metric-card ${metricQuality(ef.overall_faithfulness)}`}>
              <div className="metric-label">Overall faithfulness</div>
              <div className="metric-value">{pct(ef.overall_faithfulness)}</div>
            </div>
            <div className={`metric-card ${metricQuality(ef.answer_grounding_ratio)}`}>
              <div className="metric-label">Grounding ratio</div>
              <div className="metric-value">{pct(ef.answer_grounding_ratio)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Perturbation tested</div>
              <div className="metric-value" style={{ fontSize: '1.2rem' }}>{ef.perturbation_tested ? '✓ Yes' : '— No'}</div>
              {!ef.perturbation_tested && <div className="metric-sub">Enable checkbox above to test</div>}
            </div>
          </div>
        </div>
      )}

      {/* Per-chunk comparison table */}
      {ef.chunk_faithfulness && ef.chunk_faithfulness.length > 0 && (
        <div className="card">
          <div className="card-header">
            {hasPerturbation ? '🧪 Per-chunk perturbation results' : 'Per-chunk contribution'}
          </div>

          {hasPerturbation ? (
            /* Perturbation results table */
            <table className="claims-table">
              <thead>
                <tr>
                  <th>Chunk</th>
                  <th>Page</th>
                  <th>Retrieval score</th>
                  <th>Answer change when removed</th>
                  <th>Faithful?</th>
                </tr>
              </thead>
              <tbody>
                {ef.chunk_faithfulness.map((c, i) => {
                  const changeRatio = c.answer_change_ratio || 0
                  const baselineChunk = baselineEf?.chunk_faithfulness?.[i]
                  return (
                    <tr key={i}>
                      <td style={{ fontWeight: 600 }}>Chunk {(c.chunk_index || i) + 1}</td>
                      <td>{c.page >= 0 ? `Page ${c.page + 1}` : '—'}{c.para > 0 ? `, ¶${c.para}` : ''}</td>
                      <td>{fmt(c.retrieval_score || baselineChunk?.retrieval_score || 0, 3)}</td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: 120, height: 16, background: 'var(--surface-alt)', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
                            <div style={{
                              height: '100%',
                              width: `${Math.min(changeRatio * 100, 100)}%`,
                              background: changeRatio >= 0.15 ? 'var(--success)' : 'var(--danger)',
                              borderRadius: 4,
                              transition: 'width 0.6s'
                            }} />
                          </div>
                          <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{pct(changeRatio)}</span>
                        </div>
                      </td>
                      <td>
                        {c.is_faithful
                          ? <span style={{ padding: '3px 10px', background: 'var(--success-light)', color: 'var(--success)', borderRadius: 12, fontSize: '0.8rem', fontWeight: 600 }}>✓ Faithful</span>
                          : <span style={{ padding: '3px 10px', background: 'var(--danger-light)', color: 'var(--danger)', borderRadius: 12, fontSize: '0.8rem', fontWeight: 600 }}>✗ Not faithful</span>
                        }
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          ) : (
            /* Basic contribution bars */
            ef.chunk_faithfulness.map((c, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.4rem 0', borderBottom: '1px solid var(--surface-alt)' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, width: 80 }}>Chunk {(c.chunk_index || i) + 1}</span>
                <div style={{ flex: 1, height: 12, background: 'var(--surface-alt)', borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: pct(c.faithfulness_contribution || 0), background: c.faithfulness_contribution > 0.2 ? 'var(--success)' : c.faithfulness_contribution > 0.05 ? 'var(--warning)' : 'var(--danger)', borderRadius: 6, transition: 'width 0.6s' }} />
                </div>
                <span style={{ fontSize: '0.8rem', width: 50, textAlign: 'right' }}>{pct(c.faithfulness_contribution || 0)}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', width: 80 }}>Score: {fmt(c.retrieval_score, 3)}</span>
              </div>
            ))
          )}

          {/* Interpretation help */}
          {hasPerturbation && (
            <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'var(--surface-alt)', borderRadius: 'var(--radius)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              <strong>How to read this:</strong> Each chunk was removed from context and the answer was re-generated. 
              "Answer change" shows how much the answer changed without that chunk.
              A high change ratio (green, ≥15%) means the chunk was genuinely used — the system faithfully cited it.
              A low change ratio (red, &lt;15%) means the chunk was cited but didn't actually influence the answer.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function LimeTab({ data }) {
  if (!data || !data.word_importance || data.word_importance.length === 0) {
    return <div className="card"><div className="card-header">🧪 LIME query analysis</div><p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>No LIME data. Enable the LIME checkbox and re-run the query.</p></div>
  }

  const maxImportance = Math.max(...data.word_importance.map(w => w.importance), 0.01)

  return (
    <div>
      {/* Summary cards */}
      <div className="metrics-grid" style={{ marginBottom: '1rem' }}>
        <div className={`metric-card ${metricQuality(data.stability_score)}`}>
          <div className="metric-label">Query stability</div>
          <div className="metric-value">{pct(data.stability_score)}</div>
          <div className="score-bar"><div className={`score-bar-fill ${scoreClass(data.stability_score)}`} style={{ width: pct(data.stability_score) }} /></div>
          <div className="metric-sub">1.0 = fully robust retrieval</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Query sensitivity</div>
          <div className="metric-value">{pct(data.query_sensitivity)}</div>
          <div className="metric-sub">Higher = retrieval is fragile</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Words tested</div>
          <div className="metric-value">{data.total_perturbations}</div>
          <div className="metric-sub">Content words perturbed</div>
        </div>
      </div>

      {/* Most / least important words */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div className="card" style={{ borderTop: '3px solid var(--danger)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--danger)', marginBottom: '0.75rem' }}>
            Most important words (removing these changes retrieval most)
          </div>
          {data.most_important_words.map((w, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0.4rem 0' }}>
              <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--danger)' }}>{i + 1}.</span>
              <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'monospace', background: 'var(--danger-light)', padding: '2px 10px', borderRadius: 6, color: 'var(--danger)' }}>{w.word}</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>importance: {pct(w.importance)}</span>
            </div>
          ))}
        </div>
        <div className="card" style={{ borderTop: '3px solid var(--success)' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--success)', marginBottom: '0.75rem' }}>
            Least important words (removing these barely affects retrieval)
          </div>
          {data.least_important_words.map((w, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0.4rem 0' }}>
              <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--success)' }}>{i + 1}.</span>
              <span style={{ fontWeight: 600, fontSize: '0.95rem', fontFamily: 'monospace', background: 'var(--success-light)', padding: '2px 10px', borderRadius: 6, color: 'var(--success)' }}>{w.word}</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>importance: {pct(w.importance)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Full word importance breakdown */}
      <div className="card">
        <div className="card-header">🧪 LIME — per-word retrieval importance</div>
        <div style={{ marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Original query: <strong style={{ fontFamily: 'monospace' }}>{data.original_query}</strong>
        </div>

        {data.word_importance.map((w, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0', borderBottom: '1px solid var(--surface-alt)' }}>
            <span style={{
              fontFamily: 'monospace', fontWeight: 600, fontSize: '0.9rem', minWidth: 120,
              padding: '3px 10px', borderRadius: 6,
              background: w.importance > 0.5 ? 'var(--danger-light)' : w.importance > 0.2 ? 'var(--warning-light)' : 'var(--success-light)',
              color: w.importance > 0.5 ? 'var(--danger)' : w.importance > 0.2 ? 'var(--warning)' : 'var(--success)',
            }}>
              {w.word_clean}
            </span>
            <div style={{ flex: 1, height: 16, background: 'var(--surface-alt)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${Math.max((w.importance / maxImportance) * 100, 1)}%`,
                background: w.importance > 0.5 ? 'var(--danger)' : w.importance > 0.2 ? 'var(--warning)' : 'var(--success)',
                borderRadius: 4,
                transition: 'width 0.6s'
              }} />
            </div>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, width: 55, textAlign: 'right' }}>{pct(w.importance)}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', width: 100, textAlign: 'right' }}>
              {w.chunks_changed} chunk{w.chunks_changed !== 1 ? 's' : ''} changed
            </span>
          </div>
        ))}

        <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'var(--surface-alt)', borderRadius: 'var(--radius)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <strong>How to read this:</strong> Each word was removed from the query and retrieval was re-run.
          "Importance" is 1 minus the Jaccard overlap between the original chunks and the perturbed chunks.
          High importance (red) means removing that word completely changed which chunks were found.
          Low importance (green) means the word didn't affect retrieval — the same chunks were found without it.
        </div>
      </div>
    </div>
  )
}

function HeatmapTab({ hm }) {
  if (!hm || !hm.chunks) return <div className="card">No heatmap data available</div>
  return (
    <div className="card">
      <div className="card-header">🔥 Chunk relevance heatmap</div>
      <div className="metrics-grid" style={{ marginBottom: '1rem' }}>
        <div className="metric-card"><div className="metric-label">Max score</div><div className="metric-value" style={{ color: 'var(--success)' }}>{fmt(hm.max_score, 3)}</div></div>
        <div className="metric-card"><div className="metric-label">Min score</div><div className="metric-value" style={{ color: 'var(--danger)' }}>{fmt(hm.min_score, 3)}</div></div>
        <div className="metric-card"><div className="metric-label">Mean score</div><div className="metric-value">{fmt(hm.mean_score, 3)}</div></div>
        <div className="metric-card"><div className="metric-label">Spread</div><div className="metric-value">{fmt(hm.score_spread, 3)}</div><div className="metric-sub">Retrieval discrimination</div></div>
      </div>
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', fontSize: '0.8rem' }}>
        <span className="badge-high" style={{ padding: '3px 10px', borderRadius: 8 }}>High: {hm.score_distribution?.high_relevance || 0}</span>
        <span className="badge-medium" style={{ padding: '3px 10px', borderRadius: 8 }}>Medium: {hm.score_distribution?.medium_relevance || 0}</span>
        <span className="badge-low" style={{ padding: '3px 10px', borderRadius: 8 }}>Low: {hm.score_distribution?.low_relevance || 0}</span>
      </div>
      {hm.chunks.map((c, i) => (
        <div key={i} className="source-chunk" style={{ borderLeftColor: c.relevance_band === 'high' ? 'var(--success)' : c.relevance_band === 'medium' ? 'var(--warning)' : 'var(--danger)' }}>
          <div className="source-chunk-header">
            <span className="source-chunk-location">
              Source {i + 1} — {c.page >= 0 ? `Page ${c.page + 1}` : 'Unknown page'}{c.paragraph > 0 ? `, ¶${c.paragraph}` : ''}
            </span>
            <span className={`badge badge-${c.relevance_band}`}>{c.score} — {c.relevance_band}</span>
          </div>
          <div className="source-chunk-text">{c.text_preview}</div>
        </div>
      ))}
    </div>
  )
}

function TraceTab({ pt }) {
  if (!pt) return null
  return (
    <div className="card">
      <div className="card-header">🔗 Provenance trace — full audit trail</div>
      <div style={{ marginBottom: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
        Retrieval method: <strong>{pt.retrieval_method}</strong>
      </div>

      <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Pipeline chain</h4>
      {(pt.provenance_chain || []).map((s, i) => (
        <div key={i} className="trace-step">
          <div className="trace-step-num">{s.step}</div>
          <div className="trace-step-content">
            <div className="step-title">{s.component}</div>
            <div className="step-detail">{s.detail}</div>
          </div>
        </div>
      ))}

      {pt.chunk_to_answer_mapping && pt.chunk_to_answer_mapping.length > 0 && (
        <>
          <h4 style={{ fontSize: '0.9rem', fontWeight: 600, margin: '1.5rem 0 0.75rem' }}>Chunk-to-answer mapping</h4>
          <table className="claims-table">
            <thead>
              <tr><th>Source</th><th>Page</th><th>Para</th><th>Retrieval score</th><th>Answer contribution</th></tr>
            </thead>
            <tbody>
              {pt.chunk_to_answer_mapping.map((m, i) => (
                <tr key={i}>
                  <td>Source {m.source_index}</td>
                  <td>{m.page >= 0 ? m.page + 1 : '—'}</td>
                  <td>{m.paragraph > 0 ? m.paragraph : '—'}</td>
                  <td>{fmt(m.retrieval_score, 3)}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 80, height: 8, background: 'var(--surface-alt)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: pct(m.answer_contribution), background: 'var(--primary)', borderRadius: 4 }} />
                      </div>
                      {pct(m.answer_contribution)}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

function PipelineTab({ metrics }) {
  if (!metrics) return null
  const maxTime = Math.max(metrics.generation_ms, metrics.similarity_search_ms, metrics.explainability_ms, metrics.query_to_vector_ms, 1)

  const bars = [
    { label: 'Query encoding', ms: metrics.query_to_vector_ms },
    { label: 'Retrieval', ms: metrics.similarity_search_ms },
    { label: 'Generation', ms: metrics.generation_ms },
    { label: 'Explainability', ms: metrics.explainability_ms },
  ]

  return (
    <div className="card">
      <div className="card-header">⚡ Pipeline performance</div>
      <div className="metrics-grid" style={{ marginBottom: '1.5rem' }}>
        <div className="metric-card"><div className="metric-label">End-to-end</div><div className="metric-value">{metrics.end_to_end_ms?.toFixed(0)}ms</div></div>
        <div className="metric-card"><div className="metric-label">Memory</div><div className="metric-value">{metrics.memory_used_mb?.toFixed(1)}MB</div></div>
        <div className="metric-card"><div className="metric-label">Prompt tokens</div><div className="metric-value">{metrics.token_count_prompt}</div></div>
        <div className={`metric-card ${metricQuality(metrics.faithfulness_score)}`}><div className="metric-label">Faithfulness</div><div className="metric-value">{pct(metrics.faithfulness_score)}</div></div>
      </div>

      <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Stage timing breakdown</h4>
      {bars.map((b, i) => (
        <div key={i} className="timing-bar">
          <span className="timing-label">{b.label}</span>
          <div className="timing-track">
            <div className="timing-fill" style={{ width: `${Math.max((b.ms / maxTime) * 100, 2)}%` }}>
              {b.ms > maxTime * 0.15 && <span>{b.ms?.toFixed(0)}ms</span>}
            </div>
          </div>
          <span className="timing-value">{b.ms?.toFixed(1)}ms</span>
        </div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════
// QUERY PAGE
// ═══════════════════════════════════════════════
function QueryPage({ documents }) {
  const [question, setQuestion] = useState('')
  const [docId, setDocId] = useState(documents[0]?.id || '')
  const [loading, setLoading] = useState(false)
  const [loadingPhase, setLoadingPhase] = useState('')
  const [response, setResponse] = useState(null)
  const [baselineResponse, setBaselineResponse] = useState(null)
  const [runPerturbation, setRunPerturbation] = useState(false)
  const [runLime, setRunLime] = useState(false)
  const [error, setError] = useState(null)

  const runQuery = async (perturbation) => {
    const res = await fetch(`${API}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_id: docId,
        top_k: 5,
        run_perturbation_test: perturbation,
        run_lime_analysis: runLime
      })
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Query failed')
    }
    return res.json()
  }

  const handleQuery = async () => {
    if (!question.trim() || !docId) return
    setLoading(true)
    setError(null)
    setResponse(null)
    setBaselineResponse(null)

    try {
      // Always run baseline first
      setLoadingPhase('Retrieving and generating answer...')
      const baseline = await runQuery(false)
      setBaselineResponse(baseline)
      setResponse(baseline)

      // If perturbation checkbox is on, run again with perturbation
      if (runPerturbation) {
        setLoadingPhase('Running perturbation test (removing chunks one-by-one and re-generating)...')
        const perturbed = await runQuery(true)
        setResponse(perturbed)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      setLoadingPhase('')
    }
  }

  return (
    <div>
      <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '1.5rem' }}>Ask a question</h2>

      <div className="query-input-area">
        <select value={docId} onChange={e => setDocId(e.target.value)}>
          <option value="">Select document...</option>
          {documents.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <input
          type="text"
          placeholder="e.g. Is the Fund an exchange-traded fund?"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleQuery()}
        />
        <button className="btn btn-primary" onClick={handleQuery} disabled={loading || !docId || !question.trim()}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1.5rem', padding: '0.75rem 1rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', fontWeight: 500 }}>
          <input
            type="checkbox"
            checked={runPerturbation}
            onChange={e => setRunPerturbation(e.target.checked)}
            style={{ width: 18, height: 18, accentColor: 'var(--purple)', cursor: 'pointer' }}
          />
          Perturbation test
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem', fontWeight: 500 }}>
          <input
            type="checkbox"
            checked={runLime}
            onChange={e => setRunLime(e.target.checked)}
            style={{ width: 18, height: 18, accentColor: 'var(--info)', cursor: 'pointer' }}
          />
          LIME query analysis
        </label>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>
          {runPerturbation && runLime ? 'Both enabled — perturbation tests chunk faithfulness, LIME tests query word importance' :
           runPerturbation ? 'Removes each chunk and re-generates to test citation faithfulness' :
           runLime ? 'Removes each query word and re-runs retrieval to find which words matter most' :
           'Enable XAI tests for deeper explainability analysis'}
        </span>
      </div>

      {loading && <div className="loading"><div className="spinner" /> {loadingPhase}</div>}
      {error && <div style={{ padding: '1rem', background: 'var(--danger-light)', border: '1px solid #fca5a5', borderRadius: 'var(--radius)', color: 'var(--danger)' }}>{error}</div>}

      {response && (
        <>
          <div className="answer-card">
            <div className="answer-header">
              Answer
              {response.three_layer_explainability?.summary?.confidence_tier && (
                <span className={`confidence-badge confidence-${response.three_layer_explainability.summary.confidence_tier}`} style={{ marginLeft: 'auto', fontSize: '0.8rem' }}>
                  Confidence: {pct(response.three_layer_explainability.summary.confidence_score || 0)}
                </span>
              )}
            </div>
            <div className="answer-body">{response.answer}</div>
          </div>

          {response.three_layer_explainability && <ExplainabilityDashboard data={response} baselineData={baselineResponse} limeData={response.lime_analysis} />}

          {/* Source chunks from legacy explainability */}
          <div className="card" style={{ marginTop: '1.5rem' }}>
            <div className="card-header">📄 Retrieved source chunks</div>
            {response.explainability?.sources_used?.map((s, i) => (
              <div key={i} className="source-chunk">
                <div className="source-chunk-header">
                  <span className="source-chunk-location">
                    Source {i + 1} — {s.page_number >= 0 ? `Page ${s.page_number + 1}` : ''}{s.paragraph_number > 0 ? `, ¶${s.paragraph_number}` : ''}
                  </span>
                  <span className={`badge badge-${scoreClass(s.relevance_score)}`}>{fmt(s.relevance_score, 3)}</span>
                </div>
                <div className="source-chunk-text">{s.content}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState('upload')
  const [documents, setDocuments] = useState([])

  return (
    <div className="app">
      <header className="header">
        <div className="header-logo">
          <span>🔍</span> Explainable RAG
        </div>
        <nav className="header-nav">
          <button className={page === 'upload' ? 'active' : ''} onClick={() => setPage('upload')}>Upload</button>
          <button className={page === 'query' ? 'active' : ''} onClick={() => setPage('query')}>Query</button>
        </nav>
        <div style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
          MSc Data Analytics — DCU
        </div>
      </header>

      <main className="main">
        {page === 'upload' && <UploadPage documents={documents} setDocuments={setDocuments} />}
        {page === 'query' && <QueryPage documents={documents} />}
      </main>
    </div>
  )
}