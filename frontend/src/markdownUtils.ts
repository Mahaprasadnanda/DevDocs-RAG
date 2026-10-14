export interface SourceCitation {
  source_id: string;
  chunk_id: string;
  document_id: string;
  section_title: string | null;
  source_url: string;
  score: number;
  technology?: string;
  doc_version?: string;
  ecosystem_component?: string;
}

export interface ComponentLabelInfo {
  label: string;
  type: 'core' | 'framework' | 'gateway' | 'third-party' | 'general';
}

/**
 * Format ecosystem component into human-readable label and type.
 * Accurately distinguishes Spring Boot Core, Spring Framework,
 * Spring Cloud Gateway, and Resilience4j without hardcoding everything to core.
 */
export function formatComponentLabel(
  component?: string | null,
  technology?: string | null
): ComponentLabelInfo {
  if (!component || !component.trim()) {
    const techName = technology ? technology.toUpperCase() : 'CORE';
    return { label: `${techName} Core`, type: 'core' };
  }

  const norm = component.trim().toLowerCase();

  if (norm.includes('resilience4j')) {
    return { label: 'Resilience4j (Third-Party Library)', type: 'third-party' };
  }
  if (norm.includes('gateway')) {
    return { label: 'Spring Cloud Gateway (Gateway Component)', type: 'gateway' };
  }
  if (norm.includes('framework')) {
    return { label: 'Spring Framework', type: 'framework' };
  }
  if (norm.includes('boot')) {
    return { label: 'Spring Boot Core', type: 'core' };
  }

  // Generic formatting for other ecosystem components
  const cleaned = component.replace(/[()]/g, '');
  const words = cleaned.split(/[-_ ]+/).filter(Boolean);
  const formatted = words
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

  return { label: formatted || 'Core Component', type: 'general' };
}

/**
 * Normalize code block content by safely restoring escaped newlines (\n)
 * when a code block has been flattened, while strictly preserving indentation
 * and legitimate string escapes.
 */
export function normalizeCodeBlock(code: string, lang?: string): string {
  if (!code) return '';

  let normalized = code.replace(/\r\n/g, '\n');

  // Case 1: Entire code block is on a single line containing literal \n sequences
  // e.g., "spring:\n  cloud:\n    gateway:\n      routes:"
  if (!normalized.includes('\n') && normalized.includes('\\n')) {
    normalized = normalized.split('\\n').join('\n');
    return normalized;
  }

  // Case 2: Code block has literal \n at line breaks in YAML / configuration formats
  const isConfigLang = lang && ['yaml', 'yml', 'properties', 'conf', 'config', 'toml'].includes(lang.toLowerCase());
  if (isConfigLang && normalized.includes('\\n')) {
    // Unescape \n that occur before indented keys, list bullets, or comments
    normalized = normalized.replace(/\\n(?=\s*[-a-zA-Z0-9_#:]|\s*$)/g, '\n');
  }

  return normalized;
}

const CITATION_CONVERT_REGEX = /\[[\s\u200b-\u200d\ufeff\u202f]*[Ss](\d+)[\s\u200b-\u200d\ufeff\u202f]*\](?!\(#source-)/g;

/**
 * Prepare markdown content by converting inline citations [S1], [S2]
 * outside code blocks into markdown links [S1](#source-S1).
 */
export function prepareMarkdownContent(content: string): string {
  if (!content) return '';

  // Split on fenced code blocks (```...```) to protect code content
  const parts = content.split(/(```[\s\S]*?```)/g);

  return parts
    .map((part) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        return part;
      }
      // Replace unlinked citations [S1] or [s1] with [S1](#source-S1)
      return part.replace(CITATION_CONVERT_REGEX, '[S$1](#source-S$1)');
    })
    .join('');
}

/**
 * Remark plugin to transform literal <br>, <br/>, or <br /> HTML nodes
 * into native Markdown break nodes ({ type: 'break' }) without enabling raw HTML.
 */
export function remarkBreakHtml() {
  return (tree: any) => {
    function visit(node: any) {
      if (node.type === 'html' && /^<br\s*\/?>$/i.test(String(node.value || '').trim())) {
        node.type = 'break';
        delete node.value;
      }
      if (node.children && Array.isArray(node.children)) {
        node.children.forEach(visit);
      }
    }
    visit(tree);
  };
}

export const ALLOWED_DOC_DOMAINS = [
  'fastapi.tiangolo.com',
  'react.dev',
  'docs.djangoproject.com',
  'flask.palletsprojects.com',
  'docs.pydantic.dev',
  'pydantic.dev',
  'docs.spring.io',
  'spring.io',
  'nodejs.org',
  'go.dev',
  'resilience4j.readme.io',
  'github.com'
];

/**
 * Validates that an external URL belongs to a retrieved source or approved documentation domain.
 */
export function isAllowedUrl(url: string, sources: SourceCitation[]): boolean {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();

    // Check retrieved sources allowlist
    for (const src of sources) {
      try {
        const srcHost = new URL(src.source_url).hostname.toLowerCase();
        if (host === srcHost || host.endsWith('.' + srcHost)) {
          return true;
        }
      } catch {
        // Skip malformed source URL
      }
    }

    // Check approved doc domains
    return ALLOWED_DOC_DOMAINS.some(
      domain => host === domain || host.endsWith('.' + domain)
    );
  } catch {
    return false;
  }
}
