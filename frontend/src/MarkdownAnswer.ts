import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  type SourceCitation,
  prepareMarkdownContent,
  normalizeCodeBlock,
  remarkBreakHtml,
  isAllowedUrl
} from './markdownUtils.ts';

export interface MarkdownAnswerProps {
  content: string;
  sources?: SourceCitation[];
  className?: string;
  inline?: boolean;
}

export const MarkdownAnswer: React.FC<MarkdownAnswerProps> = ({
  content,
  sources = [],
  className = '',
  inline = false
}) => {
  const prepared = React.useMemo(() => prepareMarkdownContent(content), [content]);

  // Handle citation clicks with smooth scrolling and highlight animation
  const handleCitationClick = (e: React.MouseEvent<HTMLAnchorElement>, sourceId: string) => {
    if (typeof document === 'undefined') return;
    const target = document.getElementById(`source-${sourceId}`);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.classList.add('source-card-highlight');
      setTimeout(() => target.classList.remove('source-card-highlight'), 2000);
    }
  };

  const ContainerTag = inline ? 'span' : 'div';
  const containerClass = inline ? `markdown-answer-inline ${className}` : `markdown-answer-body ${className}`;

  if (!content) {
    return null;
  }

  return React.createElement(
    ContainerTag,
    { className: containerClass },
    React.createElement(
      ReactMarkdown,
      {
        remarkPlugins: [remarkGfm, remarkBreakHtml],
        components: {
          table({ children }) {
            return React.createElement(
              'div',
              { className: 'markdown-table-wrapper' },
              React.createElement('table', { className: 'markdown-table' }, children)
            );
          },
          thead({ children }) {
            return React.createElement('thead', { className: 'markdown-thead' }, children);
          },
          tbody({ children }) {
            return React.createElement('tbody', { className: 'markdown-tbody' }, children);
          },
          tr({ children }) {
            return React.createElement('tr', { className: 'markdown-tr' }, children);
          },
          th({ children }) {
            return React.createElement('th', { className: 'markdown-th' }, children);
          },
          td({ children }) {
            return React.createElement('td', { className: 'markdown-td' }, children);
          },
          pre({ children }) {
            return children;
          },
          code({ className: codeClass, children, node: _node, ...props }) {
            const match = /language-(\w+)/.exec(codeClass || '');
            const rawText = String(children || '').replace(/\n$/, '');
            const isBlock = Boolean(match) || rawText.includes('\n');

            if (isBlock) {
              const lang = match ? match[1] : '';
              const normalized = normalizeCodeBlock(rawText, lang);

              return React.createElement(
                'div',
                { className: 'code-block-wrapper' },
                lang ? React.createElement('span', { className: 'code-lang' }, lang.toUpperCase()) : null,
                React.createElement(
                  'pre',
                  { className: 'code-block' },
                  React.createElement('code', { className: codeClass, ...props }, normalized)
                )
              );
            }

            return React.createElement('code', { className: 'inline-code', ...props }, children);
          },
          a({ href, children, node: _node, ...props }) {
            if (href && href.startsWith('#source-')) {
              const sourceId = href.replace('#source-', '').toUpperCase();
              const matchedSource = sources.find(
                s => s.source_id.toUpperCase() === sourceId
              );

              if (matchedSource) {
                const title = matchedSource.section_title
                  ? `Jump to Source [${matchedSource.source_id}]: ${matchedSource.section_title}`
                  : `Jump to Source [${matchedSource.source_id}]`;

                const textStr = String(children || '');
                const displayText = textStr.startsWith('[') ? textStr : `[${textStr}]`;

                return React.createElement(
                  'a',
                  {
                    href,
                    className: 'citation-pill active-citation',
                    title,
                    onClick: (e: React.MouseEvent<HTMLAnchorElement>) => handleCitationClick(e, matchedSource.source_id)
                  },
                  displayText
                );
              }

              const textStr = String(children || '');
              const displayText = textStr.startsWith('[') ? textStr : `[${textStr}]`;
              return React.createElement(
                'span',
                {
                  className: 'citation-pill unverified-citation',
                  title: 'Citation not in retrieved sources for this response'
                },
                displayText
              );
            }

            if (href && isAllowedUrl(href, sources)) {
              return React.createElement(
                'a',
                {
                  href,
                  target: '_blank',
                  rel: 'noopener noreferrer',
                  className: 'markdown-link',
                  ...props
                },
                children
              );
            }

            return React.createElement(
              'span',
              {
                className: 'untrusted-link-text',
                title: 'Untrusted external link blocked for security'
              },
              children
            );
          },
          p({ children }) {
            if (inline) {
              return children;
            }
            return React.createElement('p', { className: 'answer-paragraph' }, children);
          },
          h1({ children }) {
            return React.createElement('h1', { className: 'answer-h1' }, children);
          },
          h2({ children }) {
            return React.createElement('h2', { className: 'answer-h2' }, children);
          },
          h3({ children }) {
            return React.createElement('h3', { className: 'answer-h3' }, children);
          },
          h4({ children }) {
            return React.createElement('h4', { className: 'answer-h4' }, children);
          },
          ul({ children }) {
            return React.createElement('ul', { className: 'answer-ul' }, children);
          },
          ol({ children }) {
            return React.createElement('ol', { className: 'answer-ol' }, children);
          },
          li({ children }) {
            return React.createElement('li', { className: 'answer-li' }, children);
          },
          blockquote({ children }) {
            return React.createElement('blockquote', { className: 'answer-blockquote' }, children);
          }
        }
      },
      prepared
    )
  );
};

export default MarkdownAnswer;
