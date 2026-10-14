import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import {
  formatComponentLabel,
  normalizeCodeBlock,
  prepareMarkdownContent,
  isAllowedUrl
} from './markdownUtils.ts';
import { MarkdownAnswer } from './MarkdownAnswer.ts';

describe('MarkdownAnswer & markdownUtils Unit & Regression Tests', () => {
  describe('formatComponentLabel', () => {
    it('formats Spring Boot Core component', () => {
      const res = formatComponentLabel('spring-boot-core', 'spring-boot');
      assert.equal(res.label, 'Spring Boot Core');
      assert.equal(res.type, 'core');
    });

    it('formats Spring Framework component', () => {
      const res = formatComponentLabel('spring-framework', 'spring-boot');
      assert.equal(res.label, 'Spring Framework');
      assert.equal(res.type, 'framework');
    });

    it('formats Spring Cloud Gateway component', () => {
      const res = formatComponentLabel('spring-cloud-gateway', 'spring-boot');
      assert.equal(res.label, 'Spring Cloud Gateway (Gateway Component)');
      assert.equal(res.type, 'gateway');
    });

    it('formats Resilience4j third-party library', () => {
      const res = formatComponentLabel('resilience4j', 'spring-boot');
      assert.equal(res.label, 'Resilience4j (Third-Party Library)');
      assert.equal(res.type, 'third-party');
    });

    it('formats empty or null component falling back to technology core', () => {
      const res1 = formatComponentLabel(null, 'spring-boot');
      assert.equal(res1.label, 'SPRING-BOOT Core');
      assert.equal(res1.type, 'core');

      const res2 = formatComponentLabel('', 'fastapi');
      assert.equal(res2.label, 'FASTAPI Core');
      assert.equal(res2.type, 'core');
    });

    it('formats generic component names', () => {
      const res = formatComponentLabel('orm-toolkit', 'django');
      assert.equal(res.label, 'Orm Toolkit');
      assert.equal(res.type, 'general');
    });
  });

  describe('normalizeCodeBlock', () => {
    it('restores flattened single-line escaped newlines in YAML/config', () => {
      const raw = 'spring:\\n  cloud:\\n    gateway:\\n      routes:';
      const normalized = normalizeCodeBlock(raw, 'yaml');
      assert.equal(
        normalized,
        'spring:\n  cloud:\n    gateway:\n      routes:'
      );
    });

    it('preserves indentation in multiline code blocks', () => {
      const raw = 'server:\n  port: 8080\n  servlet:\n    context-path: /api';
      const normalized = normalizeCodeBlock(raw, 'yaml');
      assert.equal(normalized, raw);
    });

    it('handles empty code input safely', () => {
      assert.equal(normalizeCodeBlock('', 'yaml'), '');
    });
  });

  describe('prepareMarkdownContent', () => {
    it('converts citations outside code blocks to anchor links', () => {
      const markdown = 'Rate limiting is documented in [S1] and [S2].';
      const prepared = prepareMarkdownContent(markdown);
      assert.equal(
        prepared,
        'Rate limiting is documented in [S1](#source-S1) and [S2](#source-S2).'
      );
    });

    it('handles lowercase citation tokens [s1]', () => {
      const markdown = 'See [s3] for details.';
      const prepared = prepareMarkdownContent(markdown);
      assert.equal(prepared, 'See [S3](#source-S3) for details.');
    });

    it('does not alter citations inside fenced code blocks', () => {
      const markdown = 'Text [S1]\n```yaml\n# Note [S2] is code\nkey: value\n```\nAfter [S3]';
      const prepared = prepareMarkdownContent(markdown);
      assert.ok(prepared.includes('[S1](#source-S1)'));
      assert.ok(prepared.includes('# Note [S2] is code'));
      assert.ok(!prepared.includes('# Note [S2](#source-S2)'));
      assert.ok(prepared.includes('[S3](#source-S3)'));
    });

    it('handles zero-width characters inside citation brackets', () => {
      const markdown = 'Evidence [\u200bS1\u200b] verifies this.';
      const prepared = prepareMarkdownContent(markdown);
      assert.equal(prepared, 'Evidence [S1](#source-S1) verifies this.');
    });
  });

  describe('isAllowedUrl', () => {
    const mockSources = [
      {
        source_id: 'S1',
        chunk_id: 'c1',
        document_id: 'd1',
        section_title: 'Spring Gateway',
        source_url: 'https://docs.spring.io/spring-cloud-gateway/docs/current/reference/html/',
        score: 0.85
      }
    ];

    it('allows official documentation domains', () => {
      assert.ok(isAllowedUrl('https://docs.spring.io/spring-boot/docs/current/', mockSources));
      assert.ok(isAllowedUrl('https://fastapi.tiangolo.com/tutorial/', mockSources));
      assert.ok(isAllowedUrl('https://react.dev/learn', mockSources));
      assert.ok(isAllowedUrl('https://resilience4j.readme.io/docs/ratelimiter', mockSources));
    });

    it('allows domains present in retrieved sources', () => {
      assert.ok(isAllowedUrl('https://docs.spring.io/spring-cloud-gateway/', mockSources));
    });

    it('rejects unapproved or untrusted external domains', () => {
      assert.equal(isAllowedUrl('https://attacker.com/malicious', mockSources), false);
      assert.equal(isAllowedUrl('http://random-blog.net/post', mockSources), false);
      assert.equal(isAllowedUrl('not-a-valid-url', mockSources), false);
    });
  });

  describe('MarkdownAnswer Static Rendering', () => {
    const mockSources = [
      {
        source_id: 'S1',
        chunk_id: 'c1',
        document_id: 'd1',
        section_title: 'Spring Cloud Gateway RateLimiter',
        source_url: 'https://docs.spring.io/spring-cloud-gateway/reference/filters/requestratelimiter.html',
        score: 0.85,
        technology: 'spring-boot',
        ecosystem_component: 'spring-cloud-gateway'
      },
      {
        source_id: 'S2',
        chunk_id: 'c2',
        document_id: 'd2',
        section_title: 'Resilience4j RateLimiter',
        source_url: 'https://resilience4j.readme.io/docs/ratelimiter',
        score: 0.78,
        technology: 'spring-boot',
        ecosystem_component: 'resilience4j'
      }
    ];

    it('renders GitHub Flavored Markdown table to proper HTML table structure', () => {
      const tableMarkdown = `
| Approach | Component | Scope |
| :--- | :--- | :--- |
| RequestRateLimiter | Spring Cloud Gateway [S1] | Gateway level |
| RateLimiter Annotation | Resilience4j [S2] | Method level |
`;
      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: tableMarkdown, sources: mockSources })
      );

      // Verify table wrapper and elements
      assert.ok(html.includes('<div class="markdown-table-wrapper">'));
      assert.ok(html.includes('<table class="markdown-table">'));
      assert.ok(html.includes('<th class="markdown-th">Approach</th>'));
      assert.ok(html.includes('<th class="markdown-th">Component</th>'));
      assert.ok(html.includes('<th class="markdown-th">Scope</th>'));
      assert.ok(html.includes('<td class="markdown-td">RequestRateLimiter</td>'));
      assert.ok(html.includes('<td class="markdown-td">RateLimiter Annotation</td>'));

      // Verify no raw table separator rows leaked as text
      assert.ok(!html.includes('| :--- |'));
      assert.ok(!html.includes('| --- |'));
    });

    it('safely converts literal <br> and <br/> tags to line breaks', () => {
      const markdown = 'First line<br>Second line<br/>Third line <br /> Fourth line';
      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: markdown, sources: mockSources })
      );

      assert.ok(html.includes('<br/>') || html.includes('<br>'));
      assert.ok(!html.includes('&lt;br&gt;'));
      assert.ok(!html.includes('&lt;br/&gt;'));
    });

    it('renders verified citations as active links with jump targets', () => {
      const markdown = 'Configured in Gateway [S1] and Resilience4j [S2].';
      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: markdown, sources: mockSources })
      );

      assert.ok(html.includes('href="#source-S1"'));
      assert.ok(html.includes('class="citation-pill active-citation"'));
      assert.ok(html.includes('title="Jump to Source [S1]: Spring Cloud Gateway RateLimiter"'));
      assert.ok(html.includes('[S1]'));

      assert.ok(html.includes('href="#source-S2"'));
      assert.ok(html.includes('title="Jump to Source [S2]: Resilience4j RateLimiter"'));
      assert.ok(html.includes('[S2]'));
    });

    it('renders unverified citation as unverified pill', () => {
      const markdown = 'Unsupported feature mentioned in [S99].';
      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: markdown, sources: mockSources })
      );

      assert.ok(html.includes('class="citation-pill unverified-citation"'));
      assert.ok(html.includes('title="Citation not in retrieved sources for this response"'));
      assert.ok(html.includes('[S99]'));
      assert.ok(!html.includes('href="#source-S99"'));
    });

    it('renders code block with language badge and pre/code formatting', () => {
      const markdown = '```yaml\nspring:\n  cloud:\n    gateway:\n      routes: []\n```';
      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: markdown, sources: mockSources })
      );

      assert.ok(html.includes('<div class="code-block-wrapper">'));
      assert.ok(html.includes('<span class="code-lang">YAML</span>'));
      assert.ok(html.includes('<pre class="code-block">'));
      assert.ok(html.includes('<code class="language-yaml">'));
      assert.ok(html.includes('routes: []'));
    });

    it('renders the complete rate-limiting test fixture deterministically', () => {
      const rateLimitingFixture = `
### Rate Limiting in Spring Boot Ecosystem

Rate limiting can be implemented at multiple levels:

| Mechanism | Component | Level | Supporting Source |
| :--- | :--- | :--- | :--- |
| \`RequestRateLimiter\` Filter | Spring Cloud Gateway | API Gateway | [S1] |
| \`@RateLimiter\` Aspect | Resilience4j | Controller / Service | [S2] |

#### Gateway Configuration Example
\`\`\`yaml
spring:
  cloud:
    gateway:
      routes:
        - id: api_route
          uri: lb://backend-service
          filters:
            - name: RequestRateLimiter
\`\`\`

Both built-in gateway mechanisms and application-level libraries are supported [S1][S2].
`;

      const html = renderToStaticMarkup(
        React.createElement(MarkdownAnswer, { content: rateLimitingFixture, sources: mockSources })
      );

      // Verify table
      assert.ok(html.includes('<table class="markdown-table">'));
      assert.ok(html.includes('<th class="markdown-th">Mechanism</th>'));
      assert.ok(html.includes('<th class="markdown-th">Supporting Source</th>'));

      // Verify citations in table and paragraph
      assert.ok(html.includes('href="#source-S1"'));
      assert.ok(html.includes('href="#source-S2"'));

      // Verify YAML code block
      assert.ok(html.includes('<span class="code-lang">YAML</span>'));
      assert.ok(html.includes('RequestRateLimiter'));

      // Verify no raw unrendered pipes
      assert.ok(!html.includes('| :--- |'));
    });
  });
});
