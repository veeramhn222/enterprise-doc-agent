// Small helpers (also loadable from Node for tests)
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}
function inline(t) {
  return t.replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
          .replace(/\[S(\d+)\]/g, '<sup class="cite">S$1</sup>');
}
// Minimal, safe markdown: paragraphs, **bold**, `code`, bullet/numbered lists, > quotes, #headings, simple tables.
function md(src) {
  const lines = esc(src || '').split(/\r?\n/);
  const out = [];
  let para = [], list = null, table = null;
  const flushPara = () => { if (para.length) { out.push('<p>' + inline(para.join('<br>')) + '</p>'); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.t}>` + list.items.map(i => '<li>' + inline(i) + '</li>').join('') + `</${list.t}>`); list = null; } };
  const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  const isSep = r => cells(r).every(c => /^:?-{2,}:?$/.test(c));
  const flushTable = () => {
    if (!table) return;
    const rows = table.filter(r => !isSep(r)).map(cells);
    if (rows.length) {
      out.push('<div class="tbl"><table><thead><tr>' + rows[0].map(c => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>' +
        rows.slice(1).map(r => '<tr>' + r.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>');
    }
    table = null;
  };
  for (const line of lines) {
    if (/^\s*\|.*\|\s*$/.test(line)) { flushPara(); flushList(); (table = table || []).push(line); continue; }
    flushTable();
    let m;
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      flushPara(); if (!list || list.t !== 'ul') { flushList(); list = {t: 'ul', items: []}; } list.items.push(m[1]);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      flushPara(); if (!list || list.t !== 'ol') { flushList(); list = {t: 'ol', items: []}; } list.items.push(m[1]);
    } else if ((m = line.match(/^&gt;\s?(.*)$/))) {
      flushPara(); flushList(); out.push('<blockquote>' + inline(m[1]) + '</blockquote>');
    } else if (/^\s*#{1,4}\s+/.test(line)) {
      flushPara(); flushList(); out.push('<h4>' + inline(line.replace(/^\s*#{1,4}\s+/, '')) + '</h4>');
    } else if (!line.trim()) {
      flushPara(); flushList();
    } else {
      flushList(); para.push(line);
    }
  }
  flushPara(); flushList(); flushTable();
  return out.join('');
}
if (typeof module !== 'undefined') module.exports = {esc, md};
