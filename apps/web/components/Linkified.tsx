import React from 'react';

const URL_PATTERN = /(https:\/\/[^\s<>"')\]]+)/g;

/** Renders text, turning https:// URLs into links. Everything else stays plain text. */
export function Linkified({ text, className }: { text: string; className?: string }) {
  const parts = text.split(URL_PATTERN);
  return (
    <span className={className}>
      {parts.map((part, index) => {
        if (index % 2 === 0) return <React.Fragment key={index}>{part}</React.Fragment>;
        // Trailing punctuation belongs to the sentence, not the address.
        const trailing = (part.match(/[.,;:!?]+$/) || [''])[0];
        const href = trailing ? part.slice(0, -trailing.length) : part;
        return (
          <React.Fragment key={index}>
            <a
              href={href}
              target="_blank"
              rel="noreferrer noopener"
              className="text-blue-400 hover:text-blue-300 underline break-all"
            >
              {href}
            </a>
            {trailing}
          </React.Fragment>
        );
      })}
    </span>
  );
}
