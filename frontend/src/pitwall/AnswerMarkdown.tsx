import Markdown from 'react-markdown'
import { Link } from 'react-router-dom'
import remarkGfm from 'remark-gfm'

/** A Pit Radio answer: markdown with GFM tables, in-app links for driver pages. */
export default function AnswerMarkdown({ text }: { text: string }) {
  return (
    <div className="space-y-2.5 font-editorial text-[19px] leading-[1.45] text-ink [&_code]:font-mono [&_code]:text-[13px] [&_li]:ml-5 [&_ol]:list-decimal [&_strong]:font-normal [&_strong]:italic [&_ul]:list-disc">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: (props) => (
            <div className="my-1 overflow-x-auto font-mono text-[12px]">
              <table className="w-full border-collapse" {...props} />
            </div>
          ),
          th: (props) => <th className="border-b border-hairline-strong px-2 py-1 text-left font-medium text-ink-3" {...props} />,
          td: (props) => <td className="border-b border-hairline px-2 py-1 tabular-nums" {...props} />,
          a: ({ href = '', children }) =>
            href.startsWith('/') ? (
              <Link to={href} viewTransition className="text-signal-ink underline decoration-dotted underline-offset-[3px] hover:text-ink">
                {children}
              </Link>
            ) : (
              <a href={href} target="_blank" rel="noreferrer" className="text-signal-ink underline underline-offset-[3px]">
                {children}
              </a>
            ),
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}
