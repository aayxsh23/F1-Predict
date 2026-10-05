import { Link } from 'react-router-dom'

import { useChrome } from './chrome'

export function NotFoundPage() {
  useChrome({ section: null, crumbs: [{ label: '404' }], chat: {}, badge: 'Pit Wall', quick: ['Who is leading the drivers championship?'] })
  return (
    <div className="relative flex min-h-full items-center justify-center overflow-hidden px-5 pb-36 pt-10">
      <span
        className="pointer-events-none absolute select-none bg-[linear-gradient(180deg,rgb(255_90_71/.35),rgb(255_90_71/0)_80%)] bg-clip-text font-headline text-[clamp(260px,40vw,560px)] font-black leading-none text-transparent"
        aria-hidden
      >
        404
      </span>
      <div className="relative flex flex-col items-center gap-4 text-center">
        <span className="pw-label">404 · off track</span>
        <h1 className="font-headline text-[clamp(56px,7vw,96px)] font-black uppercase leading-[.85]">Nothing predicted here</h1>
        <p className="font-editorial text-[24px] italic leading-[1.3] text-ink-2">This page doesn't exist.</p>
        <Link to="/" className="mt-1.5 flex h-11 items-center rounded-md bg-signal px-[18px] text-[13px] font-bold leading-none text-asphalt hover:bg-signal-ink hover:text-asphalt">
          Back to predictions
        </Link>
      </div>
    </div>
  )
}
