import { Link } from 'react-router-dom'

export function NotFound() {
  return (
    <div className="mx-auto max-w-md px-4 py-24 text-center md:px-6">
      <p className="font-mono text-sm text-text-muted">404</p>
      <h1 className="mt-2 text-2xl font-semibold text-text-primary">Nothing predicted here</h1>
      <p className="mt-2 text-sm text-text-secondary">This page doesn't exist.</p>
      <Link to="/" className="mt-6 inline-block text-sm font-medium text-accent-text hover:underline">
        Back to predictions
      </Link>
    </div>
  )
}
