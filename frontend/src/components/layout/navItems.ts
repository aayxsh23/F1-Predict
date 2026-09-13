import { Gauge, History, Scale } from 'lucide-react'

export const NAV_ITEMS = [
  { to: '/', label: 'Predictions', icon: Gauge, end: true },
  { to: '/history', label: 'History', icon: History, end: false },
  { to: '/regulations', label: 'Regulations', icon: Scale, end: false },
]
