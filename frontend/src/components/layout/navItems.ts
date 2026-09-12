import { Gauge, History, MessageCircle } from 'lucide-react'

export const NAV_ITEMS = [
  { to: '/', label: 'Predictions', icon: Gauge, end: true },
  { to: '/history', label: 'History', icon: History, end: false },
  { to: '/agent', label: 'Agent', icon: MessageCircle, end: false },
]
