'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navigation = [
  { name: 'Dashboard', href: '/dashboard' },
  { name: 'Analyze', href: '/analyze' },
  { name: 'Import', href: '/import' },
  { name: 'Database', href: '/database' },
  { name: 'Training', href: '/training' },
]

export default function Navigation() {
  const pathname = usePathname()

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Link href="/dashboard" className="text-2xl font-bold text-primary-600 hover:text-primary-700">
                🔬 Tonkatsu-OS
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navigation.map((item) => {
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`inline-flex items-center px-1 pt-1 text-sm font-medium border-b-2 transition-colors ${
                      isActive
                        ? 'text-primary-600 border-primary-500'
                        : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    {item.name}
                  </Link>
                )
              })}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500">v0.2.0</span>
          </div>
        </div>
      </div>
    </nav>
  )
}