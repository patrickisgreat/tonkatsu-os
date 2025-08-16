import { Inter } from 'next/font/google'
import { Providers } from './providers'
import { Toaster } from 'sonner'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Tonkatsu-OS | AI-Powered Raman Analysis',
  description: 'Advanced DIY Raman spectrometer with machine learning-based molecular identification',
  keywords: 'raman, spectroscopy, machine learning, molecular identification, diy, chemistry',
  authors: [{ name: 'Patrick' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>
        <Providers>
          <div className="min-h-full">
            <nav className="bg-white shadow-sm border-b border-gray-200">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <h1 className="text-2xl font-bold text-primary-600">
                        🔬 Tonkatsu-OS
                      </h1>
                    </div>
                    <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                      <a
                        href="/dashboard"
                        className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-primary-300"
                      >
                        Dashboard
                      </a>
                      <a
                        href="/analyze"
                        className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-primary-300"
                      >
                        Analyze
                      </a>
                      <a
                        href="/import"
                        className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-primary-300"
                      >
                        Import
                      </a>
                      <a
                        href="/database"
                        className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-primary-300"
                      >
                        Database
                      </a>
                      <a
                        href="/training"
                        className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900 border-b-2 border-transparent hover:border-primary-300"
                      >
                        Training
                      </a>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-gray-500">v0.2.0</span>
                  </div>
                </div>
              </div>
            </nav>
            <main className="flex-1">
              {children}
            </main>
          </div>
          <Toaster position="top-right" richColors />
        </Providers>
      </body>
    </html>
  )
}