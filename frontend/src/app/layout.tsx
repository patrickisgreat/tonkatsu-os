import { Inter } from 'next/font/google'
import { Providers } from './providers'
import { Toaster } from 'sonner'
import Navigation from '@/components/Navigation'
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
            <Navigation />
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