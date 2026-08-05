import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { PageShell } from '@/components/PageShell';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'CareerCopilot | AI Career Growth Platform',
  description: 'A modern SaaS experience for resume analysis, skill-gap insights, and roadmap planning.',
};

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full bg-slate-50 text-slate-900 transition-colors dark:bg-slate-950 dark:text-slate-100">
        <Providers>
          <PageShell>{children}</PageShell>
        </Providers>
      </body>
    </html>
  );
}
