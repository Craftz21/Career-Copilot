'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ArrowRight, Sparkles, Menu } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ThemeToggle } from '@/components/ThemeToggle';

const links = [
  { href: '/', label: 'Home' },
  { href: '/upload', label: 'Upload' },
];

export function PageShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(129,140,248,0.20),_transparent_35%),linear-gradient(135deg,_rgba(255,255,255,0.95),_rgba(241,245,249,0.98))] text-slate-900 transition-colors dark:bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.24),_transparent_35%),linear-gradient(135deg,_#020617,_#0f172a)] dark:text-slate-100">
      <header className="sticky top-0 z-40 border-b border-slate-200/70 bg-white/70 backdrop-blur-xl dark:border-slate-800/80 dark:bg-slate-950/70">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex items-center gap-3 rounded-full px-2 py-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-900 text-white shadow-lg dark:bg-white dark:text-slate-900">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">CareerCopilot</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">AI career acceleration</p>
            </div>
          </Link>

          <nav className="hidden items-center gap-2 md:flex" aria-label="Primary navigation">
            {links.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 ${active ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900' : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'}`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <button
              type="button"
              className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-300 bg-white text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800 md:hidden"
              aria-label="Open navigation menu"
              onClick={() => setMobileMenuOpen((open) => !open)}
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>
        </div>

        {mobileMenuOpen && (
          <div className="border-t border-slate-200 bg-white/90 px-4 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 md:hidden">
            <nav className="flex flex-col gap-2" aria-label="Mobile navigation">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`rounded-2xl px-3 py-2 text-sm font-medium transition ${pathname === link.href ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900' : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'}`}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        )}
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
        {children}
      </main>

      <footer className="border-t border-slate-200/70 bg-white/60 py-8 backdrop-blur dark:border-slate-800/80 dark:bg-slate-950/60">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 text-sm text-slate-600 dark:text-slate-400 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <p>Designed for ambitious professionals and hiring teams.</p>
          <a href="mailto:hello@careercopilot.ai" className="inline-flex items-center gap-2 font-medium text-slate-900 hover:underline dark:text-slate-200">
            hello@careercopilot.ai <ArrowRight className="h-4 w-4" />
          </a>
        </div>
      </footer>
    </div>
  );
}
