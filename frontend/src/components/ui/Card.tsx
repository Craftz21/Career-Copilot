import type { ReactNode } from 'react';

type CardProps = {
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
};

export function Card({ title, description, children, className = '' }: CardProps) {
  return (
    <section className={`rounded-3xl border border-slate-200/80 bg-white/80 p-6 shadow-[0_20px_60px_-30px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800 dark:bg-slate-900/80 ${className}`}>
      {(title || description) && (
        <div className="mb-4">
          {title && <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{title}</h3>}
          {description && <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{description}</p>}
        </div>
      )}
      {children}
    </section>
  );
}
