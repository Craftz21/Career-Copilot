export function ResultsSkeleton() {
  return (
    <div className="space-y-4">
      <div className="animate-pulse rounded-3xl border border-slate-200 bg-white/80 p-6 dark:border-slate-800 dark:bg-slate-900/80">
        <div className="h-4 w-24 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="mt-4 h-8 w-2/3 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="mt-3 h-3 w-full rounded bg-slate-200 dark:bg-slate-800" />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 2 }).map((_, index) => (
          <div key={index} className="animate-pulse rounded-3xl border border-slate-200 bg-white/80 p-6 dark:border-slate-800 dark:bg-slate-900/80">
            <div className="h-4 w-20 rounded bg-slate-200 dark:bg-slate-800" />
            <div className="mt-6 h-10 w-full rounded bg-slate-200 dark:bg-slate-800" />
          </div>
        ))}
      </div>
    </div>
  );
}
