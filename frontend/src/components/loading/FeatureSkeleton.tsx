export function FeatureSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {Array.from({ length: 3 }).map((_, index) => (
        <div key={index} className="animate-pulse rounded-3xl border border-slate-200 bg-white/80 p-6 dark:border-slate-800 dark:bg-slate-900/80">
          <div className="h-10 w-10 rounded-2xl bg-slate-200 dark:bg-slate-800" />
          <div className="mt-4 h-4 w-24 rounded bg-slate-200 dark:bg-slate-800" />
          <div className="mt-3 h-3 w-full rounded bg-slate-200 dark:bg-slate-800" />
          <div className="mt-2 h-3 w-5/6 rounded bg-slate-200 dark:bg-slate-800" />
        </div>
      ))}
    </div>
  );
}
