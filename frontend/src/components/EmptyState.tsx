import { Inbox } from 'lucide-react';

type EmptyStateProps = {
  title: string;
  description: string;
  action?: React.ReactNode;
};

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="rounded-3xl border border-dashed border-slate-300 bg-white/70 p-8 text-center dark:border-slate-700 dark:bg-slate-900/70">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
        <Inbox className="h-6 w-6" />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-slate-600 dark:text-slate-400">{description}</p>
      {action ? <div className="mt-6 flex justify-center">{action}</div> : null}
    </div>
  );
}
