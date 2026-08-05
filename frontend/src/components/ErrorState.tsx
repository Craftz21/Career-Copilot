import { AlertTriangle } from 'lucide-react';

type ErrorStateProps = {
  title: string;
  description: string;
  action?: React.ReactNode;
};

export function ErrorState({ title, description, action }: ErrorStateProps) {
  return (
    <div className="rounded-3xl border border-red-200 bg-red-50 p-8 text-center dark:border-red-900/60 dark:bg-red-950/30">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-red-100 text-red-700 dark:bg-red-900/60 dark:text-red-200">
        <AlertTriangle className="h-6 w-6" />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-red-900 dark:text-red-200">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-red-700 dark:text-red-300">{description}</p>
      {action ? <div className="mt-6 flex justify-center">{action}</div> : null}
    </div>
  );
}
