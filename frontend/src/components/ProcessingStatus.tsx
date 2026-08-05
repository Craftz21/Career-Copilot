'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';

type TaskStatusResponse = {
  session_id: string;
  status: string;
  progress_pct: number;
  progress_message: string | null;
  task_type: string | null;
  queued_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  result?: unknown;
  results_url?: string;
  error?: string;
};

export default function ProcessingStatus({ sessionId }: { sessionId: string }) {
  const router = useRouter();
  const [task, setTask] = useState<TaskStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      try {
        const { data } = await api.get<TaskStatusResponse>(`/v1/tasks/${sessionId}`);
        if (!active) {
          return;
        }

        setTask(data);
        setError(null);

        if (data.status === 'complete') {
          if (intervalId) {
            clearInterval(intervalId);
          }
          router.push(`/results/${sessionId}`);
          return;
        }

        if (data.status === 'failed') {
          if (intervalId) {
            clearInterval(intervalId);
          }
          setError(data.error || 'Analysis failed. Please try again.');
        }
      } catch (pollError) {
        if (!active) {
          return;
        }
        setError('Unable to load task status. Please refresh and try again.');
      }
    };

    poll();
    intervalId = setInterval(poll, 1000);

    return () => {
      active = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [router, sessionId]);

  if (error) {
    return (
      <main className="min-h-screen bg-slate-50 px-6 py-16">
        <div className="mx-auto max-w-2xl rounded-3xl border border-red-200 bg-white p-8 shadow-sm">
          <h1 className="text-2xl font-semibold text-slate-900">Analysis failed</h1>
          <p className="mt-3 text-slate-600">{error}</p>
          <button
            type="button"
            onClick={() => router.push('/upload')}
            className="mt-6 rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white"
          >
            Try again
          </button>
        </div>
      </main>
    );
  }

  const progress = task?.progress_pct ?? 0;

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-16">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="mb-6 inline-flex rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-700">
          Processing your analysis
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">Please wait while we analyze your resume</h1>
        <p className="mt-3 text-base text-slate-600">
          We are polling the existing FastAPI task endpoint and will redirect you to the results view as soon as it is ready.
        </p>

        <div className="mt-8 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex items-center justify-between text-sm text-slate-600">
            <span>{task?.progress_message || 'Queued for processing…'}</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200">
            <div className="h-2 rounded-full bg-slate-900 transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-slate-200 p-4 text-sm text-slate-600">
          <div className="flex items-center justify-between">
            <span>Status</span>
            <span className="font-medium text-slate-900">{task?.status || 'queued'}</span>
          </div>
          <div className="mt-2 flex items-center justify-between">
            <span>Session</span>
            <span className="font-medium text-slate-900">{sessionId}</span>
          </div>
        </div>
      </div>
    </main>
  );
}
