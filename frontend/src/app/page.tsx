"use client";

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { ArrowRight, BrainCircuit, ShieldCheck, Sparkles, TrendingUp, Zap } from 'lucide-react';
import { getHealth } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { FeatureSkeleton } from '@/components/loading/FeatureSkeleton';
import { EmptyState } from '@/components/EmptyState';
import { ErrorState } from '@/components/ErrorState';

const features = [
  {
    title: 'Skills intelligence',
    description: 'Map your experience to target roles with precise role-fit analysis.',
    icon: BrainCircuit,
  },
  {
    title: 'Outcome-driven plans',
    description: 'Turn gaps into a focused roadmap with measurable milestones.',
    icon: TrendingUp,
  },
  {
    title: 'Secure and reliable',
    description: 'Keep your data protected while you move toward the next opportunity.',
    icon: ShieldCheck,
  },
];

export default function Home() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
  });

  return (
    <div className="space-y-10">
      <section className="relative overflow-hidden rounded-[2rem] border border-slate-200/80 bg-white/75 p-8 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.45)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/75 sm:p-10 lg:p-14">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(129,140,248,0.24),_transparent_35%)]" />
        <div className="relative grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-100/70 px-3 py-1 text-sm font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-800/70 dark:text-slate-300">
              <Sparkles className="h-4 w-4" />
              Modern career growth for ambitious teams
            </div>
            <h1 className="mt-6 text-4xl font-semibold tracking-tight text-slate-950 dark:text-white sm:text-5xl lg:text-6xl">
              Turn your experience into your next opportunity.
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600 dark:text-slate-400">
              CareerCopilot blends AI-powered resume analysis, actionable skill-gap insights, and a polished experience for professionals who want to move faster.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Button href="/upload" size="lg" className="group">
                Start your analysis <ArrowRight className="ml-2 h-4 w-4 transition group-hover:translate-x-0.5" />
              </Button>
              <Button href="/upload" variant="secondary" size="lg">
                Try the upload flow
              </Button>
            </div>
          </div>

          <Card className="lg:p-8" title="Live platform snapshot" description="Built for responsive, accessible experiences across desktop and mobile.">
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950/70">
                <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400">
                  <span>Analysis readiness</span>
                  <span className="font-semibold text-slate-900 dark:text-slate-100">92%</span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-slate-200 dark:bg-slate-800">
                  <div className="h-2 w-[92%] rounded-full bg-gradient-to-r from-indigo-500 via-sky-500 to-cyan-400" />
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl bg-slate-900 p-4 text-white dark:bg-slate-800">
                  <div className="flex items-center gap-2 text-sm text-slate-300">
                    <Zap className="h-4 w-4" />
                    Faster feedback
                  </div>
                  <p className="mt-2 text-2xl font-semibold">2x faster</p>
                </div>
                <div className="rounded-2xl border border-slate-200 p-4 dark:border-slate-800">
                  <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                    <BrainCircuit className="h-4 w-4" />
                    AI guidance
                  </div>
                  <p className="mt-2 text-2xl font-semibold text-slate-900 dark:text-slate-100">Always on</p>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <Card title="Why teams choose CareerCopilot" description="A consistent design system with accessible interactions and polished states.">
          <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-1">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <div key={feature.title} className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4 dark:border-slate-800 dark:bg-slate-950/70">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-900 text-white dark:bg-white dark:text-slate-900">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className="mt-4 font-semibold text-slate-900 dark:text-slate-100">{feature.title}</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </Card>

        <Card title="System health" description="Live environment feedback and graceful loading states.">
          {isLoading ? (
            <FeatureSkeleton />
          ) : error ? (
            <ErrorState title="Backend unavailable" description="The FastAPI service did not respond. The UI still remains polished and accessible while the backend recovers." action={<Button href="/upload" variant="secondary">Try the upload flow</Button>} />
          ) : data ? (
            <div className="rounded-3xl border border-emerald-200 bg-emerald-50 p-5 text-sm text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-300">
              <p className="font-semibold">Everything is healthy</p>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-2xl bg-slate-900/90 p-4 text-xs text-emerald-100">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          ) : (
            <EmptyState title="No health data yet" description="The backend has not returned a response yet. Try refreshing or checking the connection." />
          )}
        </Card>
      </section>

      <Card title="Designed for every step" description="From upload to roadmap, the experience stays polished, responsive, and accessible.">
        <div className="grid gap-4 md:grid-cols-3">
          {[
            ['Upload resume', 'Drop in your resume and target role in a guided experience.'],
            ['Review progress', 'Live status updates keep the user informed with polished skeletons.'],
            ['Receive insights', 'Explore results in a clean, client-side dashboard.'],
          ].map(([title, description]) => (
            <div key={title} className="rounded-2xl border border-slate-200 bg-white/70 p-5 dark:border-slate-800 dark:bg-slate-950/70">
              <h3 className="font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">{description}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
