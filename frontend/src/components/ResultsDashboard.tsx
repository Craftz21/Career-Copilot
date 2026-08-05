'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  PieChart,
  Pie,
} from 'recharts';
import api from '@/lib/api';

type ResultsPayload = {
  session_id: string;
  target_role: string;
  readiness_score: number;
  raw_readiness_score: number;
  gap_data: {
    matched_skills?: Array<{ display_name?: string }>;
    missing_skills?: Array<{ display_name?: string }>;
    bonus_skills?: Array<{ display_name?: string }>;
    category_breakdown?: Record<string, number>;
    score_contributors?: Array<{ label?: string; value?: number }>;
  };
  roadmap: Record<string, unknown>;
  recruiter_summary: Record<string, unknown>;
  evidence_map: Record<string, unknown>;
  project_recommendations: Array<Record<string, unknown>>;
  soft_skill_inferences: Array<Record<string, unknown>>;
  skill_evidence: Record<string, unknown>;
  score_contributors: Array<Record<string, unknown>>;
  candidate_profile: Record<string, unknown>;
  role_fit: Array<Record<string, unknown>>;
  shortest_path: Record<string, unknown>;
  session: {
    session_id?: string;
    target_role?: string;
    status?: string;
    readiness_score?: number;
    expires_at?: string;
  };
  jd_analysis?: {
    jd_skills?: unknown;
    summary?: string;
    confidence?: number;
  } | null;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function normalizeText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === 'string' && item.trim() !== '').map((item) => item.trim());
}

type RoadmapResource = {
  title: string;
  platform?: string;
  url?: string;
  estimatedHours?: number;
  type?: string;
};

type RoadmapWeekItem = {
  weekNumber: number;
  title: string;
  focus: string;
  skills: string[];
  tasks: string[];
  resources: RoadmapResource[];
  estimatedEffort: string;
};

function normalizeRoadmapResource(resource: unknown): RoadmapResource | null {
  if (!isRecord(resource)) {
    return null;
  }

  const title = normalizeText(resource.title) ?? normalizeText(resource.name) ?? 'Learning resource';
  const platform = normalizeText(resource.platform);
  const url = normalizeText(resource.url);
  const estimatedHours = typeof resource.estimated_hours === 'number'
    ? resource.estimated_hours
    : typeof resource.estimatedHours === 'number'
      ? resource.estimatedHours
      : undefined;
  const type = normalizeText(resource.type) ?? 'resource';

  return {
    title,
    platform: platform ?? undefined,
    url: url ?? undefined,
    estimatedHours,
    type: type ?? undefined,
  };
}

function buildRoadmapTimeline(roadmap: Record<string, unknown> | undefined): RoadmapWeekItem[] {
  const weeks = Array.isArray(roadmap?.weeks)
    ? roadmap.weeks
    : Array.isArray(roadmap?.milestones)
      ? roadmap.milestones
      : Array.isArray(roadmap?.steps)
        ? roadmap.steps
        : [];

  if (weeks.length > 0) {
    return weeks.map((entry, index) => {
      const value = isRecord(entry) ? entry : {};
      const weekNumber = typeof value.week_number === 'number'
        ? value.week_number
        : typeof value.weekNumber === 'number'
          ? value.weekNumber
          : index + 1;
      const title = normalizeText(value.title) ?? `Week ${weekNumber}`;
      const focus = normalizeText(value.focus) ?? title;
      const skills = normalizeStringArray(value.skills);
      const tasks = normalizeStringArray(value.tasks);
      const resources = (Array.isArray(value.resources) ? value.resources : [])
        .map(normalizeRoadmapResource)
        .filter((resource): resource is RoadmapResource => Boolean(resource));
      const estimatedEffort = normalizeText(value.estimated_effort)
        ?? normalizeText(value.estimatedEffort)
        ?? (resources.reduce((total, resource) => total + (resource.estimatedHours ?? 0), 0) > 0
          ? `${resources.reduce((total, resource) => total + (resource.estimatedHours ?? 0), 0)} hrs`
          : 'Flexible');

      return {
        weekNumber,
        title,
        focus,
        skills: skills.length > 0 ? skills : tasks,
        tasks,
        resources,
        estimatedEffort,
      };
    });
  }

  const phases = Array.isArray(roadmap?.phases) ? roadmap.phases : [];
  return phases.map((entry, index) => {
    const value = isRecord(entry) ? entry : {};
    const phaseNumber = typeof value.phase_number === 'number' ? value.phase_number : index + 1;
    const title = normalizeText(value.title) ?? `Phase ${phaseNumber}`;
    const focus = normalizeText(value.goal) ?? title;
    const actions = normalizeStringArray(value.actions);
    const resources = (Array.isArray(value.resources) ? value.resources : [])
      .map(normalizeRoadmapResource)
      .filter((resource): resource is RoadmapResource => Boolean(resource));

    return {
      weekNumber: phaseNumber,
      title,
      focus,
      skills: actions,
      tasks: actions,
      resources,
      estimatedEffort: normalizeText(value.duration_weeks)
        ? `${normalizeText(value.duration_weeks)} weeks`
        : 'Flexible',
    };
  });
}

function buildRadarData(skills: Array<{ display_name?: string }>, targetSkills: Array<{ display_name?: string }>) {
  const userNames = new Set(skills.map((skill) => skill.display_name?.toLowerCase() || ''));
  const targetNames = targetSkills.map((skill) => skill.display_name?.toLowerCase() || '');
  const categories = Array.from(new Set([...targetNames, ...Array.from(userNames)]))
    .filter(Boolean)
    .slice(0, 6);

  return categories.map((name) => ({
    subject: name,
    user: userNames.has(name) ? 85 : 40,
    target: 100,
  }));
}

function buildTimelineData(roadmap: Record<string, unknown> | undefined) {
  const entries = Array.isArray(roadmap?.milestones)
    ? roadmap.milestones
    : Array.isArray(roadmap?.steps)
      ? roadmap.steps
      : [];

  return entries.map((entry: unknown, index: number) => {
    const value = entry as Record<string, unknown>;
    const title = typeof value?.title === 'string' ? value.title : `Step ${index + 1}`;
    const description = typeof value?.description === 'string' ? value.description : 'Milestone';
    return {
      name: title,
      effort: 1 + index,
      description,
    };
  });
}

export default function ResultsDashboard() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params?.sessionId;
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [completedWeeks, setCompletedWeeks] = useState<Record<number, boolean>>({});

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    let active = true;
    const loadResults = async () => {
      try {
        const { data } = await api.get<ResultsPayload>(`/v1/results/${sessionId}`);
        if (!active) {
          return;
        }
        setResults(data);
      } catch {
        if (!active) {
          return;
        }
        setError('Unable to load analysis results. Please try again.');
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadResults();
    return () => {
      active = false;
    };
  }, [sessionId]);

  if (loading) {
    return (
      <main className="min-h-screen bg-slate-50 px-6 py-16">
        <div className="mx-auto max-w-4xl rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
          <p className="text-slate-600">Loading your analysis…</p>
        </div>
      </main>
    );
  }

  if (error || !results) {
    return (
      <main className="min-h-screen bg-slate-50 px-6 py-16">
        <div className="mx-auto max-w-4xl rounded-3xl border border-red-200 bg-white p-8 shadow-sm">
          <h1 className="text-2xl font-semibold text-slate-900">Results unavailable</h1>
          <p className="mt-3 text-slate-600">{error || 'No analysis results were found for this session.'}</p>
        </div>
      </main>
    );
  }

  const matchedSkills = results.gap_data?.matched_skills ?? [];
  const missingSkills = results.gap_data?.missing_skills ?? [];
  const bonusSkills = results.gap_data?.bonus_skills ?? [];
  const readinessScore = Math.round(results.readiness_score ?? 0);
  const skillComparison = buildRadarData(matchedSkills, missingSkills);
  const roadmapTimeline = buildRoadmapTimeline(results.roadmap);
  const roadmapTitle = normalizeText(results.roadmap?.title) ?? 'Your personalized roadmap';
  const roadmapSummary = normalizeText(results.roadmap?.summary) ?? 'A week-by-week plan to help you close your skill gaps.';

  const barData = [
    { name: 'Matched', value: matchedSkills.length, fill: '#6366f1' },
    { name: 'Missing', value: missingSkills.length, fill: '#f59e0b' },
    { name: 'Bonus', value: bonusSkills.length, fill: '#0f766e' },
  ];

  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <section className="overflow-hidden rounded-[2rem] border border-slate-200/80 bg-white/80 p-8 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80 sm:p-10">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="inline-flex rounded-full border border-slate-200 bg-slate-100/80 px-3 py-1 text-sm font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-800/80 dark:text-slate-300">
                Analysis complete
              </div>
              <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950 dark:text-white sm:text-4xl">
                {results.target_role || results.session?.target_role || 'Your resume analysis'}
              </h1>
              <p className="mt-3 max-w-2xl text-base text-slate-600 dark:text-slate-400">
                Your results are now presented as interactive analytics for skills, gaps, and roadmap progress.
              </p>
            </div>
            <div className="rounded-3xl bg-slate-900 px-6 py-5 text-white shadow-lg dark:bg-white dark:text-slate-900">
              <div className="text-sm text-slate-300 dark:text-slate-600">Readiness score</div>
              <div className="text-3xl font-semibold">{readinessScore}%</div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Readiness gauge</h2>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">A circular view of your current profile strength.</p>
              </div>
            </div>
            <div className="mt-6 flex h-64 items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={[{ name: 'readiness', value: readinessScore }, { name: 'gap', value: 100 - readinessScore }]}
                    dataKey="value"
                    innerRadius={70}
                    outerRadius={95}
                    startAngle={180}
                    endAngle={0}
                    animationDuration={1200}
                    animationBegin={0}
                  >
                    <Cell fill="#6366f1" />
                    <Cell fill="#e2e8f0" />
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mx-auto max-w-[180px] -mt-32 text-center">
              <div className="text-4xl font-semibold text-slate-900 dark:text-white">{readinessScore}%</div>
              <div className="mt-2 text-sm text-slate-600 dark:text-slate-400">Current readiness</div>
            </div>
          </div>

          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Skill comparison</h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">A radar view of your strengths versus the target role.</p>
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={skillComparison}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar name="You" dataKey="user" stroke="#6366f1" fill="#6366f1" fillOpacity={0.4} animationDuration={1200} />
                  <Radar name="Target" dataKey="target" stroke="#0f766e" fill="#0f766e" fillOpacity={0.2} animationDuration={1200} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Skill balance</h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Matched, missing, and bonus skills at a glance.</p>
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]} animationDuration={1200}>
                    {barData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Learning roadmap</h2>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">A practical timeline built from the roadmap returned by the backend.</p>
              </div>
              <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                {roadmapTimeline.length > 0 ? `${roadmapTimeline.length} weeks` : 'Pending'}
              </div>
            </div>

            {roadmapTimeline.length > 0 ? (
              <div className="mt-6 space-y-4">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950/60">
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">{roadmapTitle}</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{roadmapSummary}</p>
                </div>

                <div className="relative ml-2 space-y-4 border-l border-slate-200 pl-6 dark:border-slate-800">
                  {roadmapTimeline.map((week) => (
                    <div key={`${week.weekNumber}-${week.title}`} className="relative">
                      <span className="absolute -left-[1.6rem] top-5 h-3.5 w-3.5 rounded-full border-4 border-white bg-indigo-600 shadow-sm dark:border-slate-900" />
                      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900/70 sm:p-5">
                        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                          <div className="space-y-3">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300">
                                Week {week.weekNumber}
                              </span>
                              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{week.focus}</span>
                            </div>

                            <div>
                              <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Skills to learn</h3>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {week.skills.length > 0 ? week.skills.map((skill) => (
                                  <span key={`${week.weekNumber}-${skill}`} className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                                    {skill}
                                  </span>
                                )) : (
                                  <span className="text-sm text-slate-600 dark:text-slate-400">No skills were listed for this week.</span>
                                )}
                              </div>
                            </div>
                          </div>

                          <label className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
                            <input
                              type="checkbox"
                              checked={Boolean(completedWeeks[week.weekNumber])}
                              onChange={() => setCompletedWeeks((current) => ({
                                ...current,
                                [week.weekNumber]: !current[week.weekNumber],
                              }))}
                              className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                            />
                            Complete
                          </label>
                        </div>

                        <div className="mt-4 grid gap-4 md:grid-cols-2">
                          <div>
                            <h4 className="text-sm font-semibold text-slate-900 dark:text-white">Estimated effort</h4>
                            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{week.estimatedEffort}</p>
                          </div>
                          <div>
                            <h4 className="text-sm font-semibold text-slate-900 dark:text-white">Learning resources</h4>
                            {week.resources.length > 0 ? (
                              <ul className="mt-2 space-y-2 text-sm text-slate-600 dark:text-slate-400">
                                {week.resources.map((resource) => (
                                  <li key={`${week.weekNumber}-${resource.title}`}>
                                    {resource.url ? (
                                      <a href={resource.url} target="_blank" rel="noreferrer" className="font-medium text-indigo-600 hover:underline dark:text-indigo-400">
                                        {resource.title}
                                      </a>
                                    ) : (
                                      <span>{resource.title}</span>
                                    )}
                                    {resource.platform ? <span className="ml-2 text-slate-500">({resource.platform})</span> : null}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">No resources were attached for this week.</p>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-6 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-950/50 dark:text-slate-400">
                <p className="font-semibold text-slate-900 dark:text-white">No roadmap is available yet</p>
                <p className="mt-2">The backend did not return a learning roadmap for this session yet, so there is nothing to display here. Once the analysis finishes, the week-by-week plan will appear automatically.</p>
              </div>
            )}
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Matched skills</h2>
            <div className="mt-4 flex flex-wrap gap-2">
              {matchedSkills.length > 0 ? matchedSkills.map((skill, index) => (
                <span key={`${skill.display_name ?? 'skill'}-${index}`} className="rounded-full bg-emerald-100 px-3 py-1 text-sm text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300">
                  {skill.display_name || 'Skill'}
                </span>
              )) : <p className="text-sm text-slate-600 dark:text-slate-400">No matched skills were returned.</p>}
            </div>
          </div>

          <div className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Missing skills</h2>
            <div className="mt-4 flex flex-wrap gap-2">
              {missingSkills.length > 0 ? missingSkills.map((skill, index) => (
                <span key={`${skill.display_name ?? 'missing'}-${index}`} className="rounded-full bg-amber-100 px-3 py-1 text-sm text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">
                  {skill.display_name || 'Skill'}
                </span>
              )) : <p className="text-sm text-slate-600 dark:text-slate-400">No missing skills were returned.</p>}
            </div>
          </div>
        </section>

        <section className="rounded-[2rem] border border-slate-200/80 bg-white/80 p-6 shadow-[0_30px_100px_-40px_rgba(15,23,42,0.35)] backdrop-blur dark:border-slate-800/80 dark:bg-slate-900/80">
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Bonus skills</h2>
          <div className="mt-4 flex flex-wrap gap-2">
            {bonusSkills.length > 0 ? bonusSkills.map((skill, index) => (
              <span key={`${skill.display_name ?? 'bonus'}-${index}`} className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                {skill.display_name || 'Skill'}
              </span>
            )) : <p className="text-sm text-slate-600 dark:text-slate-400">No bonus skills were returned.</p>}
          </div>
        </section>
      </div>
    </main>
  );
}
