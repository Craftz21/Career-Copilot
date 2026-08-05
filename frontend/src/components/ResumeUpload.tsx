'use client';

import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import { isAxiosError } from 'axios';
import api from '@/lib/api';

const ACCEPTED_MIME_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/msword': ['.doc'],
};

const FALLBACK_ROLES = [
  'Software Engineer',
  'Backend Software Engineer',
  'Frontend Developer',
  'Full Stack Developer',
  'Data Scientist',
  'Machine Learning Engineer',
  'Product Manager',
  'DevOps Engineer',
  'QA Engineer',
];

async function fetchRoleSuggestions(): Promise<string[]> {
  const endpoints = ['/v1/roles', '/roles', '/api/roles'];

  for (const endpoint of endpoints) {
    try {
      const response = await api.get(endpoint);
      const payload = response.data;

      if (Array.isArray(payload)) {
        return payload.filter((value: unknown): value is string => typeof value === 'string');
      }

      if (payload && Array.isArray(payload.roles)) {
        return payload.roles.filter((value: unknown): value is string => typeof value === 'string');
      }
    } catch {
      // Ignore and try the next endpoint.
    }
  }

  return FALLBACK_ROLES;
}

export default function ResumeUpload() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetRole, setTargetRole] = useState('');
  const [roleQuery, setRoleQuery] = useState('');
  const [duration, setDuration] = useState('3 months');
  const [roles, setRoles] = useState<string[]>(FALLBACK_ROLES);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const loadRoles = async () => {
      const suggestions = await fetchRoleSuggestions();
      if (active) {
        setRoles(suggestions);
      }
    };

    loadRoles();

    return () => {
      active = false;
    };
  }, []);

  const filteredRoles = useMemo(() => {
    const normalized = roleQuery.trim().toLowerCase();
    if (!normalized) {
      return roles.slice(0, 8);
    }

    return roles.filter((role) => role.toLowerCase().includes(normalized)).slice(0, 8);
  }, [roleQuery, roles]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setSubmitError(null);
    setValidationErrors((current) => ({ ...current, file: '' }));

    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      const reason = rejection.errors?.[0]?.message || 'Unsupported file.';
      setValidationErrors((current) => ({ ...current, file: reason }));
      setSelectedFile(null);
      return;
    }

    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_MIME_TYPES,
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024,
    multiple: false,
  });

  const validateForm = () => {
    const nextErrors: Record<string, string> = {};

    if (!selectedFile) {
      nextErrors.file = 'Please upload a PDF or DOCX resume.';
    }

    if (!targetRole.trim()) {
      nextErrors.targetRole = 'Please enter the role you want to target.';
    } else if (targetRole.trim().length < 2) {
      nextErrors.targetRole = 'Please enter a longer role title.';
    }

    setValidationErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitError(null);

    if (!validateForm() || !selectedFile) {
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_role', targetRole.trim());
    formData.append('duration', duration);

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await api.post('/v1/resume/upload', formData, {
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentage);
          }
        },
      });

      const sessionId = response.data?.session_id;
      if (sessionId) {
        router.push(`/processing/${sessionId}`);
        return;
      }

      setSubmitError('Upload completed, but no session was returned.');
    } catch (error: unknown) {
      if (isAxiosError(error) && error.response?.data?.detail) {
        setSubmitError(error.response.data.detail);
      } else {
        setSubmitError('Upload failed. Please try again.');
      }
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-16">
      <div className="mx-auto flex max-w-3xl flex-col gap-6 rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="space-y-2">
          <div className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-700">
            Resume upload
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Analyze your resume against a target role
          </h1>
          <p className="text-base text-slate-600">
            Upload a PDF or DOCX resume and we will evaluate your readiness, surface skill gaps, and generate a roadmap.
          </p>
        </div>

        <form className="space-y-6" onSubmit={handleSubmit} noValidate>
          <div className="space-y-2">
            <label htmlFor="targetRole" className="text-sm font-medium text-slate-700">
              Target role
            </label>
            <input
              id="targetRole"
              name="targetRole"
              type="text"
              value={targetRole}
              onChange={(event) => {
                const nextValue = event.target.value;
                setTargetRole(nextValue);
                setRoleQuery(nextValue);
                setValidationErrors((current) => ({ ...current, targetRole: '' }));
              }}
              placeholder="e.g. Backend Software Engineer"
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm shadow-sm outline-none transition focus:border-slate-500"
              autoComplete="off"
            />

            {filteredRoles.length > 0 && (
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-2">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Suggested roles
                </div>
                <div className="flex flex-wrap gap-2">
                  {filteredRoles.map((role) => (
                    <button
                      key={role}
                      type="button"
                      className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-700 transition hover:border-slate-400"
                      onClick={() => {
                        setTargetRole(role);
                        setRoleQuery(role);
                        setValidationErrors((current) => ({ ...current, targetRole: '' }));
                      }}
                    >
                      {role}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {validationErrors.targetRole && (
              <p className="text-sm text-red-600">{validationErrors.targetRole}</p>
            )}
          </div>

          <div className="space-y-2">
            <label htmlFor="duration" className="text-sm font-medium text-slate-700">
              Preparation window
            </label>
            <select
              id="duration"
              name="duration"
              value={duration}
              onChange={(event) => setDuration(event.target.value)}
              className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm shadow-sm outline-none transition focus:border-slate-500"
            >
              <option value="4 weeks">4 weeks</option>
              <option value="2 months">2 months</option>
              <option value="3 months">3 months</option>
              <option value="6 months">6 months</option>
              <option value="12 months">12 months</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-700">Resume file</label>
            <div
              {...getRootProps({})}
              className={`rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
                isDragActive ? 'border-slate-500 bg-slate-100' : 'border-slate-300 bg-slate-50'
              }`}
            >
              <input {...getInputProps()} />
              <p className="text-sm font-medium text-slate-700">
                {isDragActive ? 'Drop your resume here' : 'Drag and drop your resume here, or click to browse'}
              </p>
              <p className="mt-2 text-sm text-slate-500">Accepted formats: PDF, DOCX, DOC. Maximum size: 5 MB.</p>
              {selectedFile && (
                <p className="mt-4 font-medium text-slate-800">Selected: {selectedFile.name}</p>
              )}
            </div>
            {validationErrors.file && <p className="text-sm text-red-600">{validationErrors.file}</p>}
          </div>

          {isUploading && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-2 flex items-center justify-between text-sm text-slate-600">
                <span>Uploading resume…</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-200">
                <div className="h-2 rounded-full bg-slate-900 transition-all" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}

          {submitError && <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{submitError}</p>}

          <button
            type="submit"
            disabled={isUploading}
            className="w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isUploading ? 'Uploading…' : 'Analyze my resume'}
          </button>
        </form>
      </div>
    </main>
  );
}
