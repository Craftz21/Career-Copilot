import ProcessingStatus from '@/components/ProcessingStatus';

export default async function ProcessingPage({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return <ProcessingStatus sessionId={sessionId} />;
}
