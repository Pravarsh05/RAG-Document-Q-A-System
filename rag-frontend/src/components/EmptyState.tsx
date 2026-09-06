export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-sm border border-dashed border-ink-700 py-16 text-center">
      <div className="font-display text-base font-medium text-mist-100">{title}</div>
      <p className="mt-1 max-w-sm font-body text-sm text-mist-400">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
