import { Badge } from "@mantine/core";

import type { JobState } from "../hooks/useAdminJobs";

const COLOR: Record<JobState, string> = {
  queued: "gray",
  running: "blue",
  done: "green",
  failed: "red",
  cancelled: "orange",
  crashed: "grape",
};

export function JobStatusChip({ state }: { state: JobState }) {
  return (
    <Badge color={COLOR[state]} variant="filled" radius="sm">
      {state}
    </Badge>
  );
}
