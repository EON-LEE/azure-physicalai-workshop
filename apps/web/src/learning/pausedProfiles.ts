import { z } from 'zod';

export const PAUSED_PROFILE_V1 = 'franka-position-hold-10hz-paused-v1';
export const PAUSED_PROFILE_V2 = 'franka-position-hold-10hz-paused-v2';
export const pausedProfileSchema = z.enum([PAUSED_PROFILE_V1, PAUSED_PROFILE_V2]);
export type PausedProfileId = z.infer<typeof pausedProfileSchema>;
export const pausedProfileSteps: Record<PausedProfileId, number> = {
  [PAUSED_PROFILE_V1]: 1800, [PAUSED_PROFILE_V2]: 3600,
};
export const isPausedProfile = (value: unknown): value is PausedProfileId =>
  value === PAUSED_PROFILE_V1 || value === PAUSED_PROFILE_V2;

export const pausedEnvironmentSchema = z.object({
  schema: z.enum(['physicalai.paused-simulation/v1', 'physicalai.paused-simulation/v2']),
  execution_timing: z.literal('paused_simulation'), profile_id: pausedProfileSchema,
  max_simulation_seconds: z.number().int().min(1).max(60),
  max_wall_seconds: z.number().int().min(1).max(600),
}).strict().refine((value) =>
  value.schema === (value.profile_id === PAUSED_PROFILE_V1 ? 'physicalai.paused-simulation/v1' : 'physicalai.paused-simulation/v2') &&
  value.max_simulation_seconds <= pausedProfileSteps[value.profile_id] / 60,
);
