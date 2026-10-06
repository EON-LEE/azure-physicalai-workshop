import { z } from 'zod';
import { bootstrapReportSchema, reportSchema } from '../learning/contracts';
import { publicSimulationReportSchema } from '../learning/simulationReports';

const sha = z.string().regex(/^[a-f0-9]{64}$/);
const checkpointSampleSchema = z.object({
  optimizer_steps: z.number().int().positive(),
  loss: z.number().nonnegative().nullable(),
  measured_at: z.iso.datetime({ offset: true }),
  checkpoint_sha256: sha,
});

export const publicLearningSchema = z.object({
  api_version: z.literal('public-learning-v1'),
  status: z.enum(['not_published', 'published']),
  publication: z.object({
    title: z.string(), task: z.string(), policy_type: z.string(),
    recorded_at: z.iso.datetime({ offset: true }),
    evaluation_status: z.enum(['succeeded', 'failed', 'cancelled', 'timed_out']),
    data_provenance: z.object({ human_teleop: z.number().int().nonnegative(), reference_controller: z.number().int().nonnegative(), learned: z.number().int().nonnegative() }),
    training: z.object({
      optimizer_steps: z.number().int().positive(), model_sha256: z.string(), parent_model_sha256: z.string(),
      dataset_sha256: z.string(), created_at: z.iso.datetime({ offset: true }), loss: z.number().nullable(),
      // Verified checkpoint-time progress samples, oldest first. Empty means no published
      // checkpoint evidence exists yet; the UI must show this as unavailable, never a flat/fake line.
      checkpoints: z.array(checkpointSampleSchema).max(200),
    }).superRefine((value, context) => {
      const steps = value.checkpoints.map((item) => item.optimizer_steps);
      if (steps.some((step, index) => index > 0 && step <= steps[index - 1]!)) {
        context.addIssue({ code: 'custom', message: 'Checkpoint progress must be strictly ordered by step.' });
      }
      if (steps.some((step) => step > value.optimizer_steps)) {
        context.addIssue({ code: 'custom', message: 'Checkpoint progress cannot exceed the final reconciled step count.' });
      }
    }),
    comparison: z.union([
      publicSimulationReportSchema,
      reportSchema.omit({ artifact_id: true }), bootstrapReportSchema.omit({ artifact_id: true }),
    ]),
    execution: z.literal('recorded_evaluation_not_live'),
  }).nullable(),
}).refine((data) => (data.status === 'published') === (data.publication !== null));
