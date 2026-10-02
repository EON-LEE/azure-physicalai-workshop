import { z } from 'zod';
import { bootstrapReportSchema, reportSchema } from '../learning/contracts';
import { publicSimulationReportSchema } from '../learning/simulationReports';

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
    }),
    comparison: z.union([
      publicSimulationReportSchema,
      reportSchema.omit({ artifact_id: true }), bootstrapReportSchema.omit({ artifact_id: true }),
    ]),
    execution: z.literal('recorded_evaluation_not_live'),
  }).nullable(),
}).refine((data) => (data.status === 'published') === (data.publication !== null));
