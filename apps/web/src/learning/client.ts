import { z } from 'zod';
import {
  artifactOperationSchema, capabilitiesSchema, coachSchema, datasetSchema, evaluationSchema, grantSchema, jobSchema,
  projectSchema, recordSchema, referenceCollectionSchema, releaseSchema, resource, resourceList, teachingSchema,
  type CreateProjectBody, type JogBody, type LearningApi, type LearningRecord, type TrainingBody,
} from './contracts';

export type LearningRequest = <T>(
  path: string, schema: z.ZodType<T>,
  options?: { body?: unknown; method?: 'GET' | 'POST'; etag?: string; signal?: AbortSignal },
) => Promise<T>;

const key = (id: string) => encodeURIComponent(id);
export class LearningClient implements LearningApi {
  constructor(
    private readonly request: LearningRequest,
    private readonly download: (path: string, signal?: AbortSignal) => Promise<Blob>,
  ) {}
  capabilities(signal?: AbortSignal) { return this.request('/api/learning/capabilities', capabilitiesSchema, { signal }); }
  projects(signal?: AbortSignal) { return this.request('/api/learning/projects', resourceList(projectSchema), { signal }); }
  createProject(body: CreateProjectBody, signal?: AbortSignal) { return this.request('/api/learning/projects', resource(projectSchema), { method: 'POST', body, signal }); }
  records(id: string, kind: LearningRecord['kind'], signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/records?kind=${kind}`, resourceList(recordSchema), { signal });
  }
  teach(id: string, body: Parameters<LearningApi['teach']>[1], etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/teaching-sessions`, resource(teachingSchema), { method: 'POST', body, etag, signal });
  }
  teaching(id: string, signal?: AbortSignal) { return this.request(`/api/teaching-sessions/${key(id)}`, resource(teachingSchema), { signal }); }
  startReference(id: string, body: Parameters<LearningApi['startReference']>[1], etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/reference-collections`, resource(referenceCollectionSchema), { method: 'POST', body, etag, signal });
  }
  reference(id: string, signal?: AbortSignal) {
    return this.request(`/api/reference-collections/${key(id)}`, resource(referenceCollectionSchema), { signal });
  }
  cancelReference(id: string, signal?: AbortSignal) {
    return this.request(`/api/reference-collections/${key(id)}/cancel`, resource(referenceCollectionSchema), { method: 'POST', signal });
  }
  arm(id: string, body: JogBody, etag: string, signal?: AbortSignal) {
    return this.request(`/api/teaching-sessions/${key(id)}/arm`, resource(grantSchema), { method: 'POST', body, etag, signal });
  }
  jog(id: string, body: JogBody, etag: string, signal?: AbortSignal) {
    return this.request(`/api/teaching-sessions/${key(id)}/jog`, resource(teachingSchema), { method: 'POST', body, etag, signal });
  }
  teachingControl(id: string, action: 'finish' | 'cancel', body: Parameters<LearningApi['teachingControl']>[2], etag: string, signal?: AbortSignal) {
    return this.request(`/api/teaching-sessions/${key(id)}/${action}`, resource(teachingSchema), { method: 'POST', body, etag, signal });
  }
  seal(id: string, body: Parameters<LearningApi['seal']>[1], etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/datasets`, resource(z.union([datasetSchema, artifactOperationSchema])), { method: 'POST', body, etag, signal });
  }
  artifactOperation(id: string, signal?: AbortSignal) {
    return this.request(`/api/learning/artifact-operations/${key(id)}`, resource(artifactOperationSchema), { signal });
  }
  dataset(id: string, signal?: AbortSignal) {
    return this.request(`/api/learning/datasets/${key(id)}`, resource(datasetSchema), { signal });
  }
  train(id: string, body: TrainingBody, etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/train`, resource(jobSchema), { method: 'POST', body, etag, signal });
  }
  evaluate(id: string, body: Parameters<LearningApi['evaluate']>[1], etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/evaluate`, resource(evaluationSchema), { method: 'POST', body, etag, signal });
  }
  job(id: string, signal?: AbortSignal) { return this.request(`/api/learning/jobs/${key(id)}`, resource(jobSchema), { signal }); }
  reportDocument(id: string, signal?: AbortSignal) { return this.download(`/api/learning/jobs/${key(id)}/report`, signal); }
  cancelJob(id: string, requestId: string, etag: string, signal?: AbortSignal) {
    return this.request(`/api/learning/jobs/${key(id)}/cancel`, resource(jobSchema), { method: 'POST', body: { request_id: requestId }, etag, signal });
  }
  release(body: Parameters<LearningApi['release']>[0], etag: string, signal?: AbortSignal) {
    return this.request('/api/policy-releases', resource(releaseSchema), { method: 'POST', body, etag, signal });
  }
  coach(id: string, body: Parameters<LearningApi['coach']>[1], signal?: AbortSignal) {
    return this.request(`/api/learning/projects/${key(id)}/coach`, coachSchema, { method: 'POST', body, signal });
  }
}
