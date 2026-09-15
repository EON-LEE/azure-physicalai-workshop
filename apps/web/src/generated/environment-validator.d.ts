import type { ValidateFunction } from 'ajv';
import type { EnvironmentJsonSchema } from '../api/contracts';

declare const validate: ValidateFunction;
export default validate;
export const expectedSchema: EnvironmentJsonSchema;
