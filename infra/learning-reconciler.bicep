targetScope = 'resourceGroup'

@description('Source-only default: no scheduler resource exists until explicitly enabled and deployed.')
param enabled bool = false
param jobName string
param location string = resourceGroup().location
param managedEnvironmentId string

@description('Existing dedicated learning-worker UAMI; no API/simulator identity or new role grants.')
param workerIdentityResourceId string
param entraTenantId string
param workerAudience string
param registryServer string
param workerImageRepository string

@minLength(64)
@maxLength(64)
param workerImageSha256 string

param registryAccountUrl string
param registryContainer string
param captureAccountUrl string
param captureContainer string

@maxLength(20)
param reconciliationActorIds array = []

@description('Exact reviewed actor/job/specification SHA/configuration SHA/UTC deadline bindings; never wildcards.')
@maxLength(20)
param reconciliationTargets array = []

resource workerIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: last(split(workerIdentityResourceId, '/'))
  scope: resourceGroup(split(workerIdentityResourceId, '/')[2], split(workerIdentityResourceId, '/')[4])
}

resource reconciler 'Microsoft.App/jobs@2025-07-01' = if (enabled) {
  name: jobName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workerIdentityResourceId}': {}
    }
  }
  properties: {
    environmentId: managedEnvironmentId
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 120
      replicaRetryLimit: 0
      scheduleTriggerConfig: {
        cronExpression: '* * * * *'
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: [{
        server: registryServer
        identity: workerIdentityResourceId
      }]
    }
    template: {
      containers: [{
        name: 'learning-reconciler'
        image: '${registryServer}/${workerImageRepository}@sha256:${workerImageSha256}'
        command: ['/srv/apps/learning_worker/.venv/bin/python', '-m', 'apps.learning_worker.reconcile']
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [
          { name: 'LEARNING_WORKER_TENANT_ID', value: entraTenantId }
          { name: 'LEARNING_WORKER_AUDIENCE', value: workerAudience }
          { name: 'LEARNING_WORKER_MANAGED_IDENTITY_CLIENT_ID', value: workerIdentity.properties.clientId }
          { name: 'LEARNING_WORKER_ALLOWED_API_PRINCIPALS', value: '[]' }
          { name: 'LEARNING_WORKER_REGISTRY_ACCOUNT_URL', value: registryAccountUrl }
          { name: 'LEARNING_WORKER_REGISTRY_CONTAINER', value: registryContainer }
          { name: 'LEARNING_WORKER_CAPTURE_ACCOUNT_URL', value: captureAccountUrl }
          { name: 'LEARNING_WORKER_CAPTURE_CONTAINER', value: captureContainer }
          { name: 'LEARNING_WORKER_ALLOWED_POLICY_TYPES', value: '[]' }
          { name: 'LEARNING_WORKER_BOOTSTRAP_OWNER_IDS', value: '[]' }
          { name: 'LEARNING_WORKER_RECONCILIATION_ENABLED', value: 'true' }
          { name: 'LEARNING_WORKER_RECONCILIATION_ACTOR_IDS', value: string(reconciliationActorIds) }
          { name: 'LEARNING_WORKER_RECONCILIATION_TARGETS', value: string(reconciliationTargets) }
        ]
      }]
    }
  }
}
