targetScope = 'resourceGroup'

@description('Private worker app name. This template creates no GPU compute, role grants or identities.')
param appName string
param location string = resourceGroup().location

@description('Existing approved ACA managed environment with the private AML/storage network path.')
param managedEnvironmentId string

@description('Existing dedicated worker UAMI resource ID. Must not be the API/simulator-control identity.')
param workerIdentityResourceId string

@description('Only the API managed-identity object/principal ID may call protected worker routes.')
@minLength(36)
@maxLength(36)
param apiPrincipalId string

@minLength(36)
@maxLength(36)
param entraTenantId string

@description('Entra audience application/client UUID accepted by the worker, not a secret.')
@minLength(36)
@maxLength(36)
param workerAudience string

param registryServer string
param workerImageRepository string

@description('Immutable SHA256 of the already-built private worker image; never a tag.')
@minLength(64)
@maxLength(64)
param workerImageSha256 string

param registryAccountUrl string
param registryContainer string
param captureAccountUrl string
param captureContainer string

@description('Explicitly keep one warm replica only after operational approval. This is not a model-quality flag.')
param enabled bool = false

@description('Empty until model artifact, license and hardware admission is verified. No GR00T fallback.')
@allowed(['smolvla'])
param allowedPolicyTypes array = []

@description('Empty until the deployment operator explicitly approves bootstrap owner UUIDs.')
param bootstrapOwnerIds array = []

resource workerIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: last(split(workerIdentityResourceId, '/'))
  scope: resourceGroup(split(workerIdentityResourceId, '/')[2], split(workerIdentityResourceId, '/')[4])
}

resource worker 'Microsoft.App/containerApps@2025-07-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workerIdentityResourceId}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        allowInsecure: false
        targetPort: 8080
        transport: 'http'
      }
      registries: [{
        server: registryServer
        identity: workerIdentityResourceId
      }]
    }
    template: {
      containers: [{
        name: 'learning-worker'
        image: '${registryServer}/${workerImageRepository}@sha256:${workerImageSha256}'
        resources: { cpu: json('1.0'), memory: '2Gi' }
        env: [
          { name: 'LEARNING_WORKER_TENANT_ID', value: entraTenantId }
          { name: 'LEARNING_WORKER_AUDIENCE', value: workerAudience }
          { name: 'LEARNING_WORKER_MANAGED_IDENTITY_CLIENT_ID', value: workerIdentity.properties.clientId }
          { name: 'LEARNING_WORKER_ALLOWED_API_PRINCIPALS', value: string([apiPrincipalId]) }
          { name: 'LEARNING_WORKER_REGISTRY_ACCOUNT_URL', value: registryAccountUrl }
          { name: 'LEARNING_WORKER_REGISTRY_CONTAINER', value: registryContainer }
          { name: 'LEARNING_WORKER_CAPTURE_ACCOUNT_URL', value: captureAccountUrl }
          { name: 'LEARNING_WORKER_CAPTURE_CONTAINER', value: captureContainer }
          { name: 'LEARNING_WORKER_ALLOWED_POLICY_TYPES', value: string(allowedPolicyTypes) }
          { name: 'LEARNING_WORKER_BOOTSTRAP_OWNER_IDS', value: string(bootstrapOwnerIds) }
        ]
        probes: [
          {
            type: 'Startup'
            httpGet: { path: '/healthz', port: 8080 }
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30
          }
          {
            type: 'Liveness'
            httpGet: { path: '/healthz', port: 8080 }
            initialDelaySeconds: 20
            periodSeconds: 30
            timeoutSeconds: 5
          }
          {
            type: 'Readiness'
            httpGet: { path: '/healthz', port: 8080 }
            periodSeconds: 10
            timeoutSeconds: 5
          }
        ]
      }]
      scale: {
        minReplicas: enabled ? 1 : 0
        maxReplicas: 1
        rules: [{
          name: 'bounded-private-http'
          http: { metadata: { concurrentRequests: '2' } }
        }]
      }
    }
  }
}

output workerEndpoint string = 'https://${worker.properties.configuration.ingress.fqdn}'
output workerIdentityClientId string = workerIdentity.properties.clientId
output workerIdentityPrincipalId string = workerIdentity.properties.principalId
