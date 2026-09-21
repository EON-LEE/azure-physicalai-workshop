targetScope = 'resourceGroup'

param prefix string
param location string
param foundation object
param apiImage string
param entraTenantId string
param entraSpaClientId string
param entraApiClientId string
param publicDemoOwnerId string
param publicDemoEnvironmentId string
param publicDemoRevision string
param presentationId string
@minValue(1)
@maxValue(100)
param cycles int = 20
@minValue(30)
@maxValue(21600)
param maximumSeconds int = 1800

resource job 'Microsoft.App/jobs@2025-07-01' = {
  name: '${prefix}-presentation'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${foundation.apiIdentityId}': {} }
  }
  properties: {
    environmentId: foundation.environmentId
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: maximumSeconds + 120
      replicaRetryLimit: 0
      manualTriggerConfig: { parallelism: 1, replicaCompletionCount: 1 }
      registries: [{ server: foundation.registryServer, identity: foundation.apiIdentityId }]
    }
    template: {
      containers: [{
        name: 'presentation'
        image: apiImage
        command: ['/app/.venv/bin/python', '-u', '-m', 'scripts.run_reference_demo']
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [
          { name: 'DEPLOYMENT_MODE', value: 'azure' }
          { name: 'AZURE_CLIENT_ID', value: foundation.apiClientId }
          { name: 'ENTRA_TENANT_ID', value: entraTenantId }
          { name: 'ENTRA_SPA_CLIENT_ID', value: entraSpaClientId }
          { name: 'ENTRA_API_CLIENT_ID', value: entraApiClientId }
          { name: 'FOUNDRY_PROJECT_ENDPOINT', value: foundation.foundryEndpoint }
          { name: 'FOUNDRY_AGENT_NAME', value: '${prefix}-inspection' }
          { name: 'COSMOS_ENDPOINT', value: foundation.cosmosEndpoint }
          { name: 'STORAGE_ACCOUNT_URL', value: foundation.storageUrl }
          { name: 'SIM_BRIDGE_ENDPOINT', value: 'https://sim.physicalai.internal:8443' }
          { name: 'ALLOW_REFERENCE_PRESENTATION', value: 'true' }
          { name: 'PUBLIC_DEMO_PUBLISH_LIVE', value: 'true' }
          { name: 'PUBLIC_DEMO_OWNER_ID', value: publicDemoOwnerId }
          { name: 'PUBLIC_DEMO_ENVIRONMENT_ID', value: publicDemoEnvironmentId }
          { name: 'PUBLIC_DEMO_REVISION', value: publicDemoRevision }
          { name: 'PRESENTATION_ID', value: presentationId }
          { name: 'PRESENTATION_CYCLES', value: string(cycles) }
          { name: 'PRESENTATION_MAX_SECONDS', value: string(maximumSeconds) }
          { name: 'AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED', value: 'false' }
        ]
      }]
    }
  }
}

output jobName string = job.name
