targetScope = 'resourceGroup'

param prefix string
param location string
param foundation object
param apiImage string
param entraTenantId string
param entraSpaClientId string
param entraApiClientId string
param deploymentRevision string
param appName string = '${prefix}-web'
param publicDemoPublishLive bool = false
param publicDemoOwnerId string = ''
param publicDemoEnvironmentId string = ''
param publicDemoRevision string = ''
param publicDemoDefectEnvironmentId string = ''
param publicDemoDefectRevision string = ''
param publicDemoPresentationId string = ''

resource web 'Microsoft.App/containerApps@2025-07-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${foundation.apiIdentityId}': {} }
  }
  properties: {
    managedEnvironmentId: foundation.environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, allowInsecure: false, targetPort: 8000, transport: 'http' }
      registries: [{ server: foundation.registryServer, identity: foundation.apiIdentityId }]
    }
    template: {
      containers: [{
        name: 'web'
        image: apiImage
        resources: { cpu: json('1.0'), memory: '2Gi' }
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
          { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: foundation.insightsConnectionString }
          { name: 'AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED', value: 'false' }
          { name: 'DEPLOYMENT_REVISION', value: deploymentRevision }
          { name: 'PUBLIC_DEMO_PUBLISH_LIVE', value: string(publicDemoPublishLive) }
          ...(publicDemoPublishLive ? [
            { name: 'PUBLIC_DEMO_OWNER_ID', value: publicDemoOwnerId }
            { name: 'PUBLIC_DEMO_ENVIRONMENT_ID', value: publicDemoEnvironmentId }
            { name: 'PUBLIC_DEMO_REVISION', value: publicDemoRevision }
          ] : [])
          ...(publicDemoPublishLive && !empty(publicDemoPresentationId) ? [
            { name: 'PUBLIC_DEMO_DEFECT_ENVIRONMENT_ID', value: publicDemoDefectEnvironmentId }
            { name: 'PUBLIC_DEMO_DEFECT_REVISION', value: publicDemoDefectRevision }
            { name: 'PUBLIC_DEMO_PRESENTATION_ID', value: publicDemoPresentationId }
          ] : [])
        ]
        probes: [
          {
            type: 'Liveness'
            httpGet: { path: '/healthz', port: 8000 }
            initialDelaySeconds: 15
            periodSeconds: 30
          }
          {
            type: 'Startup'
            httpGet: { path: '/healthz', port: 8000 }
            periodSeconds: 10
            failureThreshold: 30
          }
        ]
      }]
      scale: { minReplicas: publicDemoPublishLive ? 1 : 0, maxReplicas: 2 }
    }
  }
}

output webUrl string = 'https://${web.properties.configuration.ingress.fqdn}'
