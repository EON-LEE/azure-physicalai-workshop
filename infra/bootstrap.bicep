targetScope = 'resourceGroup'

param prefix string
param location string
param foundation object
@description('Immutable ACR API image reference, including @sha256 digest.')
param apiImage string
param simulatorPrivateIp string = '10.42.4.4'

resource job 'Microsoft.App/jobs@2025-07-01' = {
  name: '${prefix}-bootstrap'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${foundation.bootstrapIdentityId}': {} }
  }
  properties: {
    environmentId: foundation.environmentId
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 600
      replicaRetryLimit: 0
      manualTriggerConfig: { parallelism: 1, replicaCompletionCount: 1 }
      registries: [{
        server: foundation.registryServer
        identity: foundation.bootstrapIdentityId
      }]
    }
    template: {
      containers: [{
        name: 'bootstrap'
        image: apiImage
        command: ['/app/.venv/bin/python', '-m', 'scripts.bootstrap']
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [
          { name: 'AZURE_CLIENT_ID', value: foundation.bootstrapClientId }
          { name: 'ALLOW_BOOTSTRAP_WRITE', value: 'true' }
          { name: 'FOUNDRY_PROJECT_ENDPOINT', value: foundation.foundryEndpoint }
          { name: 'FOUNDRY_AGENT_NAME', value: '${prefix}-inspection' }
          { name: 'FOUNDRY_MODEL_DEPLOYMENT', value: foundation.modelDeployment }
          { name: 'STORAGE_ACCOUNT_URL', value: foundation.storageUrl }
          { name: 'STORAGE_CONTAINER', value: 'artifacts' }
          { name: 'KEY_VAULT_URL', value: foundation.vaultUrl }
          { name: 'SIM_HOSTNAME', value: 'sim.physicalai.internal' }
          { name: 'SIM_PRIVATE_IP', value: simulatorPrivateIp }
        ]
      }]
    }
  }
}

output jobName string = job.name
