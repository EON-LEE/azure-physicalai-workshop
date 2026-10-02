targetScope = 'resourceGroup'

@description('Explicit alternative bootstrap runner when Container Apps Jobs cannot start replicas.')
param prefix string
param location string
param foundation object
param apiImage string
param runnerName string = '${prefix}-bootstrap-private'
param simulatorPrivateIp string = '10.42.4.4'

resource bootstrap 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: runnerName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${foundation.bootstrapIdentityId}': {} }
  }
  properties: {
    osType: 'Linux'
    restartPolicy: 'Never'
    subnetIds: [{ id: foundation.bootstrapSubnetId }]
    dnsConfig: { nameServers: ['168.63.129.16'] }
    imageRegistryCredentials: [{
      server: foundation.registryServer
      identity: foundation.bootstrapIdentityId
    }]
    containers: [{
      name: 'bootstrap'
      properties: {
        image: apiImage
        command: ['/app/.venv/bin/python', '-m', 'scripts.bootstrap']
        resources: { requests: { cpu: 1, memoryInGB: json('1.5') } }
        environmentVariables: [
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
      }
    }]
  }
}

output containerGroupName string = bootstrap.name
