targetScope = 'resourceGroup'

param enabled bool = false
param nodeIdentityName string
param location string
param registryName string
param storageAccountName string

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = if (enabled) {
  name: nodeIdentityName
  location: location
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: registryName
}
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}
resource blobs 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storage
  name: 'default'
}
resource inputs 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' existing = {
  parent: blobs
  name: 'artifacts'
}
resource outputs 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' existing = {
  parent: blobs
  name: 'demonstrations'
}

resource pull 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enabled) {
  name: guid(registry.id, identity.id, 'AcrPull')
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: enabled ? identity!.properties.principalId : ''
    principalType: 'ServicePrincipal'
  }
}

resource readInputs 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enabled) {
  name: guid(inputs.id, identity.id, 'BlobReader')
  scope: inputs
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
    principalId: enabled ? identity!.properties.principalId : ''
    principalType: 'ServicePrincipal'
  }
}

resource writeOutputs 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enabled) {
  name: guid(outputs.id, identity.id, 'BlobContributor')
  scope: outputs
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: enabled ? identity!.properties.principalId : ''
    principalType: 'ServicePrincipal'
  }
}

output nodeIdentityResourceId string = enabled ? identity.id : ''
output nodeIdentityClientId string = enabled ? identity!.properties.clientId : ''
output nodeIdentityPrincipalId string = enabled ? identity!.properties.principalId : ''
