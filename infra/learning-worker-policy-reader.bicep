targetScope = 'resourceGroup'

param enabled bool = false
param storageAccountName string
@minLength(36)
@maxLength(36)
param workerPrincipalId string

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource retention 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' existing = {
  name: 'default'
  parent: storage
}

resource reader 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enabled) {
  name: guid(retention.id, workerPrincipalId, 'retention-policy-reader')
  scope: retention
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'acdd72a7-3385-48ef-bd42-f606fba81ae7')
    principalId: workerPrincipalId
    principalType: 'ServicePrincipal'
  }
}
