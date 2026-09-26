targetScope = 'resourceGroup'

param enabled bool = false
param batchAccountName string

@description('Existing private worker managed-identity principal; no identity or credential is created.')
@minLength(36)
@maxLength(36)
param submitterPrincipalId string

resource account 'Microsoft.Batch/batchAccounts@2024-07-01' existing = {
  name: batchAccountName
}

resource submitter 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enabled) {
  name: guid(account.id, submitterPrincipalId, '48e5e92e-a480-4e71-aa9c-2778f4c13781')
  scope: account
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '48e5e92e-a480-4e71-aa9c-2778f4c13781')
    principalId: submitterPrincipalId
    principalType: 'ServicePrincipal'
  }
}
