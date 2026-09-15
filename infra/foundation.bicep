targetScope = 'resourceGroup'

@minLength(3)
@maxLength(10)
param prefix string
param location string
param foundryLocation string
param modelName string
param modelVersion string
param modelDeploymentName string
param modelSkuName string
@minValue(1)
@maxValue(10)
param modelCapacity int
@description('Resolved Foundry User (or supported equivalent) built-in role GUID, not Contributor.')
param foundryUserRoleDefinitionId string
@description('The explicitly authorized deployment operator, for uploading licensed assets.')
param operatorObjectId string

var suffix = uniqueString(resourceGroup().id, prefix)
var compact = replace(prefix, '-', '')
var tags = {
  physicalaiEnvironment: prefix
  project: 'azure-physicalai-workshop'
}

resource apiIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-api-mi'
  location: location
  tags: tags
}
resource simulatorIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-sim-mi'
  location: location
  tags: tags
}
resource bootstrapIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-bootstrap-mi'
  location: location
  tags: tags
}
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${compact}${suffix}'
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: '${compact}${suffix}'
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    defaultToOAuthAuthentication: true
  }
  tags: tags
}
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}
resource artifacts 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'artifacts'
  properties: { publicAccess: 'None' }
}
resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: '${prefix}-${suffix}'
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
    minimalTlsVersion: 'Tls12'
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    capabilities: [{ name: 'EnableServerless' }]
    locations: [{ locationName: location, failoverPriority: 0 }]
  }
  tags: tags
}
resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  parent: cosmos
  name: 'physicalai'
  properties: { resource: { id: 'physicalai' } }
}
resource state 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  parent: database
  name: 'state'
  properties: {
    resource: {
      id: 'state'
      partitionKey: { paths: ['/owner_key'], kind: 'Hash' }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [{ path: '/*' }]
        excludedPaths: [{ path: '/value/environment_document/*' }]
      }
    }
  }
}
resource foundry 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: '${prefix}-ai-${suffix}'
  location: foundryLocation
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    allowProjectManagement: true
    customSubDomainName: '${prefix}-ai-${suffix}'
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}
resource project 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  parent: foundry
  name: 'factory'
  location: foundryLocation
  identity: { type: 'SystemAssigned' }
  properties: {
    displayName: 'Physical AI factory'
    description: 'Dedicated physical AI demo project; no customer production connections.'
  }
}
resource model 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = {
  parent: foundry
  name: modelDeploymentName
  sku: { name: modelSkuName, capacity: modelCapacity }
  properties: {
    model: { format: 'OpenAI', name: modelName, version: modelVersion }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
}
resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${prefix}-kv-${take(suffix, 8)}'
  location: location
  properties: {
    tenantId: tenant().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    enablePurgeProtection: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}
resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    retentionInDays: 30
    sku: { name: 'PerGB2018' }
    workspaceCapping: { dailyQuotaGb: 1 }
  }
  tags: tags
}
resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
  }
  tags: tags
}
resource outboundIp 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: '${prefix}-egress-ip'
  location: location
  sku: { name: 'Standard' }
  properties: { publicIPAllocationMethod: 'Static' }
  tags: tags
}
resource nat 'Microsoft.Network/natGateways@2024-05-01' = {
  name: '${prefix}-nat'
  location: location
  sku: { name: 'Standard' }
  properties: {
    publicIpAddresses: [{ id: outboundIp.id }]
    idleTimeoutInMinutes: 4
  }
  tags: tags
}
resource simNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${prefix}-sim-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'ApiSubnetToAuthenticatedBridge'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: '10.42.0.0/23'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8443'
        }
      }
      {
        name: 'DenyOtherInbound'
        properties: {
          priority: 200
          direction: 'Inbound'
          access: 'Deny'
          protocol: '*'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
  tags: tags
}
resource network 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${prefix}-vnet'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.42.0.0/16'] }
    subnets: [
      {
        name: 'apps'
        properties: {
          addressPrefix: '10.42.0.0/23'
          delegations: [{
            name: 'container-apps'
            properties: { serviceName: 'Microsoft.App/environments' }
          }]
        }
      }
      {
        name: 'simulation'
        properties: {
          addressPrefix: '10.42.4.0/24'
          networkSecurityGroup: { id: simNsg.id }
          natGateway: { id: nat.id }
        }
      }
    ]
  }
  tags: tags
}
resource dns 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'physicalai.internal'
  location: 'global'
  tags: tags
}
resource dnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: dns
  name: '${prefix}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: { id: network.id }
  }
}
resource simDns 'Microsoft.Network/privateDnsZones/A@2020-06-01' = {
  parent: dns
  name: 'sim'
  properties: { ttl: 60, aRecords: [{ ipv4Address: '10.42.4.4' }] }
}
resource environment 'Microsoft.App/managedEnvironments@2025-07-01' = {
  name: '${prefix}-apps'
  location: location
  properties: {
    vnetConfiguration: {
      infrastructureSubnetId: '${network.id}/subnets/apps'
      internal: false
    }
    workloadProfiles: [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
    peerAuthentication: { mtls: { enabled: true } }
  }
  tags: tags
}
var imageConsumers = [apiIdentity.id, simulatorIdentity.id, bootstrapIdentity.id]
var imagePrincipals = [apiIdentity.properties.principalId, simulatorIdentity.properties.principalId, bootstrapIdentity.properties.principalId]
resource acrRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (identityId, index) in imageConsumers: {
  name: guid(registry.id, identityId, 'AcrPull')
  scope: registry
  properties: {
    principalId: imagePrincipals[index]
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}]
var artifactWriters = [apiIdentity.id, bootstrapIdentity.id]
var writerPrincipals = [apiIdentity.properties.principalId, bootstrapIdentity.properties.principalId]
resource artifactRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (identityId, index) in artifactWriters: {
  name: guid(artifacts.id, identityId, 'BlobContributor')
  scope: artifacts
  properties: {
    principalId: writerPrincipals[index]
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}]
resource assetReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(artifacts.id, simulatorIdentity.id, 'BlobReader')
  scope: artifacts
  properties: {
    principalId: simulatorIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
  }
}
resource assetUploader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(artifacts.id, operatorObjectId, 'AssetUploader')
  scope: artifacts
  properties: {
    principalId: operatorObjectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  }
}
resource cosmosRole 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2024-05-15' = {
  parent: cosmos
  name: guid(cosmos.id, apiIdentity.id, 'SqlDataContributor')
  properties: {
    principalId: apiIdentity.properties.principalId
    roleDefinitionId: '${cosmos.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    scope: cosmos.id
  }
}
resource foundryRoles 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for (identityId, index) in artifactWriters: {
  name: guid(foundry.id, identityId, 'FoundryUser')
  scope: foundry
  properties: {
    principalId: writerPrincipals[index]
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', foundryUserRoleDefinitionId)
  }
}]
resource bootstrapVaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, bootstrapIdentity.id, 'SecretsOfficer')
  scope: vault
  properties: {
    principalId: bootstrapIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
  }
}
resource simulatorVaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, simulatorIdentity.id, 'SecretsUser')
  scope: vault
  properties: {
    principalId: simulatorIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
  }
}

output registryName string = registry.name
output registryServer string = registry.properties.loginServer
output storageName string = storage.name
output storageUrl string = storage.properties.primaryEndpoints.blob
output cosmosEndpoint string = cosmos.properties.documentEndpoint
output foundryEndpoint string = 'https://${foundry.name}.services.ai.azure.com/api/projects/factory'
output modelDeployment string = modelDeploymentName
output vaultUrl string = vault.properties.vaultUri
output environmentId string = environment.id
output simulationSubnetId string = '${network.id}/subnets/simulation'
output apiIdentityId string = apiIdentity.id
output apiClientId string = apiIdentity.properties.clientId
output apiPrincipalId string = apiIdentity.properties.principalId
output simulatorIdentityId string = simulatorIdentity.id
output simulatorClientId string = simulatorIdentity.properties.clientId
output bootstrapIdentityId string = bootstrapIdentity.id
output bootstrapClientId string = bootstrapIdentity.properties.clientId
output insightsConnectionString string = insights.properties.ConnectionString
