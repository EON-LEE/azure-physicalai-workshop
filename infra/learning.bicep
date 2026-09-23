targetScope = 'resourceGroup'

@minLength(3)
@maxLength(33)
param workspaceName string
param location string
param storageAccountId string
param keyVaultId string
param containerRegistryId string
param workspaceIdentityId string
param computeIdentityId string
param privateEndpointSubnetId string
@minLength(2)
@maxLength(2)
@description('Existing privatelink.api.azureml.ms and privatelink.notebooks.azure.net zone IDs, already linked to the authorized client VNet.')
param privateDnsZoneIds array
param computeName string
param computeSize string
@allowed([
  'Dedicated'
  'LowPriority'
])
param computeTier string
param tags object = {}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${workspaceIdentityId}': {}
    }
  }
  properties: {
    friendlyName: workspaceName
    description: 'Private, explicitly approved Physical AI learning; no automatic training submission.'
    storageAccount: storageAccountId
    keyVault: keyVaultId
    containerRegistry: containerRegistryId
    primaryUserAssignedIdentity: workspaceIdentityId
    publicNetworkAccess: 'Disabled'
    allowPublicAccessWhenBehindVnet: false
    hbiWorkspace: true
    v1LegacyMode: false
    managedNetwork: {
      isolationMode: 'AllowOnlyApprovedOutbound'
      outboundRules: {
        approvedBlob: {
          type: 'PrivateEndpoint'
          category: 'UserDefined'
          destination: {
            serviceResourceId: storageAccountId
            subresourceTarget: 'blob'
          }
        }
        approvedFile: {
          type: 'PrivateEndpoint'
          category: 'UserDefined'
          destination: {
            serviceResourceId: storageAccountId
            subresourceTarget: 'file'
          }
        }
        approvedKeyVault: {
          type: 'PrivateEndpoint'
          category: 'UserDefined'
          destination: {
            serviceResourceId: keyVaultId
            subresourceTarget: 'vault'
          }
        }
        approvedRegistry: {
          type: 'PrivateEndpoint'
          category: 'UserDefined'
          destination: {
            serviceResourceId: containerRegistryId
            subresourceTarget: 'registry'
          }
        }
      }
    }
  }
}

resource endpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = {
  name: '${workspaceName}-private'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: '${workspaceName}-workspace'
        properties: {
          privateLinkServiceId: workspace.id
          groupIds: [
            'amlworkspace'
          ]
        }
      }
    ]
  }
}

resource endpointDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = {
  name: 'approved-zones'
  parent: endpoint
  properties: {
    privateDnsZoneConfigs: [for (zoneId, index) in privateDnsZoneIds: {
      name: 'zone-${index}'
      properties: {
        privateDnsZoneId: zoneId
      }
    }]
  }
}

resource compute 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  name: computeName
  parent: workspace
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${computeIdentityId}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    computeLocation: location
    disableLocalAuth: true
    properties: {
      vmSize: computeSize
      vmPriority: computeTier
      osType: 'Linux'
      enableNodePublicIp: false
      remoteLoginPortPublicAccess: 'Disabled'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
    }
  }
}

output workspaceId string = workspace.id
output computeId string = compute.id
output workspacePrivateEndpointId string = endpoint.id
output approvedComputeTier string = computeTier
output approvedComputeSize string = computeSize
