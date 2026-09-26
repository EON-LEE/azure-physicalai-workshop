targetScope = 'resourceGroup'

@description('Create only the private managed-job foundation. No pool, node or task is started.')
param enabled bool = false

@minLength(3)
@maxLength(24)
param batchAccountName string
param location string
param virtualNetworkId string
param privateEndpointSubnetId string
param natGatewayId string
param nodeSubnetName string
param nodeSubnetPrefix string
param tags object = {}

resource network 'Microsoft.Network/virtualNetworks@2024-05-01' existing = {
  name: last(split(virtualNetworkId, '/'))
}

resource nodeSecurity 'Microsoft.Network/networkSecurityGroups@2024-05-01' = if (enabled) {
  name: '${batchAccountName}-nodes'
  location: location
  tags: tags
  properties: {
    securityRules: [{
      name: 'NoInboundNodeAccess'
      properties: {
        access: 'Deny'
        direction: 'Inbound'
        priority: 200
        protocol: '*'
        sourceAddressPrefix: '*'
        sourcePortRange: '*'
        destinationAddressPrefix: '*'
        destinationPortRange: '*'
      }
    }]
  }
}

resource nodeSubnet 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' = if (enabled) {
  name: nodeSubnetName
  parent: network
  properties: {
    addressPrefix: nodeSubnetPrefix
    defaultOutboundAccess: false
    networkSecurityGroup: {
      id: nodeSecurity.id
    }
    natGateway: {
      id: natGatewayId
    }
  }
}

resource account 'Microsoft.Batch/batchAccounts@2024-07-01' = if (enabled) {
  name: batchAccountName
  location: location
  tags: tags
  properties: {
    allowedAuthenticationModes: ['AAD']
    poolAllocationMode: 'BatchService'
    publicNetworkAccess: 'Disabled'
    encryption: {
      keySource: 'Microsoft.Batch'
    }
  }
}

resource dns 'Microsoft.Network/privateDnsZones@2020-06-01' = if (enabled) {
  name: 'privatelink.batch.azure.com'
  location: 'global'
  tags: tags
}

resource dnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = if (enabled) {
  name: '${batchAccountName}-clients'
  parent: dns
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetworkId
    }
  }
}

resource accountEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (enabled) {
  name: '${batchAccountName}-api'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointSubnetId
    }
    privateLinkServiceConnections: [{
      name: 'batch-api'
      properties: {
        privateLinkServiceId: account.id
        groupIds: ['batchAccount']
      }
    }]
  }
}

resource nodeEndpoint 'Microsoft.Network/privateEndpoints@2024-05-01' = if (enabled) {
  name: '${batchAccountName}-management'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointSubnetId
    }
    privateLinkServiceConnections: [{
      name: 'batch-node-management'
      properties: {
        privateLinkServiceId: account.id
        groupIds: ['nodeManagement']
      }
    }]
  }
}

resource accountDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (enabled) {
  name: 'batch'
  parent: accountEndpoint
  properties: {
    privateDnsZoneConfigs: [{
      name: 'batch'
      properties: {
        privateDnsZoneId: dns.id
      }
    }]
  }
}

resource nodeDns 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = if (enabled) {
  name: 'batch'
  parent: nodeEndpoint
  properties: {
    privateDnsZoneConfigs: [{
      name: 'batch'
      properties: {
        privateDnsZoneId: dns.id
      }
    }]
  }
}

output batchAccountId string = enabled ? account.id : ''
output batchAccountEndpoint string = enabled ? account.properties.accountEndpoint : ''
output batchNodeSubnetId string = enabled ? nodeSubnet.id : ''
