targetScope = 'resourceGroup'

param gpuName string
param gpuLocation string = 'westus2'
param appVnetName string = 'factory20-vnet'
param gpuPrivateIp string = '10.43.0.4'
param shutdownTimeUtc string

var tags = {
  project: 'azure-physicalai-workshop'
  purpose: 'physicalai-live-runtime'
}
resource ip 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: '${gpuName}-egress'
  location: gpuLocation
  sku: { name: 'Standard' }
  properties: { publicIPAllocationMethod: 'Static', publicIPAddressVersion: 'IPv4' }
  tags: tags
}
resource nat 'Microsoft.Network/natGateways@2024-05-01' = {
  name: '${gpuName}-nat'
  location: gpuLocation
  sku: { name: 'Standard' }
  properties: { publicIpAddresses: [{ id: ip.id }], idleTimeoutInMinutes: 4 }
  tags: tags
}
resource nsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${gpuName}-nsg'
  location: gpuLocation
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'AzureApiToAuthenticatedBridge'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefixes: ['10.42.0.0/22', '10.42.8.0/24']
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8443'
        }
      }
      {
        name: 'DenyAllInbound'
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
}
resource gpuVnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${gpuName}-vnet'
  location: gpuLocation
  tags: tags
  properties: {
    addressSpace: { addressPrefixes: ['10.43.0.0/16'] }
    subnets: [{
      name: 'probe'
      properties: {
        addressPrefix: '10.43.0.0/24'
        defaultOutboundAccess: false
        networkSecurityGroup: { id: nsg.id }
        natGateway: { id: nat.id }
      }
    }]
  }
}
resource appVnet 'Microsoft.Network/virtualNetworks@2024-05-01' existing = {
  name: appVnetName
}
resource toGpu 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = {
  parent: appVnet
  name: 'physicalai-to-gpu'
  properties: {
    remoteVirtualNetwork: { id: gpuVnet.id }
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
  }
}
resource toApp 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = {
  parent: gpuVnet
  name: 'physicalai-to-app'
  properties: {
    remoteVirtualNetwork: { id: appVnet.id }
    allowVirtualNetworkAccess: true
    allowForwardedTraffic: false
    allowGatewayTransit: false
    useRemoteGateways: false
  }
}
var zoneNames = [
  'physicalai.internal'
  'privatelink.blob.${environment().suffixes.storage}'
  'privatelink.documents.azure.com'
  'privatelink.vaultcore.azure.net'
]
resource zones 'Microsoft.Network/privateDnsZones@2020-06-01' existing = [for zoneName in zoneNames: {
  name: zoneName
}]
resource links 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [for (zoneName, index) in zoneNames: {
  parent: zones[index]
  name: '${gpuName}-link'
  location: 'global'
  properties: { registrationEnabled: false, virtualNetwork: { id: gpuVnet.id } }
}]
resource internalZone 'Microsoft.Network/privateDnsZones@2020-06-01' existing = {
  name: 'physicalai.internal'
}
resource simDns 'Microsoft.Network/privateDnsZones/A@2020-06-01' = {
  parent: internalZone
  name: 'sim'
  properties: { ttl: 60, aRecords: [{ ipv4Address: gpuPrivateIp }] }
}
resource shutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${gpuName}'
  location: gpuLocation
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: { time: shutdownTimeUtc }
    timeZoneId: 'UTC'
    targetResourceId: resourceId('Microsoft.Compute/virtualMachines', gpuName)
    notificationSettings: { status: 'Disabled' }
  }
}

output bridgeEndpoint string = 'https://sim.physicalai.internal:8443'
output gpuSubnetId string = '${gpuVnet.id}/subnets/probe'
