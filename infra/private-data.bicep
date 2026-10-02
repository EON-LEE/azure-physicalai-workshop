targetScope = 'resourceGroup'

param prefix string
param location string
param virtualNetworkId string
param endpointSubnetId string
param storageAccountId string
param cosmosAccountId string
param vaultId string

var services = [
  { key: 'blob', zone: 'privatelink.blob.${environment().suffixes.storage}', target: storageAccountId, groupId: 'blob' }
  { key: 'cosmos', zone: 'privatelink.documents.azure.com', target: cosmosAccountId, groupId: 'Sql' }
  { key: 'vault', zone: 'privatelink.vaultcore.azure.net', target: vaultId, groupId: 'vault' }
]
resource zones 'Microsoft.Network/privateDnsZones@2020-06-01' = [for service in services: {
  name: service.zone
  location: 'global'
}]
resource links 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [for (service, index) in services: {
  parent: zones[index]
  name: '${prefix}-link'
  location: 'global'
  properties: { registrationEnabled: false, virtualNetwork: { id: virtualNetworkId } }
}]
resource endpoints 'Microsoft.Network/privateEndpoints@2024-05-01' = [for service in services: {
  name: '${prefix}-${service.key}-pe'
  location: location
  properties: {
    subnet: { id: endpointSubnetId }
    privateLinkServiceConnections: [{
      name: '${prefix}-${service.key}'
      properties: { privateLinkServiceId: service.target, groupIds: [service.groupId] }
    }]
  }
}]
resource zoneGroups 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2024-05-01' = [for (service, index) in services: {
  parent: endpoints[index]
  name: 'default'
  properties: {
    privateDnsZoneConfigs: [{
      name: service.key
      properties: { privateDnsZoneId: zones[index].id }
    }]
  }
}]
