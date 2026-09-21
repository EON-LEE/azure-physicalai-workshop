targetScope = 'resourceGroup'

@minLength(3)
@maxLength(24)
param probeName string
param location string
param vmSize string
param sshPublicKey string
@description('Positive Spot VM compute ceiling in USD/hour; not a total cost cap.')
param maxPrice string
param shutdownTimeUtc string
param probeId string
param adminUsername string = 'probeoperator'

var tags = {
  project: 'azure-physicalai-workshop'
  purpose: 'gpu-capacity-probe'
  probeId: probeId
}
resource nsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${probeName}-nsg'
  location: location
  tags: tags
  properties: {
    securityRules: [{
      name: 'DenyAllInbound'
      properties: {
        priority: 100
        direction: 'Inbound'
        access: 'Deny'
        protocol: '*'
        sourceAddressPrefix: '*'
        sourcePortRange: '*'
        destinationAddressPrefix: '*'
        destinationPortRange: '*'
      }
    }]
  }
}
resource network 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${probeName}-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: { addressPrefixes: ['10.43.0.0/16'] }
    subnets: [{
      name: 'probe'
      properties: {
        addressPrefix: '10.43.0.0/24'
        defaultOutboundAccess: false
        networkSecurityGroup: { id: nsg.id }
      }
    }]
  }
}
resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: '${probeName}-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [{
      name: 'private'
      properties: {
        privateIPAllocationMethod: 'Dynamic'
        subnet: { id: '${network.id}/subnets/probe' }
      }
    }]
  }
}
resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: probeName
  location: location
  tags: tags
  properties: {
    priority: 'Spot'
    evictionPolicy: 'Deallocate'
    billingProfile: { maxPrice: json(maxPrice) }
    hardwareProfile: { vmSize: vmSize }
    diagnosticsProfile: { bootDiagnostics: { enabled: true } }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        name: '${probeName}-os'
        createOption: 'FromImage'
        diskSizeGB: 64
        managedDisk: { storageAccountType: 'StandardSSD_LRS' }
        deleteOption: 'Delete'
      }
    }
    osProfile: {
      computerName: probeName
      adminUsername: adminUsername
      customData: base64(loadTextContent('gpu-probe-cloud-init.yml'))
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [{
            path: '/home/${adminUsername}/.ssh/authorized_keys'
            keyData: sshPublicKey
          }]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [{ id: nic.id, properties: { primary: true, deleteOption: 'Delete' } }]
    }
  }
}
resource shutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${probeName}'
  location: location
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: { time: shutdownTimeUtc }
    timeZoneId: 'UTC'
    targetResourceId: vm.id
    notificationSettings: { status: 'Disabled' }
  }
}

output vmId string = vm.id
output vmName string = vm.name
output allocatedSize string = vmSize
output region string = location
output noPublicIp bool = true
