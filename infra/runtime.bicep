targetScope = 'resourceGroup'

param prefix string
param location string
param foundation object
param apiImage string
param simulatorImage string
param entraTenantId string
param entraSpaClientId string
param entraApiClientId string
@description('Select and benchmark an Isaac-compatible RT-core SKU; there is no implicit GPU default.')
param gpuVmSize string
param sshPublicKey string
param adminUsername string = 'azureuser'
param assetSha256 string
param frankaUsdRelativePath string = 'Franka/franka.usd'
param acceptNvidiaEula bool
@description('New value per controlled rollout to refresh GPU TLS and API bootstrap state.')
param deploymentRevision string
param sourceRevision string
param enableDemonstrationCapture bool = false
@description('Daily UTC deallocation time, HHmm. This is not a hard spend cap.')
param shutdownTimeUtc string = '2300'
@minValue(128)
param gpuDiskSizeGB int = 256

var runtimeConfiguration = {
  REGISTRY_NAME: foundation.registryName
  SIMULATOR_IMAGE: simulatorImage
  AZURE_CLIENT_ID: foundation.simulatorClientId
  ENTRA_TENANT_ID: entraTenantId
  ENTRA_API_CLIENT_ID: entraApiClientId
  BRIDGE_ALLOWED_PRINCIPAL_IDS: string([foundation.apiPrincipalId])
  KEY_VAULT_URL: foundation.vaultUrl
  STORAGE_ACCOUNT_URL: foundation.storageUrl
  STORAGE_CONTAINER: 'artifacts'
  FRANKA_ASSET_BLOB: 'assets/franka.tar'
  FRANKA_ASSET_SHA256: assetSha256
  FRANKA_USD_RELATIVE_PATH: frankaUsdRelativePath
  ACCEPT_EULA: acceptNvidiaEula ? 'Y' : 'N'
  PRIVACY_CONSENT: 'N'
  NVIDIA_DRIVER_CAPABILITIES: 'all'
  SOURCE_REVISION: sourceRevision
  CAPTURE_ENABLED: enableDemonstrationCapture ? 'true' : 'false'
  DEMONSTRATION_CONTAINER: 'demonstrations'
}
var startScript = replace(
  loadTextContent('start-simulator.sh'),
  '__CONFIGURATION_BASE64__',
  base64(string(runtimeConfiguration))
)

resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: '${prefix}-sim-nic'
  location: location
  properties: {
    ipConfigurations: [{
      name: 'private'
      properties: {
        privateIPAllocationMethod: 'Static'
        privateIPAddress: '10.42.4.4'
        subnet: { id: foundation.simulationSubnetId }
      }
    }]
  }
}
resource simulator 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-sim'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${foundation.simulatorIdentityId}': {} }
  }
  properties: {
    hardwareProfile: { vmSize: gpuVmSize }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: gpuDiskSizeGB
        managedDisk: { storageAccountType: 'Premium_LRS' }
        deleteOption: 'Delete'
      }
    }
    osProfile: {
      computerName: '${prefix}-sim'
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        provisionVMAgent: true
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
resource driver 'Microsoft.Compute/virtualMachines/extensions@2024-07-01' = {
  parent: simulator
  name: 'nvidia-driver'
  location: location
  properties: {
    publisher: 'Microsoft.HpcCompute'
    type: 'NvidiaGpuDriverLinux'
    typeHandlerVersion: '1.10'
    autoUpgradeMinorVersion: true
  }
}
resource start 'Microsoft.Compute/virtualMachines/extensions@2024-07-01' = {
  parent: simulator
  name: 'start-simulator'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    forceUpdateTag: deploymentRevision
    settings: { script: base64(startScript) }
  }
  dependsOn: [driver]
}
resource shutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${simulator.name}'
  location: location
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: { time: shutdownTimeUtc }
    timeZoneId: 'UTC'
    targetResourceId: simulator.id
    notificationSettings: { status: 'Disabled' }
  }
}
module web 'web.bicep' = {
  name: '${prefix}-web-deployment'
  params: {
    prefix: prefix
    location: location
    foundation: foundation
    apiImage: apiImage
    entraTenantId: entraTenantId
    entraSpaClientId: entraSpaClientId
    entraApiClientId: entraApiClientId
    deploymentRevision: deploymentRevision
  }
}

output webUrl string = web.outputs.webUrl
output simulatorId string = simulator.id
output controllerPrincipalId string = foundation.apiPrincipalId
