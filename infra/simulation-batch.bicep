targetScope = 'resourceGroup'

@description('Pool creation is separate from the private Batch foundation and explicit job submission.')
param provisionPool bool = false
param batchAccountName string
@minLength(1)
@maxLength(64)
param poolName string
@description('Existing private node subnet with nodeManagement/Blob/ACR connectivity and approved driver-download egress.')
param nodeSubnetId string
@description('Existing dedicated node UAMI. This module never creates identities or assigns roles.')
param nodeIdentityId string
@description('Digest-pinned reviewed simulator image containing simulation.batch_task; never a mutable tag.')
param simulatorImage string
param registryServer string
@description('Explicit host candidate; the Ubuntu 24.04 Batch bootstrap failed on the tested A10 node.')
@allowed(['2204', '2404'])
param hostImageSku string
@allowed(['extension', 'bootstrap'])
param driverInstallation string = 'extension'
@description('Exact command from simulation.batch.driver_bootstrap_command; validated before a warmup job is submitted.')
param driverBootstrapCommand string = ''
@description('Immutable deployment-window start, in UTC; redeploying a new window requires explicit approval.')
param allocationStartUtc string
@minValue(1)
@maxValue(60)
param allocationMinutes int = 60
param tags object = {}

var allocationDeadlineUtc = dateTimeAdd(allocationStartUtc, 'PT${allocationMinutes}M')
var hostImages = {
  '2204': {
    version: '22.04.2026082801'
    nodeAgentSkuId: 'batch.node.ubuntu 22.04'
  }
  '2404': {
    version: '24.04.2026092501'
    nodeAgentSkuId: 'batch.node.ubuntu 24.04'
  }
}
var hostImage = hostImages[hostImageSku]
var autoscale = format('''
$TargetDedicatedNodes = 0;
$samples = $PendingTasks.GetSamplePercent(TimeInterval_Minute * 5);
$tasks = $samples < 70 ? max(0, $PendingTasks.GetSample(1)) : max(0, max($PendingTasks.GetSample(TimeInterval_Minute * 5)));
$TargetLowPriorityNodes = time() < time("{0}") ? min(1, $tasks) : 0;
$NodeDeallocationOption = terminate;
''', allocationDeadlineUtc)
var containerOptions = '--entrypoint /usr/bin/timeout --cap-drop ALL --security-opt no-new-privileges --shm-size 2g --tmpfs /data:rw,nosuid,nodev,mode=1777,size=2147483648 --tmpfs /isaac-sim/.cache:rw,nosuid,nodev,mode=1777,size=2147483648 --tmpfs /isaac-sim/.nv/ComputeCache:rw,nosuid,nodev,mode=1777,size=536870912 --tmpfs /isaac-sim/.nvidia-omniverse/logs:rw,nosuid,nodev,mode=1777,size=134217728'
var gridExtensions = [{
  name: 'nvidia-grid'
  publisher: 'Microsoft.HpcCompute'
  type: 'NvidiaGpuDriverLinux'
  typeHandlerVersion: '1.14'
  autoUpgradeMinorVersion: false
  enableAutomaticUpgrade: false
  settings: {
    driverVersion: '570.237'
    installCUDA: false
    updateOS: false
  }
}]
var extensionStartTask = {
  commandLine: '--signal=TERM --kill-after=5s 60s /isaac-sim/python.sh -m simulation.batch_task preflight --output preflight.json'
  containerSettings: {
    imageName: simulatorImage
    containerRunOptions: containerOptions
    workingDirectory: 'TaskWorkingDirectory'
  }
  environmentSettings: [
    { name: 'PYTHONPATH', value: '/app' }
    { name: 'NVIDIA_DRIVER_CAPABILITIES', value: 'all' }
  ]
  userIdentity: {
    autoUser: {
      scope: 'Task'
      elevationLevel: 'NonAdmin'
    }
  }
  maxTaskRetryCount: 0
  waitForSuccess: true
}
var bootstrapStartTask = {
  commandLine: driverBootstrapCommand
  userIdentity: {
    autoUser: {
      scope: 'Pool'
      elevationLevel: 'Admin'
    }
  }
  maxTaskRetryCount: 0
  waitForSuccess: true
}

resource account 'Microsoft.Batch/batchAccounts@2025-06-01' existing = {
  name: batchAccountName
}

resource pool 'Microsoft.Batch/batchAccounts/pools@2025-06-01' = if (provisionPool) {
  name: poolName
  parent: account
  tags: union(tags, {
    workload: 'physicalai-managed-simulator'
    executionMode: 'NON_REALTIME_SIMULATION'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${nodeIdentityId}': {}
    }
  }
  properties: {
    displayName: 'Single approved RTX simulator task; no automatic episode replay'
    vmSize: 'Standard_NV36ads_A10_v5'
    interNodeCommunication: 'Disabled'
    taskSlotsPerNode: 1
    taskSchedulingPolicy: {
      nodeFillType: 'Pack'
    }
    deploymentConfiguration: {
      virtualMachineConfiguration: {
        imageReference: {
          publisher: 'microsoft-dsvm'
          offer: 'ubuntu-hpc'
          sku: hostImageSku
          version: hostImage.version
        }
        nodeAgentSkuId: hostImage.nodeAgentSkuId
        osDisk: {
          diskSizeGB: 128
        }
        extensions: driverInstallation == 'extension' ? gridExtensions : []
        containerConfiguration: {
          type: 'DockerCompatible'
          containerImageNames: [
            simulatorImage
          ]
          containerRegistries: [
            {
              registryServer: registryServer
              identityReference: {
                resourceId: nodeIdentityId
              }
            }
          ]
        }
      }
    }
    networkConfiguration: {
      subnetId: nodeSubnetId
      publicIPAddressConfiguration: {
        provision: 'NoPublicIPAddresses'
      }
    }
    scaleSettings: {
      autoScale: {
        evaluationInterval: 'PT5M'
        formula: autoscale
      }
    }
    startTask: driverInstallation == 'bootstrap' ? bootstrapStartTask : extensionStartTask
    metadata: [
      {
        name: 'physicalaiDriver'
        value: 'GRID-570.237'
      }
      {
        name: 'physicalaiAllocationDeadline'
        value: allocationDeadlineUtc
      }
    ]
  }
}

output poolId string = provisionPool ? pool.id : ''
output approvedAllocationDeadlineUtc string = allocationDeadlineUtc
output maxLowPriorityNodes int = 1
output maxConcurrentPhysicsInstances int = 1
