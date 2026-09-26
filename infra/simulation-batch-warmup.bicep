targetScope = 'resourceGroup'

param enabled bool = false
param jobName string
param location string
param managedEnvironmentId string
param workerIdentityResourceId string
param registryServer string
@minLength(64)
@maxLength(64)
param workerImageSha256 string
param platform object
param warmupId string

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: last(split(workerIdentityResourceId, '/'))
}

var invocation = '''
import os, sys, tempfile
from pathlib import Path
from simulation.batch import main
with tempfile.TemporaryDirectory(prefix='approved-warmup-') as directory:
    platform = Path(directory) / 'platform.json'
    platform.write_text(os.environ['BATCH_PLATFORM_JSON'])
    sys.argv = ['simulation.batch', 'warmup', '--platform', str(platform), '--warmup-id', os.environ['BATCH_WARMUP_ID'], '--confirm-submission']
    main()
'''

resource job 'Microsoft.App/jobs@2025-07-01' = if (enabled) {
  name: jobName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${workerIdentityResourceId}': {} }
  }
  properties: {
    environmentId: managedEnvironmentId
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 180
      replicaRetryLimit: 0
      manualTriggerConfig: { parallelism: 1, replicaCompletionCount: 1 }
      registries: [{ server: registryServer, identity: workerIdentityResourceId }]
    }
    template: {
      containers: [{
        name: 'warmup-controller'
        image: '${registryServer}/physicalai-learning-worker@sha256:${workerImageSha256}'
        command: ['/srv/apps/learning_worker/.venv/bin/python', '-c', invocation]
        resources: { cpu: json('1.0'), memory: '2Gi' }
        env: [
          { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
          { name: 'BATCH_PLATFORM_JSON', value: string(platform) }
          { name: 'BATCH_WARMUP_ID', value: warmupId }
        ]
      }]
    }
  }
}

output controllerJobName string = enabled ? job.name : ''
