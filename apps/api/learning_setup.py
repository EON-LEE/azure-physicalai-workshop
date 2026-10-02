from apps.api.learning_service import LearningService


def azure_learning_service(configuration, factory):
    resources = []
    gateway = None
    coach = None
    if configuration.learning_enabled and configuration.learning_worker_endpoint:
        from azure.identity import ManagedIdentityCredential

        from apps.api.learning_gateway import ManagedLearningGateway

        credential = ManagedIdentityCredential(client_id=str(configuration.azure_client_id))
        gateway = ManagedLearningGateway(
            configuration.learning_worker_endpoint, credential, configuration.learning_worker_scope
        )
        resources.extend((gateway, credential))
        if configuration.learning_coach_agent_name:
            from agents.learning_coach import FoundryLearningCoach

            coach = FoundryLearningCoach(
                configuration.foundry_project_endpoint,
                credential,
                configuration.learning_coach_agent_name,
                configuration.learning_coach_agent_version,
                configuration.model_timeout_seconds,
            )
            resources.insert(0, coach)
    return LearningService(
        factory,
        factory.store,
        gateway,
        gateway,
        gateway,
        enabled=configuration.learning_enabled,
        runtime=factory.bridge if gateway else None,
        coach=coach,
        bootstrap_principal_ids=configuration.learning_bootstrap_principal_ids,
        allowed_policy_types=configuration.learning_policy_types,
        reference_collections_enabled=configuration.learning_reference_collections_enabled,
        paused_training_enabled=configuration.learning_paused_training_enabled,
        paused_evaluation_enabled=configuration.learning_paused_evaluation_enabled,
        paused_release_enabled=configuration.learning_paused_release_enabled,
    ), resources
