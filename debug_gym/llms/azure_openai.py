import logging
import time

from azure.core.exceptions import ClientAuthenticationError
from openai import AzureOpenAI, PermissionDeniedError

from debug_gym.llms.constants import LLM_API_KEY_PLACEHOLDER, LLM_SCOPE_PLACEHOLDER
from debug_gym.llms.openai import OpenAILLM

# Set logging level down to WARNING for endpoint queries.
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


class AzureOpenAILLM(OpenAILLM):

    CLIENT_MAX_AGE_SECONDS = 3600  # 1 hour
    AUTH_RETRY_DELAY_SECONDS = 5

    def __init__(self, model_name, logger=None, llm_config=None, llm_config_file=None):
        super().__init__(model_name, logger, llm_config, llm_config_file)
        self._client = None
        self._client_created_at = 0

    @property
    def client(self):
        now = time.time()
        reason = None

        if self._client is None:
            reason = "initialize"
        elif now - self._client_created_at >= self.CLIENT_MAX_AGE_SECONDS:
            reason = f"age>{self.CLIENT_MAX_AGE_SECONDS}s"

        if reason:
            self.logger.debug("Creating Azure OpenAI client (%s)", reason)
            self._client = self._get_azure_oai_client()
            self._client_created_at = time.time()

        return self._client

    def _get_azure_oai_client(self):
        """
        Returns the Azure OpenAI client. This will use either an API key or Azure Identity.
        If the first attempt with Default and Managed Identity credentials fails,
        try again using only CliCredential (az login).

        Raises ValueError: If neither an API key nor a scope is provided in the configuration.
        """
        api_key = self.config.api_key
        scope = self.config.scope
        kwargs = {
            "azure_endpoint": self.config.endpoint,
            "api_version": self.config.api_version,
            "timeout": None,
        }
        if api_key not in [LLM_API_KEY_PLACEHOLDER, None]:  # api key
            kwargs["api_key"] = api_key
            aoai_client = AzureOpenAI(**kwargs)
        elif scope not in [LLM_SCOPE_PLACEHOLDER, None]:  # az login
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )

            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    DefaultAzureCredential(),
                    ManagedIdentityCredential(),
                    AzureCliCredential(),
                ),
                scope,
            )
            kwargs["azure_ad_token_provider"] = credential
            aoai_client = AzureOpenAI(**kwargs)
            try:
                aoai_client.models.list()  # test the connection
            except PermissionDeniedError:
                # if auth works but permission denied, try AzureCliCredential
                self.logger.warning(
                    "Permission denied for DefaultAzureCredential. Trying AzureCliCredential."
                )
                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    AzureCliCredential(), scope
                )
                aoai_client = AzureOpenAI(**kwargs)
        else:
            raise ValueError(
                "Invalid LLM configuration for AzureOpenAI. "
                "Please provide an `api_key or `scope` in the configuration."
            )
        return aoai_client

    def need_to_be_retried(self, exception) -> bool:
        need_to_retry = super().need_to_be_retried(exception)

        if isinstance(exception, ClientAuthenticationError):
            self.logger.info(
                "Client authentication error detected, refreshing Azure OpenAI client"
            )
            self._invalidate_client_cache()
            need_to_retry = True
            time.sleep(self.AUTH_RETRY_DELAY_SECONDS)

        return need_to_retry

    def _invalidate_client_cache(self):
        self._client = None
        self._client_created_at = 0
