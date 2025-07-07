import logging
from os.path import join as pjoin

from debug_gym.agents.history_tracker import build_history_prompt
from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import is_subdirectory, show_line_number
from debug_gym.llms.base import LLM


@Toolbox.register()
class AgentTool(EnvironmentTool):
    name: str = "agent"
    examples = [
        """agent(query="") Executes query to send to this specialized agent.\n"""
    ]
    description = (
        "This tool allows to call a large language model specialized agent, and therefore to query for information and exploit capabilities in the model.\n"
        + "\n".join(examples)
    )
    arguments = {
        "query": {
            "type": ["string"],
            "description": "The request to the agent.",
        },
    }

    def __init__(
        self, history, llm_name=None, llm_config=None, llm_config_file_path=None
    ):
        self._history = history
        self.logger = logging.getLogger("agent_logger")
        self._llm = LLM.instantiate(
            llm_name=llm_name,
            llm_config=llm_config,
            llm_config_file_path=llm_config_file_path,
            logger=self.logger,
        )
        self.llm_config = self._llm.config
        super().__init__()

    def use(
        self,
        environment,
        query: str,
    ) -> Observation:
        messages = build_history_prompt(
            self._history.filter_out(actions=[None]),
            self._llm,
            False,
        )
        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )
        llm_response = self._llm.generate(messages, [], tool_choice="none")

        return Observation(
            self.name,
            llm_response.response,
        )

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        result.logger = logging.getLogger("agent_logger")
        result._llm = LLM.instantiate(
            llm_config=result.llm_config,
            logger=result.logger,
        )
        result._history = self._history.copy()
        return result
