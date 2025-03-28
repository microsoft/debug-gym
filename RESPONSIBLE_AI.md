# debug-gym 

`debug-gym` is a research framework that enables users to equip their debugging agents with interactive debugging tools to boost their debugging ability. `debug-gym` is designed to be easily extensible so users can design and customize their own tools for their specific use cases.  

### WHAT CAN DEBUG-GYM DO 

Most existing LLM-based debugging agents perform debugging in a static manner, i.e., given a buggy code, they take the error message returned from the terminal and use it as part of the prompt and hope the LLM could correct the bug; they do this iteratively until there is no error message being generated, or exhausts some step budget.  

We propose `debug-gym`, a framework that practitioners could equip their debugging agent with interactive debugging tools such as [`pdb`](https://docs.python.org/3/library/pdb.html) (in Python programming language). This enables a more human-like debugging behavior, where debugging agents could interact with `debug-gym`. Specifically, agents can set break points, navigate between break points, print necessary values at break points, or even generate small test functions on the fly while running the buggy code. By doing this, the agent could explore and gather necessary information from the much bigger hidden semantic space represented by the code itself and thus have a better understanding of where the bugs could be.  

We design `debug-gym` with two principles in mind, namely safety and extensibility. First, due to its interactive nature, `debug-gym` could receive harmful commands from LLM-based agents (which are prone to fabrication/hallucination) that interact with the file system in an undesirable way (e.g., deleting random files). To prevent such possibilities, we make sure all terminal interaction (i.e., pseudo terminals running debuggers and the test code) are inside a Docker container so the sandbox file system does not have access to the user’s true file system. Second, we understand that users could have different tools for tackling different debugging use cases, we thus make it straightforward for them to import or design their own tools in `debug-gym`.  

A detailed discussion of `debug-gym`, including how it was developed and tested, can be found in our technical report at: [https://arxiv.org/abs/2503.21557](https://arxiv.org/abs/2503.21557).

### INTENDED USES

In the current version of release, `debug-gym` is best suited for equipping LLM-based debugging agents with interactive debugging tools for debugging repository-level code written in Python programming language. 

`debug-gym` is being shared with the research community to facilitate reproduction of our results and foster further research in this area. The current version of `debug-gym` is not intended for commercialization, we see it purely as a research platform. 

`debug-gym` is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them. This includes experts in developing LLM-based agents as well as expert Python programmers.  

 

### OUT-OF-SCOPE USES 

In the current release, `debug-gym` is not well suited for debugging code repositories that are written in language other than Python.  

We do not recommend using `debug-gym` in commercial or real-world applications without further testing and development. It is being released for research purposes. 

`debug-gym` was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use. 

We do not recommend using `debug-gym` in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare). Python programs debugged using `debug-gym` should be verified by human experts in such use cases.  

 

## HOW TO GET STARTED

We will keep updating the [README.md](https://github.com/microsoft/debug-gym/blob/main/README.md) file in our repository.  
 

## EVALUATION 

`debug-gym` was evaluated in two ways: 

1. `debug-gym` is a simulation environment that helps users to develop their own AI debugging agent by enabling them to equip their agents with interactive debugging tools. As a result, `debug-gym` is evaluated by how reliable and robust the simulation environment is. We make sure to include unit tests for all core components in our environment implementation, every change to our codebase will trigger an automatic test on the entire codebase. The overall coverage of the unit test is higher than 85%, which is a high standard.  

2. With the simulation environment, we also provide the users some example agents that solely aim to demonstrate how the `debug-gym` APIs work. Because the example agents are minimal, we do not expect them to work particularly well on any evaluation dataset. We nonetheless conduct experiments and report the example agent’s performance on a few standard code generation and debugging benchmarks. 

A detailed discussion of our evaluation methods and results can be found in our technical report at: [https://arxiv.org/abs/2503.21557](https://arxiv.org/abs/2503.21557).

 

## LIMITATIONS 

`debug-gym` was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios. 

`debug-gym` was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language. 

Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs. 

`debug-gym` should not be used in highly regulated domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities. 

`debug-gym` inherits any biases, errors, or omissions produced by the backbone LLM users plug-in. Developers are advised to choose an appropriate base LLM/MLLM carefully, depending on the intended use case. 

`debug-gym` was not specifically designed to debug any particular security vulnerability, such as indirect prompt injection attacks (XPIA), and there has not been a systematic effort to ensure that `debug-gym` is protected from such attacks. We recommend that any system designed by users should be thoroughly tested and appropriate mitigations put in place. 

 

## BEST PRACTICES 

Better performance can be achieved by following our guide in our technical report: [https://arxiv.org/abs/2503.21557](https://arxiv.org/abs/2503.21557).  

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications: 

- [Blog post on responsible AI features in AOAI that were presented at Ignite 2023](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-new-ai-safety-amp-responsible-ai-features-in-azure/ba-p/3983686) 

- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview) 

- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note) 

- [OpenAI’s Usage policies](https://openai.com/policies/usage-policies) 

- [Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct) 

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with `debug-gym`.

 
## LICENSE 

We use the MIT license, please see the [license file](https://github.com/microsoft/debug-gym/blob/main/LICENSE).  


## CONTACT 

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us via [GitHub issues](https://github.com/microsoft/debug-gym/issues) or at debug-gym@microsoft.com . If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations. 

 