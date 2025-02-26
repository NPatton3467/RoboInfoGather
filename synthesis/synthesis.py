import openai
import yaml
import os
import sys
from pathlib import Path
from RoboInfoGather.synthesis.utils import load_config, openai_generate_completion
from RoboInfoGather.dsl import *

# Get the parent directory of the current file's grandparent
# This is often done to maintain a consistent import path regardless of where the script is run from
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path for module resolution
sys.path.append(str(parent_dir))


class SynthesisResult:
    """
    A class to encapsulate the results of a model's output, specifically
    designed to handle code synthesis.
    """

    def __init__(self, model_output: str) -> None:
        """
        Initialize the SynthesisResult with the model's output.

        Args:
            model_output (str): The raw code output from the LLM.
        """
        self.model_output = model_output

    def get_model_output(self) -> str:
        """
        Returns the raw output of the LLM.
        """
        return self.model_output

    def get_program_object(self):
        """
        Execute the LLM's output and retrieve the synthesized program object.
        This method assumes the model output contains a Python code definition for an object named 'program'.
        """
        local_scope = {}
        
        try:
            exec(self.model_output, globals(), local_scope)
        except Exception as e:
            raise RuntimeError(f"Failed to execute model output: {str(e)}")

        if "program" not in local_scope:
            raise KeyError("'program' not defined in the LLM's output")

        return local_scope["program"]


class Synthesizer:
    """
    A class for synthesizing responses using OpenAI's API based on given examples and preambles.

    Attributes:
        config (dict): Synthesis configuration settings.
        model (str): LLM name retrieved from the configuration.
        examples (dict): Synthesis examples.
        preamble (str): Predefined text loaded from a file to prepend to queries.
        client (openai.Client): The OpenAI API client.
    """

    def __init__(
        self, config_file_path: str, example_file_path: str, preamble_file_path: str
    ) -> None:
        """
        Initializes the Synthesizer with specified file paths for configuration,
        examples, and preamble.
        """
        self.config = load_config(config_file_path)
        self.model = self.config["model"]["name"]

        with open(example_file_path, "r") as file:
            self.examples = yaml.safe_load(file)

        with open(preamble_file_path, "r") as file:
            self.preamble = file.read()

        with open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r') as f:
            api_key = f.read()

        self.client = openai.Client(
            api_key=api_key,
            organization=self.config["identification"]["openai_org"],
        )

    def generate(self, input_query: str) -> SynthesisResult:
        """
        Generates a SynthesisResult for the input_query, using the stored examples and preamble.

        Args:
            input_query (str): The user input query of the desired information gathering task.

        Returns:
            SynthesisResult: The synthesis result.
        """

        # Add the preamble.
        prompt = self.preamble
        # Add examples.
        for example in self.examples["Examples"]:
            ex_prompt = example["prompt"]
            ex_prog = example["program"]
            prompt += f"\n{ex_prompt}:\n{ex_prog}\n"

        # Add the user query to the message.
        prompt += input_query

        messages = [{"role": "system", "content": prompt, "temperature":0.25}]

        # Call the LLM and get the textual response.
        response = openai_generate_completion(
            client=self.client, model=self.model, messages=messages
        )
        print("GPT Query Gen Response:\n", response)
        response = response.lstrip(' ```python')
        response = response.rstrip('```')
        print("GPT Query Gen Response:\n", response)
        return SynthesisResult(response)


if __name__ == "__main__":
    # Create a synthesizer instance with paths to necessary files.
    synthesizer = Synthesizer(
        config_file_path="synthesis/synthesis_config.yaml",
        example_file_path="synthesis/examples.yaml",
        preamble_file_path="synthesis/prompt_preamble.txt",
    )

    # List of test prompts for the synthesizer to generate programs from.
    test_prompts = [
        "Count the number of cups",
        "Find a vacant conference room with a whiteboard",
        "Which kitchens have coffee makers?",
        "Where’s my toolbox?",
        "How much free counter space is there?",
        "Where are my keys?",
        "Where is my tallest cup?",
        "Is my smallest cup clean?",
        "Where is my glass of water?",
    ]

    # Necessary import for execution of synthesized programs.
    from dsl import *

    # Process each prompt and print the synthesized program.
    for test_prompt in test_prompts:
        result = synthesizer.generate(test_prompt)
        synthesized_program = result.get_program_object()
        print("*" * 80)
        print("Prompt:", test_prompt)
        print("Program:", synthesized_program.pretty_str())
        print()
