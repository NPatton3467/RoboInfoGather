import yaml
import openai

def load_config(file_path: str) -> dict:
    """
    Load a YAML configuration file from the specified file path.
    
    Args:
        file_path (str): The path to the YAML file to be loaded.
        
    Returns:
        dict: The YAML file content parsed into a dictionary.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    # Open and read the YAML file safely
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError("The file specified does not exist.")
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML file: {exc}")

def openai_generate_completion(client: openai.OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    """
    Generate a text completion using the OpenAI model based on the input messages.
    
    Args:
        client (openai.OpenAI): The OpenAI client instance to use.
        model (str): The model ID of the OpenAI model to use for text completion.
        messages (list[dict[str, str]]): A list of message dictionaries to generate completions from.
        
    Returns:
        str: The generated message content from the model.
    
    Raises:
        AssertionError: If the response from the API does not contain the expected content.
    """
    # Ensure client is initialized
    if client is None or not isinstance(client, openai.OpenAI):
        raise ValueError("Invalid OpenAI client provided.")

    # Send request to OpenAI and receive the response
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Asserts to validate the API response structure
    assert response, "API response is empty."
    assert response.choices, "Response does not contain any choices."
    assert response.choices[0].message, "First choice does not contain a message."

    # Return the content of the first choice message
    return response.choices[0].message.content
