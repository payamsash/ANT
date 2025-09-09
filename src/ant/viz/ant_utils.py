# src/utils/ant_utils.py
# pylint: disable=no-member

import json
import argparse

def load_config(config_file_path):
    """
    Load the configuration from a JSON file.
    
    Args:
        config_file_path (str): Path to the JSON config file.
    
    Returns:
        dict: Loaded configuration as a dictionary.
    """
    try:
        with open(config_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in config file: {config_file_path}")
    
def parse_arguments():
    """
    Parse command-line arguments for the config file.
    
    Returns:
        argparse.Namespace: Parsed arguments containing the config file path.
    """
    parser = argparse.ArgumentParser(description="Main script for Alpha Wave generation and Rendering")
    parser.add_argument('config', type=str, help="Path to the main config JSON file")
    return parser.parse_args()