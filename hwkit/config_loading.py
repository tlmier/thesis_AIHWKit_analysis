import json


def dump_sim_config(config):
  '''Serialize a simulation config dict to .sim file text.

  Args:
      config (_type_): config to save

  Returns:
      str: json data of config'''
  return json.dumps(config, indent=2)


def load_sim_config(path):
  '''Parse a .sim file (written by dump_sim_config()) back into a dict.
    Args:
      path (str): path where .net is saved

    Returns:
      str: json data of config'''
  with open(path) as f:
    return json.load(f)


def dump_network(network):
  '''Serialize a network config dict to .net file text.
    Args:
      network (_type_): network to safe 
    Returns:
      str: json data to safe
  '''
  return json.dumps(network, indent=2)


def load_network(path):
  '''Parse a .net file (written by dump_network()) back into a dict.
      Args:
        path (str): path where .net is saved

      Returns:
      str: json data of network'''
  with open(path) as f:
      return json.load(f)
