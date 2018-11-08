import os # TODO: remove this import?

def create_configuration(mapping, filename="config.py"):
    with open(filename, 'w') as f:
        for key, value in mapping.items():
            if isinstance(value, str):
                value = "\"" + value + "\""
            f.write("%s = %s\n" % (key, value))