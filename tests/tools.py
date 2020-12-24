def create_file(path: str, content: str):
    """"
    Function to create a file for test
    """
    with open(path, "w") as f:
        f.write(content)
