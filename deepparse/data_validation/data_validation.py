def is_whitespace_only_address(address: str) -> bool:
    """
    Validate if an address is composed of only whitespace.

    Args:
        address (str): A string address to validate.

    Return:
        Either or not, the address is composed only of whitespace or not.
    """
    return len(address.strip(" ").split()) == 0
