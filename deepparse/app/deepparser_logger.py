import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s; %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
