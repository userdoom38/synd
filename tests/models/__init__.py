import sys
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s\t] %(message)s'
)

console_handler.setFormatter(formatter)
log.addHandler(console_handler)

