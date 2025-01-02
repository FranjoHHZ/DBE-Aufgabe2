import logging
import time
from functools import wraps

def my_logger(orig_func):
    logging.basicConfig(filename=f'{orig_func.__name__}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = orig_func(*args, **kwargs)
        end_time = time.time()
        logging.info(f'{orig_func.__name__} ran in: {end_time - start_time:.4f} sec')
        return result
    return wrapper
