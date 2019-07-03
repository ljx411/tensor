import matplotlib.pyplot as plt
import numpy as np
import re
import logging
logging.basicConfig(filename='123.log',filemode='a',format='%(asctime)s: %(message)s',datefmt='%Y-%M-%d %H:%M:%S',level=logging.INFO)
# def debug(level):
#     def wrap(func):
#         def inner_wrap(*args,**kwargs):
#             print('{level}: {name}'.format(level=level,name=func.__name__))
#             return func(*args,**kwargs)
#         return inner_wrap
#     return wrap
class debug():
    def __init__(self,level):
        self.level=level
    def __call__(self, func):
        def wrap(*args,**kwargs):
            logging.info('{level}: {name}'.format(level=self.level, name=func.__name__))
            return func(*args,**kwargs)
        return wrap
@debug(level='WARNING')
def say_hello(x):
    print(x)
    print('hello')
if __name__ == '__main__':
    say_hello(2)