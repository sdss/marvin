#!/usr/bin/python

''' This file collects classes that implement useful design patterns in Python. '''


# used by memoize
import collections
import functools

#
# Taken from: http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
#
class memoize(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
	  self.func = func
	  self.cache = {}
   def __call__(self, *args):
	  if not isinstance(args, collections.Hashable):
		 # uncacheable. a list, for instance.
		 # better to not cache than blow up.
		 return self.func(*args)
	  if args in self.cache:
		 return self.cache[args]
	  else:
		 value = self.func(*args)
		 self.cache[args] = value
		 return value
   def __repr__(self):
	  '''Return the function's docstring.'''
	  return self.func.__doc__
   def __get__(self, obj, objtype):
	  '''Support instance methods.'''
	  return functools.partial(self.__call__, obj)

#
# See also: http://wiki.python.org/moin/PythonDecoratorLibrary#Singleton
#
def singleton(cls):
	'''
	This function implements the singleton design pattern in Python.
	To use it, simply import this file:
	
	from singleton import singleton
	
	and declare your class as such:
	
	@singleton
	class A(object):
		pass
	'''
	instance_container = []
	def getinstance():
		if not len(instance_container):
			instance_container.append(cls())
		return instance_container[0]
	return getinstance
	
	








