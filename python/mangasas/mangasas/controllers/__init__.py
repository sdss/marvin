#!/usr/bin/python

from flask import request

# List all of the views to be automatically imported here.
__all__ = ["index", "search","current","plate","images","comments"]


def valueFromRequest(key=None, request=None, default=None, lower=False, list=False, boolean=False):
	''' Convenience function to retrieve values from HTTP requests (GET or POST).

		@param key Key to extract from HTTP request.
		@param request The HTTP request from Flask.
		@param default The default value if key is not found.
		@param lower Make the string lower case.
		@param list Check for a comma-separated list, returns a list of values.
	'''
	if request.method == 'POST':
		try:
			value = request.form[key]
			if boolean:
				return True
			if lower:
				value = value.lower()
			if list:
				value = value.split(",")
			return value
		except KeyError:
			return default
	else: # GET
		value = request.args.get(key, default)
		if value != None:
			if lower:
				value = value.lower()
			if list:
				value = value.split(",")
		return value

