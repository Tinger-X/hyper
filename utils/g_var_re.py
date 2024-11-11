__all__ = ["Var"]

import logging


class Var:
    __dict = {}

    @staticmethod
    def get(key, default=None):
        return Var.__dict.get(key) or default

    @staticmethod
    def set(key, value):
        Var.__dict[key] = value

    @staticmethod
    def dict():
        return Var.__dict

    @staticmethod
    def have(key):
        return key in Var.__dict

    @staticmethod
    def delete(key):
        del Var.__dict[key]


if not Var.have("logger"):
    Var.set("logger", logging.getLogger("Hyper"))
